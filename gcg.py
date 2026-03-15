"""GCG attack for harm classifiers.

Thin wrapper around the canonical implementation in freeze-lat/attacks/gcg_fast.py.
Adds:
  - Multi-model loading (SBERT + AutoModelForSequenceClassification via --model_type)
  - Incremental saving via on_example_complete (enabled by default)
  - OOM-resilient candidate eval via find_executable_batch_size (enabled by default)
  - Backward-compat helpers for pgd_attack.py (get_word_embeddings, forward_with_embeds, etc.)
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safetensors.torch import load_file

# ---- Import canonical GCG implementation ----
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ibp_huggingface', 'attacks'))
from gcg_fast import (  # noqa: E402
    SBERTwithClassifier,
    BERTGCGConfig,
    BERTGCGOptimizer,
    get_nonascii_toks_bert,
    sample_control_bert,
    _get_gcg_backbone,
    _gcg_pool,
    create_adversarial_suffix_bert_gcg,
    execute_gcg_attack_sbert,
    DEVICE,
    load_from_disk,
)

# ---- Backward-compat helpers for pgd_attack.py ----
# pgd_attack.py imports these; they use the explicit model_type approach
# rather than gcg_fast's auto-detection via _get_gcg_backbone.


def get_word_embeddings(model, model_type):
    """Return the word embedding layer for the given model architecture."""
    if model_type == "sbert":
        return model.backbone.embeddings.word_embeddings
    elif model_type == "bert":
        return model.bert.embeddings.word_embeddings
    elif model_type == "roberta":
        return model.roberta.embeddings.word_embeddings
    elif model_type == "deberta":
        return model.deberta.embeddings.word_embeddings
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_model_dtype(model, model_type):
    """Return the dtype of the model's parameters."""
    return get_word_embeddings(model, model_type).weight.dtype


def forward_with_embeds(model, model_type, inputs_embeds, attention_mask):
    """Compute logits from inputs_embeds for any supported architecture."""
    if model_type == "sbert":
        outputs = model.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        lhs = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(lhs.size()).float()
        pooled = torch.sum(lhs * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
        return model.classifier(pooled)
    else:
        return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits


def forward_with_ids(model, model_type, input_ids, attention_mask):
    """Compute logits from token IDs for any supported architecture."""
    if model_type == "sbert":
        return model(input_ids=input_ids, attention_mask=attention_mask)
    else:
        return model(input_ids=input_ids, attention_mask=attention_mask).logits


def _model_forward_logits(model, **inputs):
    """Forward pass returning logits, handling both SBERT and HF model outputs."""
    outputs = model(**inputs)
    if isinstance(outputs, tuple):
        return outputs[0]
    elif hasattr(outputs, 'logits'):
        return outputs.logits
    return outputs


# ---- Model loading ----

def load_model(model_dir, model_type, device=DEVICE):
    """Load a model by type. Returns (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if model_type == "sbert":
        model = SBERTwithClassifier(model_dir, num_labels=2).to(device)
        st_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(st_path):
            state_dict = load_file(st_path)
        else:
            bin_path = os.path.join(model_dir, "pytorch_model.bin")
            state_dict = torch.load(bin_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GCG Attack on SBERT/BERT/RoBERTa/DeBERTa")
    parser.add_argument("--base_model_dir", type=str,
                        default="/home/dcheung2/new/ibp_huggingface/sbert/models/sbert_combined_binary",
                        help="Path to the trained model directory")
    parser.add_argument("--model_type", type=str, default="sbert",
                        choices=["sbert", "bert", "roberta", "deberta"],
                        help="Model architecture type")
    parser.add_argument("--num_examples", type=int, default=100,
                        help="Max number of examples to attack")
    parser.add_argument("--dataset_path", type=str,
                        default="/home/dcheung2/new/ibp_huggingface/sbert/datasets/aegis_preprocessed",
                        help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of examples to attack simultaneously")
    parser.add_argument("--search_width", type=int, default=2048,
                        help="Number of candidate suffixes per step")
    parser.add_argument("--num_restarts", type=int, default=1,
                        help="Number of random restarts per example")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose debug output")
    parser.add_argument("--output", type=str, default="gcg_results.pt",
                        help="Path to save attack results")
    args = parser.parse_args()

    print(f"Loading model from {args.base_model_dir} (type={args.model_type})")
    model, tokenizer = load_model(args.base_model_dir, args.model_type, DEVICE)

    # --- DATASET LOADING ---
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.map(lambda x: {"label": 0 if x["label"] == 0 else 1})

    if "test" in dataset:
        target_dataset = dataset["test"]
    else:
        print("Warning: No test split found, falling back to train.")
        target_dataset = dataset["train"]

    target_dataset = target_dataset.filter(lambda x: x["label"] == 1)

    # --- Pre-filter examples where model is correct ---
    print(f"\nPre-filtering {min(len(target_dataset), args.num_examples)} examples...")
    attack_examples = []

    for i in range(min(len(target_dataset), args.num_examples)):
        test_example = target_dataset[i]
        input_text = test_example.get("text", test_example.get("sentence"))
        true_label = test_example["label"]

        if not input_text:
            continue

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512 - 20).to(DEVICE)
        with torch.inference_mode():
            logits = _model_forward_logits(model, **inputs)
            original_pred = logits.argmax(dim=1).item()

        if original_pred != true_label:
            continue

        attack_examples.append((input_text, original_pred))

    total_attacked = len(attack_examples)
    print(f"Found {total_attacked} correctly-classified examples to attack.")

    # --- BATCHED ATTACK (incremental saving + OOM resilience enabled) ---
    config = BERTGCGConfig(
        num_steps=250,
        search_width=args.search_width,
        topk=min(512, args.search_width),
        num_restarts=args.num_restarts,
        verbose=args.verbose,
        oom_resilient_eval=True,
        early_stop=False,
    )
    optimizer = BERTGCGOptimizer(model, tokenizer, config, DEVICE)

    all_attack_examples = [(text, 1 - orig_pred) for text, orig_pred in attack_examples]

    # Incremental save: write results to disk after each example completes
    save_data = []

    def _on_complete(idx, result):
        input_text, orig_pred = attack_examples[idx]
        save_data.append({
            "example_idx": idx,
            "input_text": input_text,
            "original_label": orig_pred,
            "target_label": 1 - orig_pred,
            "suffix_ids": result["best_ids"].cpu(),
            "suffix_text": result["best_suffix"],
            "succeeded": result["succeeded"],
            "steps": result["steps"],
            "restart": result.get("restart", 0),
        })
        torch.save(save_data, args.output)

    try:
        results_list = optimizer.step_batched(
            all_attack_examples, batch_size=args.batch_size,
            on_example_complete=_on_complete,
        )
    except Exception as e:
        print(f"Attack failed: {e}")
        import traceback; traceback.print_exc()
        results_list = [None] * total_attacked

    # Final summary
    num_succ = sum(1 for e in save_data if e["succeeded"])
    print(f"\nTotal successful attacks: {num_succ} / {total_attacked}")
    print(f"Saved {len(save_data)} results to {args.output}")

    # --- VERIFY SAVED FILE ---
    loaded = torch.load(args.output, weights_only=False)
    verify_pass = 0
    verify_fail = 0
    for entry in loaded:
        if not entry["succeeded"]:
            continue
        full_text = entry["input_text"] + " " + entry["suffix_text"]
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.inference_mode():
            logits = _model_forward_logits(model, **inputs)
            pred = logits.argmax(dim=1).item()
        if pred == entry["target_label"]:
            verify_pass += 1
        else:
            verify_fail += 1
            print(f"  [VERIFY FAIL] Example {entry['example_idx']}: pred={pred}, target={entry['target_label']}")

    print(f"Verification: {verify_pass}/{verify_pass + verify_fail} saved attacks still succeed after reload")
