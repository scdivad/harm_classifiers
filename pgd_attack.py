"""PGD embedding-space attack on BERT/RoBERTa/DeBERTa harm classifiers.

Adapted from obfuscated-activations/train_time_experiments/src/attacks.py.
Finds continuous perturbations in embedding space that flip classification
from harmful (1) to safe (0).

Unlike GCG (discrete token search), this optimizes directly in continuous
embedding space using projected gradient descent with L2 constraints.
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

import datasets as hf_datasets
if not hasattr(hf_datasets, 'load_from_disk'):
    current_dir = os.getcwd()
    sys.path = [p for p in sys.path if p != current_dir and p != '']
    import importlib
    importlib.reload(hf_datasets)
    from datasets import load_from_disk
else:
    from datasets import load_from_disk

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safetensors.torch import load_file

from gcg import (
    SBERTwithClassifier,
    get_word_embeddings,
    get_model_dtype,
    forward_with_embeds,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Enable TF32 for H100/A100
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================
# Adversary classes (faithful to obfuscated-activations)
# ============================================================

class GDAdversary(nn.Module):
    """Per-example gradient descent adversary with L2 epsilon ball constraint.

    Adds a learnable perturbation to embeddings at masked positions.
    Faithful to obfuscated-activations GDAdversary.
    """

    def __init__(self, dim, epsilon, attack_mask, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.epsilon = epsilon

        batch_size, seq_len = attack_mask.shape
        self.attack = nn.Parameter(
            torch.zeros(batch_size, seq_len, dim, device=device, dtype=dtype or torch.float32)
        )
        nn.init.kaiming_uniform_(self.attack)
        self.clip_attack()
        self.attack_mask = attack_mask

    def forward(self, x):
        if self.device is None or self.device != x.device:
            self.device = x.device
            self.attack.data = self.attack.data.to(self.device)
            self.attack_mask = self.attack_mask.to(self.device)

        perturbed_acts = x[self.attack_mask[:, :x.shape[1]]] + self.attack[
            :, :x.shape[1]
        ][self.attack_mask[:, :x.shape[1]]].to(x.dtype)
        x = x.clone()
        x[self.attack_mask[:, :x.shape[1]]] = perturbed_acts
        return x

    def clip_attack(self):
        with torch.no_grad():
            norms = torch.norm(self.attack, dim=-1, keepdim=True)
            scale = torch.clamp(norms / self.epsilon, min=1)
            self.attack.div_(scale)


def zero_nan_grads(model):
    flag = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                flag = True
                p.grad[torch.isnan(p.grad)] = 0.0
    if flag:
        print("NaN gradient detected. Setting to zero.")


# ============================================================
# PGD attack for BERT classifiers
# ============================================================

def pgd_attack(
    model,
    model_type,
    tokenizer,
    input_ids,
    attention_mask,
    target_label,
    epsilon=50.0,
    learning_rate=1e-4,
    pgd_iterations=250,
    l2_regularization=0,
    clip_grad=1.0,
    device=DEVICE,
    verbose=False,
):
    """Run PGD embedding-space attack on a single batch of examples.

    Optimizes a continuous perturbation in embedding space to flip
    classification to target_label.

    Args:
        model: Classifier model.
        model_type: One of 'sbert', 'bert', 'roberta', 'deberta'.
        tokenizer: Corresponding tokenizer.
        input_ids: [batch_size, seq_len] token IDs.
        attention_mask: [batch_size, seq_len] attention mask.
        target_label: Target class (int) to flip towards.
        epsilon: L2 norm bound per token position.
        learning_rate: Adam learning rate.
        pgd_iterations: Number of PGD steps.
        l2_regularization: L2 penalty coefficient.
        clip_grad: Gradient clipping threshold.
        device: torch device.
        verbose: Print progress.

    Returns:
        Dict with attack results per example.
    """
    model.eval()
    embedding_layer = get_word_embeddings(model, model_type)
    batch_size, seq_len = input_ids.shape
    hidden_dim = embedding_layer.embedding_dim
    _dtype = get_model_dtype(model, model_type)

    # All positions are attackable (full attack mask)
    attack_mask = attention_mask.bool().to(device)

    # Create adversary
    adversary = GDAdversary(
        dim=hidden_dim,
        epsilon=epsilon,
        attack_mask=attack_mask,
        device=device,
        dtype=_dtype,
    )

    # Get clean embeddings (frozen)
    with torch.no_grad():
        clean_embeds = embedding_layer(input_ids).clone()  # [B, seq_len, H]

    # Optimizer for adversary parameters only
    params = list(adversary.parameters())
    adv_optim = torch.optim.AdamW(params, lr=learning_rate)

    target_labels = torch.full((batch_size,), target_label, device=device, dtype=torch.long)

    losses_over_time = []

    for step in tqdm(range(pgd_iterations), desc="PGD", disable=not verbose):
        adv_optim.zero_grad()

        # Apply perturbation to clean embeddings
        perturbed_embeds = adversary(clean_embeds.clone())

        # Forward through model
        logits = forward_with_embeds(
            model, model_type, perturbed_embeds, attention_mask.float()
        )

        # Cross-entropy loss towards target label
        loss = F.cross_entropy(logits, target_labels)

        loss.backward()

        losses = {"ce": loss.item()}

        # L2 regularization (faithful to reference)
        if l2_regularization:
            reg_loss = torch.norm(adversary.attack)
            num_el = adversary.attack.numel()
            (l2_regularization * reg_loss / np.sqrt(num_el)).backward()
            losses["l2_norm"] = reg_loss.item() / np.sqrt(num_el)

        # Gradient cleanup
        zero_nan_grads(adversary)

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(params, clip_grad)

        adv_optim.step()
        adversary.clip_attack()

        losses_over_time.append(copy.deepcopy(losses))

        if verbose and (step + 1) % 10 == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                n_flipped = (preds == target_label).sum().item()
            print(f"  Step {step+1}: loss={loss.item():.4f} flipped={n_flipped}/{batch_size}")

    # Final evaluation
    with torch.no_grad():
        perturbed_embeds = adversary(clean_embeds.clone())
        logits = forward_with_embeds(
            model, model_type, perturbed_embeds, attention_mask.float()
        )
        final_preds = logits.argmax(dim=1)
        final_probs = F.softmax(logits, dim=1)

    return {
        "final_preds": final_preds.cpu(),
        "final_probs": final_probs.cpu(),
        "succeeded": (final_preds == target_label).cpu(),
        "losses": losses_over_time,
        "perturbation_norms": torch.norm(adversary.attack.data, dim=-1).cpu(),  # [B, seq_len]
    }


def run_pgd_attack(
    model,
    model_type,
    tokenizer,
    dataset,
    num_examples=100,
    batch_size=32,
    epsilon=50.0,
    pgd_iterations=250,
    learning_rate=1e-4,
    save_path=None,
    device=DEVICE,
):
    """Run PGD attack on harmful examples from a dataset.

    Args:
        model: Classifier model (already on device).
        model_type: One of 'sbert', 'bert', 'roberta', 'deberta'.
        tokenizer: Corresponding tokenizer.
        dataset: HF dataset with 'text' and 'label' fields.
        num_examples: Number of harmful examples to attack.
        batch_size: Batch size for PGD.
        epsilon: L2 norm bound.
        pgd_iterations: Number of PGD steps.
        learning_rate: Adam learning rate.
        save_path: Path to save results.
        device: torch device.

    Returns:
        Dict with attack metrics.
    """
    model.eval()

    # Get test split
    if "test" in dataset:
        test_data = dataset["test"]
    else:
        print("Warning: No test split found, falling back to train.")
        test_data = dataset["train"]

    # Binarize labels
    test_data = test_data.map(lambda x: {"label": 0 if x["label"] == 0 else 1})

    # Filter harmful examples
    harmful = test_data.filter(lambda x: x["label"] == 1)

    # Pre-filter: only attack examples the model classifies correctly
    print(f"Pre-filtering {min(len(harmful), num_examples)} harmful examples...")
    attack_texts = []
    for i in range(min(len(harmful), num_examples)):
        text = harmful[i].get("text", harmful[i].get("sentence"))
        if not text:
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.inference_mode():
            out = model(**inputs)
            logits = out.logits if hasattr(out, 'logits') else (out[0] if isinstance(out, tuple) else out)
            pred = logits.argmax(dim=1).item()
        if pred == 1:  # correctly classified as harmful
            attack_texts.append(text)

    num_correct = len(attack_texts)
    print(f"Found {num_correct} correctly-classified harmful examples to attack.")

    if num_correct == 0:
        return {"robust_acc": 0.0, "num_total": num_examples, "num_attacked": 0,
                "num_succeeded": 0, "asr": 0.0}

    # Run PGD in batches
    all_results = []
    total_succeeded = 0

    for start in range(0, num_correct, batch_size):
        end = min(start + batch_size, num_correct)
        batch_texts = attack_texts[start:end]
        bs = len(batch_texts)

        batch_inputs = tokenizer(
            batch_texts, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(device)

        result = pgd_attack(
            model=model,
            model_type=model_type,
            tokenizer=tokenizer,
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            target_label=0,  # flip to safe
            epsilon=epsilon,
            learning_rate=learning_rate,
            pgd_iterations=pgd_iterations,
            device=device,
            verbose=True,
        )

        n_succ = result["succeeded"].sum().item()
        total_succeeded += n_succ
        print(f"Batch {start//batch_size + 1}: {n_succ}/{bs} succeeded")

        for j in range(bs):
            all_results.append({
                "example_idx": start + j,
                "input_text": batch_texts[j],
                "original_label": 1,
                "target_label": 0,
                "succeeded": result["succeeded"][j].item(),
                "final_pred": result["final_preds"][j].item(),
                "final_probs": result["final_probs"][j].tolist(),
                "perturbation_norm_mean": result["perturbation_norms"][j].mean().item(),
            })

    asr = total_succeeded / num_correct
    print(f"\nPGD Attack Results: {total_succeeded}/{num_correct} succeeded (ASR={asr:.4f})")

    if save_path:
        torch.save(all_results, save_path)
        print(f"Saved {len(all_results)} results to {save_path}")

    return {
        "num_total": num_examples,
        "num_attacked": num_correct,
        "num_succeeded": total_succeeded,
        "asr": asr,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGD embedding-space attack on harm classifiers")
    parser.add_argument("--base_model_dir", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["sbert", "bert", "roberta", "deberta"])
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epsilon", type=float, default=50.0, help="L2 norm bound per token")
    parser.add_argument("--pgd_iterations", type=int, default=250)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="pgd_results.pt")
    args = parser.parse_args()

    print(f"Loading model from {args.base_model_dir} (type={args.model_type})")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)

    if args.model_type == "sbert":
        model = SBERTwithClassifier(args.base_model_dir, num_labels=2).to(DEVICE)
        st_path = os.path.join(args.base_model_dir, "model.safetensors")
        if os.path.exists(st_path):
            state_dict = load_file(st_path)
        else:
            bin_path = os.path.join(args.base_model_dir, "pytorch_model.bin")
            state_dict = torch.load(bin_path, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.base_model_dir).to(DEVICE)

    model.eval()

    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)

    results = run_pgd_attack(
        model=model,
        model_type=args.model_type,
        tokenizer=tokenizer,
        dataset=dataset,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
        epsilon=args.epsilon,
        pgd_iterations=args.pgd_iterations,
        learning_rate=args.learning_rate,
        save_path=args.output,
        device=DEVICE,
    )
