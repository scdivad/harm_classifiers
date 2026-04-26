import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_from_disk
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, '..', 'freeze-lat', 'sbert'),
              os.path.join(_here, '..', 'ibp_huggingface', 'sbert')):
    if os.path.exists(os.path.join(_cand, 'SBERTwithClassifier.py')):
        sys.path.insert(0, _cand)
        break
from SBERTwithClassifier import SBERTwithClassifier

LABEL_NAMES = ["safe", "unsafe"]

DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 0
DEFAULT_LR_WARMUP = 0.1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1


def get_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer on Aegis 2.0 prompt-only binary safety."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "sbert"])
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to DatasetDict saved by build_aegis_v2_prompt_dataset.py")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--lr_warmup", type=float, default=DEFAULT_LR_WARMUP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--truncate_left", type=str, default="false",
                        choices=["true", "false"])
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    safe_mask = labels == 0
    unsafe_mask = labels == 1
    safe_acc = (preds[safe_mask] == 0).mean() if safe_mask.sum() > 0 else float("nan")
    unsafe_acc = (preds[unsafe_mask] == 1).mean() if unsafe_mask.sum() > 0 else float("nan")
    return {
        "accuracy": float(acc),
        "safe_accuracy": float(safe_acc),
        "unsafe_accuracy": float(unsafe_acc),
    }


def train_sbert(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_from_disk(args.dataset)
    eval_split = "test" if "test" in dataset else "validation"

    train_labels = torch.tensor(dataset["train"]["label"])
    eval_labels = torch.tensor(dataset[eval_split]["label"])
    print(f"Training set: {len(train_labels)} examples  "
          f"(safe={( train_labels==0).sum()}, unsafe={(train_labels==1).sum()})")

    model = SBERTwithClassifier(args.model, num_labels=2).to(device)
    for param in model.parameters():
        param.requires_grad = True

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("\nTokenizing data...")
    train_enc = tokenizer([str(t) for t in dataset["train"]["text"]],
                          padding=True, truncation=True,
                          max_length=args.max_length, return_tensors="pt")
    eval_enc = tokenizer([str(t) for t in dataset[eval_split]["text"]],
                         padding=True, truncation=True,
                         max_length=args.max_length, return_tensors="pt")

    train_ds = TensorDataset(train_enc["input_ids"], train_enc["attention_mask"], train_labels)
    eval_ds = TensorDataset(eval_enc["input_ids"], eval_enc["attention_mask"], eval_labels)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = correct = total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for input_ids, attention_mask, labels in pbar:
            input_ids, attention_mask, labels = (
                input_ids.to(device), attention_mask.to(device), labels.to(device))
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.2%}"})
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
              f"Train Acc={correct/total:.2%}")

    print(f"\nEvaluating on {eval_split} set...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in tqdm(eval_loader, desc="Eval"):
            logits = model(input_ids.to(device), attention_mask.to(device))
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    print("\n" + "=" * 40)
    print(f"FINAL RESULTS — {args.model} (sbert)")
    print("=" * 40)
    print(f"  Overall Accuracy: {(all_preds == all_labels).mean():.2%}")
    print(f"  Safe accuracy:    {(all_preds[all_labels==0] == 0).mean():.2%}")
    print(f"  Unsafe accuracy:  {(all_preds[all_labels==1] == 1).mean():.2%}")
    print("=" * 40)

    os.makedirs(args.save_dir, exist_ok=True)
    save_file(model.state_dict(), os.path.join(args.save_dir, "model.safetensors"))
    model.backbone.config.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print(f"\nModel saved to {args.save_dir}")


def main():
    args = get_args()

    if args.model_type == "sbert":
        train_sbert(args)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.truncate_left == "true":
        tokenizer.truncation_side = "left"
        print(f"Left truncation enabled — model will see the last {args.max_length} tokens")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label={0: "safe", 1: "unsafe"},
        label2id={"safe": 0, "unsafe": 1},
    )

    dataset = load_from_disk(args.dataset)
    eval_split = "test" if "test" in dataset else "validation"

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.lr_warmup,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=10,
        seed=args.seed,
        report_to="wandb" if args.wandb_run_name else "none",
        run_name=args.wandb_run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized[eval_split],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate()
    print("\n" + "=" * 40)
    print(f"FINAL RESULTS — {args.model}")
    print("=" * 40)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("=" * 40)

    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()
