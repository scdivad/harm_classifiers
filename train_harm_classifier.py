import argparse
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

LABEL_NAMES = [
    "Safe",
    "Criminal Planning/Confessions",
    "PII/Privacy",
    "Sexual",
    "Harassment",
    "Guns and Illegal Weapons",
    "Violence",
    "Controlled/Regulated Substances",
]

DEFAULT_DATASET = "datasets/aegis"
DEFAULT_EPOCHS = 5
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 0
DEFAULT_LR_WARMUP = 0.1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 1


def get_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer model on the AEGIS harmfulness dataset."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name (e.g. bert-base-uncased)")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--lr_warmup", type=float, default=DEFAULT_LR_WARMUP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--multiclass", action="store_true",
                        help="Use all 8 AEGIS categories instead of binary (safe/harmful)")
    parser.add_argument("--truncate_left", type=str, default="auto",
                        choices=["true", "false", "auto"],
                        help="Truncate from the left so model sees last max_length tokens. "
                             "'auto' enables for hhrlhf datasets (default: auto)")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser.parse_args()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    # Binary accuracy (safe vs harmful) regardless of multiclass setup
    binary_acc = ((preds == 0) == (labels == 0)).mean()
    return {"accuracy": acc, "binary_accuracy": binary_acc}


def main():
    args = get_args()

    binary = not args.multiclass
    num_labels = 2 if binary else len(LABEL_NAMES)
    label_names = ["Safe", "Harmful"] if binary else LABEL_NAMES

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Left truncation: keep the last max_length tokens (useful for conversational
    # datasets like HH-RLHF where the assistant response is at the end).
    truncate_left = args.truncate_left
    if truncate_left == "auto":
        truncate_left = "hhrlhf" in args.dataset.lower().replace("-", "").replace("_", "")
    else:
        truncate_left = truncate_left == "true"
    if truncate_left:
        tokenizer.truncation_side = "left"
        print(f"Left truncation enabled — model will see the last {args.max_length} tokens")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
    )

    dataset = load_from_disk(args.dataset)

    if binary:
        dataset = dataset.map(
            lambda x: {"label": 0 if x["label"] == 0 else 1},
        )

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
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Final evaluation
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
