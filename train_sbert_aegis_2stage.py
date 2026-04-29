"""Replicate the original SBERT training recipe for aegis_sbert.

Stage 1: Fine-tune distilbert-base-uncased as a SentenceTransformer with
         BatchHardSoftMarginTripletLoss using 8-class aegis labels for richer
         triplet structure.
Stage 2: Wrap the trained encoder in SBERTwithClassifier (mean pool + linear
         head) and fit the head with CE loss on binary labels.

Saves output in SBERTwithClassifier format (model.safetensors + config + tokenizer)
compatible with downstream LAT/MALATANG code.

Usage (defaults match the orig_sbert.py recipe except the head is NN not SVM):
  python train_sbert_aegis_2stage.py \
      --dataset /home/dcheung2/new/ibp_huggingface/datasets/aegis_sbert \
      --save_dir models/sbert_aegis_binary
"""
import argparse
import os
import sys
import shutil
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_from_disk, Dataset
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoTokenizer

from sentence_transformers import SentenceTransformer
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.losses import BatchHardSoftMarginTripletLoss

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(_here, '..', 'freeze-lat', 'sbert'),
              os.path.join(_here, '..', 'ibp_huggingface', 'sbert')):
    if os.path.exists(os.path.join(_cand, 'SBERTwithClassifier.py')):
        sys.path.insert(0, _cand)
        break
from SBERTwithClassifier import SBERTwithClassifier


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="distilbert/distilbert-base-uncased")
    p.add_argument("--dataset", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--stage1_epochs", type=int, default=10)
    p.add_argument("--stage1_lr", type=float, default=2e-5)
    p.add_argument("--stage1_batch", type=int, default=16)
    p.add_argument("--stage1_warmup", type=float, default=0.1)
    p.add_argument("--stage2_epochs", type=int, default=10)
    p.add_argument("--stage2_lr", type=float, default=2e-5)
    p.add_argument("--stage2_batch", type=int, default=32)
    p.add_argument("--stage2_freeze_backbone", action="store_true",
                   help="Freeze the encoder during stage 2 (only train head).")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--triplet_label_mode", choices=["multiclass", "binary"],
                   default="multiclass",
                   help="multiclass: use 8-class aegis labels for triplet "
                        "(richer signal). binary: collapse 0=safe, 1=harmful.")
    return p.parse_args()


# ── Stage 1: SentenceTransformer triplet fine-tune ──────────────────

def stage1_triplet(args, stage1_dir):
    print("\n" + "="*60)
    print(f"STAGE 1: triplet fine-tune ({args.triplet_label_mode} labels)")
    print("="*60)

    raw = load_from_disk(args.dataset)
    train_texts = [str(t) for t in raw["train"]["text"]]
    raw_labels = list(raw["train"]["label"])

    if args.triplet_label_mode == "binary":
        labels = [0 if l == 0 else 1 for l in raw_labels]
    else:
        labels = [int(l) for l in raw_labels]

    print(f"train size: {len(train_texts)}, "
          f"unique labels: {sorted(set(labels))}")

    # Triplet trainer expects columns (sentence, label).
    train_ds = Dataset.from_dict({"sentence": train_texts, "label": labels})

    model = SentenceTransformer(model_name_or_path=args.base_model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=stage1_dir,
        eval_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=args.stage1_batch,
        per_device_eval_batch_size=args.stage1_batch,
        learning_rate=args.stage1_lr,
        num_train_epochs=args.stage1_epochs,
        warmup_ratio=args.stage1_warmup,
        save_strategy="no",
        seed=args.seed,
        report_to="none",
    )

    criterion = BatchHardSoftMarginTripletLoss(model=model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        loss=criterion,
    )
    trainer.train()

    model.save(stage1_dir)
    print(f"[stage1] saved encoder to {stage1_dir}")
    return stage1_dir


# ── Stage 2: SBERTwithClassifier head training ──────────────────────

def stage2_head(args, encoder_dir, save_dir):
    print("\n" + "="*60)
    print(f"STAGE 2: head training (freeze_backbone={args.stage2_freeze_backbone})")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The SentenceTransformer save dir typically has subdirs like
    # 0_Transformer/ for the underlying HF model. Find it.
    inner_dir = encoder_dir
    transformer_subdir = os.path.join(encoder_dir, "0_Transformer")
    if os.path.isdir(transformer_subdir):
        inner_dir = transformer_subdir
    print(f"[stage2] loading encoder from: {inner_dir}")

    model = SBERTwithClassifier(inner_dir, num_labels=2).to(device)
    # Replace the freshly-initialized backbone weights with the trained ones
    from transformers import AutoModel
    trained_backbone = AutoModel.from_pretrained(inner_dir).to(device)
    model.backbone.load_state_dict(trained_backbone.state_dict(), strict=True)
    del trained_backbone
    torch.cuda.empty_cache()

    if args.stage2_freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        print("[stage2] backbone frozen; only head trains")

    tokenizer = AutoTokenizer.from_pretrained(inner_dir)

    raw = load_from_disk(args.dataset)
    train_texts = [str(t) for t in raw["train"]["text"]]
    test_texts = [str(t) for t in raw["test"]["text"]]
    train_y = torch.tensor([0 if l == 0 else 1 for l in raw["train"]["label"]])
    test_y = torch.tensor([0 if l == 0 else 1 for l in raw["test"]["label"]])

    print(f"train: {(train_y==0).sum()} safe, {(train_y==1).sum()} harmful")
    print(f"test:  {(test_y==0).sum()} safe, {(test_y==1).sum()} harmful")

    print("Tokenizing...")
    enc_tr = tokenizer(train_texts, padding=True, truncation=True,
                       max_length=args.max_length, return_tensors="pt")
    enc_te = tokenizer(test_texts, padding=True, truncation=True,
                       max_length=args.max_length, return_tensors="pt")

    train_ds = TensorDataset(enc_tr["input_ids"], enc_tr["attention_mask"], train_y)
    test_ds = TensorDataset(enc_te["input_ids"], enc_te["attention_mask"], test_y)
    train_loader = DataLoader(train_ds, batch_size=args.stage2_batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.stage2_batch, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.stage2_lr,
    )

    for epoch in range(args.stage2_epochs):
        model.train()
        total = correct = 0
        loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"Stage2 ep{epoch+1}/{args.stage2_epochs}")
        for ids, am, y in pbar:
            ids, am, y = ids.to(device), am.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(ids, am)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            preds = logits.argmax(-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=f"{loss_sum/total:.4f}",
                             acc=f"{correct/total:.4f}")

        # Test eval
        model.eval()
        t_correct = t_total = 0
        safe_c = safe_t = harm_c = harm_t = 0
        with torch.no_grad():
            for ids, am, y in test_loader:
                ids, am, y = ids.to(device), am.to(device), y.to(device)
                preds = model(ids, am).argmax(-1)
                t_correct += (preds == y).sum().item()
                t_total += y.size(0)
                safe = (y == 0)
                harm = (y == 1)
                safe_c += (preds[safe] == y[safe]).sum().item()
                safe_t += safe.sum().item()
                harm_c += (preds[harm] == y[harm]).sum().item()
                harm_t += harm.sum().item()
        print(f"  test acc {t_correct/t_total:.4f} | "
              f"safe {safe_c/max(safe_t,1):.4f} | "
              f"harmful {harm_c/max(harm_t,1):.4f}")

    # Save in SBERTwithClassifier format
    os.makedirs(save_dir, exist_ok=True)
    save_file(model.state_dict(), os.path.join(save_dir, "model.safetensors"))
    model.backbone.config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\n[saved] {save_dir}")

    # Final report
    print("\n" + "="*40)
    print("FINAL RESULTS")
    print("="*40)
    print(f"  Test Accuracy:    {t_correct/t_total:.2%}")
    print(f"  Safe accuracy:    {safe_c/max(safe_t,1):.2%}")
    print(f"  Harmful accuracy: {harm_c/max(harm_t,1):.2%}")


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    stage1_dir = os.path.join(args.save_dir, "_stage1_encoder")

    stage1_triplet(args, stage1_dir)
    stage2_head(args, stage1_dir, args.save_dir)


if __name__ == "__main__":
    main()
