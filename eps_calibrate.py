"""Measure GCG activation shifts at every layer for any harm classifier.

Computes global L2, per-position L2, suffix/content breakdown, and optional
target-dim L2 norms of the activation difference (clean - adversarial) at
each layer. Used to calibrate epsilon ball ranges for LAT surgery sweeps.

Supports BERT, RoBERTa, DeBERTa, and SBERT (DistilBERT) model types.

Usage:
    conda run -n bertibp python eps_calibrate.py --model models/bert_harm_binary --gcg gcg_results/bert_harm.pt
    conda run -n bertibp python eps_calibrate.py --model models/roberta_harm_binary --gcg gcg_results/roberta_harm.pt
    conda run -n bertibp python eps_calibrate.py --model models/deberta_harm_binary --gcg gcg_results/deberta_harm.pt --target_dims 682,683
    conda run -n bertibp python eps_calibrate.py --model models/sbert_harm --gcg gcg_results/sbert_harm.pt --model_type sbert
"""

import argparse, os, torch, numpy as np

os.environ["TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION"] = "1"

BACKBONE_ATTR = {
    "bert": "bert",
    "roberta": "roberta",
    "deberta": "deberta",
    "deberta-v2": "deberta",
}


def get_backbone(model, model_type):
    attr = BACKBONE_ATTR.get(model_type)
    if attr is None:
        raise ValueError(f"Unsupported model_type={model_type}. "
                         f"Supported: {list(BACKBONE_ATTR.keys())}")
    return getattr(model, attr)


def get_layer_modules(backbone, model_type):
    """Return the list of transformer layer modules."""
    if model_type == "sbert":
        return backbone.transformer.layer
    return backbone.encoder.layer


def capture_all(model, backbone, tokenizer, text, all_layers, max_length,
                device, model_type="bert"):
    """Capture embedding + all layer outputs for a single text."""
    captured = {}

    def emb_hook(_m, _i, output):
        out = output[0] if isinstance(output, tuple) else output
        captured["embed"] = out.detach().clone()

    def make_hook(idx):
        def hook(_m, _i, output):
            out = output[0] if isinstance(output, tuple) else output
            captured[idx] = out.detach().clone()
        return hook

    hooks = [backbone.embeddings.register_forward_hook(emb_hook)]
    layer_modules = get_layer_modules(backbone, model_type)
    for idx in all_layers:
        hooks.append(layer_modules[idx].register_forward_hook(make_hook(idx)))

    inputs = tokenizer(text, padding="max_length", truncation=True,
                       max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        if model_type == "sbert":
            model(inputs["input_ids"], inputs["attention_mask"])
        else:
            model(**inputs)

    for h in hooks:
        h.remove()
    return captured, inputs["attention_mask"].squeeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to harm classifier model")
    parser.add_argument("--gcg", required=True, help="Path to GCG cache .pt file")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--target_dims", type=str, default=None,
                        help="Comma-separated target dims for per-dim analysis (e.g. 682,683)")
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["bert", "roberta", "deberta", "sbert"],
                        help="Override model type (required for sbert)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda:$CUDA_DEVICE or cuda:0)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        _gpu = os.environ.get("CUDA_DEVICE", "0")
        device = torch.device(f"cuda:{_gpu}" if torch.cuda.is_available() else "cpu")

    target_dims = []
    if args.target_dims:
        target_dims = [int(d) for d in args.target_dims.split(",")]

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    from safetensors.torch import load_file

    cfg = AutoConfig.from_pretrained(args.model)
    model_type = args.model_type or cfg.model_type
    if model_type == "sbert":
        num_layers = getattr(cfg, "n_layers", getattr(cfg, "num_hidden_layers", 6))
    else:
        num_layers = cfg.num_hidden_layers
    all_layers = list(range(num_layers))

    print(f"Model type: {model_type}, layers: {num_layers}")

    print("Loading cached GCG attacks...")
    raw = torch.load(args.gcg, map_location="cpu", weights_only=False)
    attacks = [a for a in raw if a.get("succeeded", False)]
    print(f"  {len(attacks)} successful attacks")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if model_type == "sbert":
        # SBERTwithClassifier lives in the freeze-lat sibling repo (local) or
        # ibp_huggingface (SDSC). Probe both.
        import sys
        _here = os.path.dirname(os.path.abspath(__file__))
        for _cand in (os.path.join(_here, '..', 'freeze-lat', 'sbert'),
                      os.path.join(_here, '..', 'ibp_huggingface', 'sbert')):
            if os.path.exists(os.path.join(_cand, 'SBERTwithClassifier.py')):
                sys.path.insert(0, _cand)
                break
        from SBERTwithClassifier import SBERTwithClassifier
        model = SBERTwithClassifier(args.model, num_labels=2).to(device)
        st = load_file(os.path.join(args.model, "model.safetensors"))
        model.load_state_dict(st, strict=True)
        backbone = model.backbone
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, use_safetensors=True
        ).to(device)
        backbone = get_backbone(model, model_type)
    model.eval()

    keys = ["embed"] + all_layers
    stats = {k: {
        "l2_global": [],
        "l2_per_pos_max": [],
        "l2_per_pos_mean": [],
        "l2_per_pos_p90": [],
        "l2_per_pos_median": [],
        "l2_suffix_global": [],
        "l2_suffix_per_pos_max": [],
        "l2_content_global": [],
        "l2_content_per_pos_max": [],
    } for k in keys}
    for d in target_dims:
        for k in keys:
            stats[k][f"l2_dim{d}"] = []

    n_used = 0
    for i, atk in enumerate(attacks):
        clean_text = atk["input_text"]
        adv_text = clean_text + " " + atk["suffix_text"]

        clean_acts, clean_mask = capture_all(
            model, backbone, tokenizer, clean_text, all_layers,
            args.max_length, device, model_type)
        adv_acts, adv_mask = capture_all(
            model, backbone, tokenizer, adv_text, all_layers,
            args.max_length, device, model_type)

        clean_len = clean_mask.sum().item()
        adv_len = adv_mask.sum().item()

        # Verify adversarial example actually flips the prediction
        adv_ids = tokenizer(adv_text, padding="max_length", truncation=True,
                            max_length=args.max_length, return_tensors="pt")
        with torch.no_grad():
            if model_type == "sbert":
                a_logits = model(adv_ids["input_ids"].to(device),
                                 adv_ids["attention_mask"].to(device))
            else:
                a_logits = model(adv_ids["input_ids"].to(device),
                                 adv_ids["attention_mask"].to(device)).logits
        if a_logits.argmax(-1).item() == atk["original_label"]:
            continue

        n_used += 1

        # Suffix positions: tokens beyond clean's [SEP]
        # clean: [CLS] content [SEP] [PAD]...
        # adv:   [CLS] content suffix [SEP] [PAD]...
        suffix_start = clean_len - 1
        suffix_end = adv_len - 1

        for k in keys:
            c = clean_acts[k].squeeze(0)
            p = adv_acts[k].squeeze(0)
            seq = min(c.shape[0], p.shape[0])
            diff = c[:seq] - p[:seq]

            active = min(adv_len, seq)
            active_diff = diff[:active]
            l2_global = active_diff.norm(p=2).item()
            stats[k]["l2_global"].append(l2_global)

            per_pos = active_diff.norm(p=2, dim=1)
            stats[k]["l2_per_pos_max"].append(per_pos.max().item())
            stats[k]["l2_per_pos_mean"].append(per_pos.mean().item())
            stats[k]["l2_per_pos_p90"].append(
                torch.quantile(per_pos.float(), 0.9).item())
            stats[k]["l2_per_pos_median"].append(
                torch.quantile(per_pos.float(), 0.5).item())

            if suffix_start < active and suffix_end <= active:
                suffix_diff = diff[suffix_start:suffix_end]
                stats[k]["l2_suffix_global"].append(suffix_diff.norm(p=2).item())
                stats[k]["l2_suffix_per_pos_max"].append(
                    suffix_diff.norm(p=2, dim=1).max().item() if suffix_diff.shape[0] > 0 else 0.0)
            else:
                stats[k]["l2_suffix_global"].append(0.0)
                stats[k]["l2_suffix_per_pos_max"].append(0.0)

            content_diff = diff[1:suffix_start]
            if content_diff.shape[0] > 0:
                stats[k]["l2_content_global"].append(content_diff.norm(p=2).item())
                stats[k]["l2_content_per_pos_max"].append(
                    content_diff.norm(p=2, dim=1).max().item())
            else:
                stats[k]["l2_content_global"].append(0.0)
                stats[k]["l2_content_per_pos_max"].append(0.0)

            for d in target_dims:
                dim_l2 = active_diff[:, d].norm(p=2).item()
                stats[k][f"l2_dim{d}"].append(dim_l2)

        if n_used % 10 == 0:
            print(f"  processed {n_used}/{len(attacks)}")

    print(f"\n{n_used} attacks used (verified flips)")

    # ========== Print results ==========
    model_name = os.path.basename(args.model)
    print(f"\n{'='*90}")
    print(f"GCG ACTIVATION SHIFTS — {model_name}")
    print(f"{'='*90}")
    print(f"  Model: {args.model}")
    print(f"  model_type: {model_type}, layers: {num_layers}")
    print(f"  max_length: {args.max_length}")
    print(f"  Verified GCG attacks: {n_used}")

    print(f"\n{'─'*90}")
    print("PART 1: GLOBAL L2 NORMS (all active positions)")
    print(f"{'─'*90}")
    print(f"  {'Layer':<8} {'Mean':>8} {'Med':>8} {'P75':>8} {'P90':>8} {'Max':>8}")
    print(f"  {'─'*50}")
    for k in keys:
        s = stats[k]
        vals = s["l2_global"]
        lbl = f"L{k}" if isinstance(k, int) else "Embed"
        print(f"  {lbl:<8} {np.mean(vals):>8.2f} {np.median(vals):>8.2f} "
              f"{np.percentile(vals, 75):>8.2f} {np.percentile(vals, 90):>8.2f} "
              f"{np.max(vals):>8.2f}")

    print(f"\n{'─'*90}")
    print("PART 2: PER-POSITION L2 NORMS (mean across attacks)")
    print(f"{'─'*90}")
    print(f"  {'Layer':<8} {'MaxPos':>8} {'MeanPos':>8} {'MedPos':>8} {'P90Pos':>8}")
    print(f"  {'─'*42}")
    for k in keys:
        s = stats[k]
        lbl = f"L{k}" if isinstance(k, int) else "Embed"
        print(f"  {lbl:<8} {np.mean(s['l2_per_pos_max']):>8.2f} "
              f"{np.mean(s['l2_per_pos_mean']):>8.2f} "
              f"{np.mean(s['l2_per_pos_median']):>8.2f} "
              f"{np.mean(s['l2_per_pos_p90']):>8.2f}")

    print(f"\n{'─'*90}")
    print("PART 3: SUFFIX vs CONTENT POSITIONS")
    print(f"{'─'*90}")
    print(f"  {'Layer':<8} {'Sfx_Gbl':>8} {'Sfx_Max':>8} {'Cnt_Gbl':>8} {'Cnt_Max':>8} {'Sfx%':>8}")
    print(f"  {'─'*48}")
    for k in keys:
        s = stats[k]
        sg = np.mean(s["l2_suffix_global"])
        sm = np.mean(s["l2_suffix_per_pos_max"])
        cg = np.mean(s["l2_content_global"])
        cm = np.mean(s["l2_content_per_pos_max"])
        total = sg + cg + 1e-10
        lbl = f"L{k}" if isinstance(k, int) else "Embed"
        print(f"  {lbl:<8} {sg:>8.2f} {sm:>8.2f} {cg:>8.2f} {cm:>8.2f} "
              f"{100*sg/total:>7.1f}%")

    if target_dims:
        print(f"\n{'─'*90}")
        print("PART 4: TARGET DIMENSIONS")
        print(f"{'─'*90}")
        header = f"  {'Layer':<8}"
        for d in target_dims:
            header += f" {'d'+str(d)+'_L2':>8}"
        header += f" {'%ofTotal':>10}"
        print(header)
        print(f"  {'─'*40}")
        for k in keys:
            s = stats[k]
            lbl = f"L{k}" if isinstance(k, int) else "Embed"
            dim_vals = [np.mean(s[f"l2_dim{d}"]) for d in target_dims]
            tgt_l2 = np.sqrt(sum(v**2 for v in dim_vals))
            gbl = np.mean(s["l2_global"])
            pct = 100 * tgt_l2 / (gbl + 1e-10)
            line = f"  {lbl:<8}"
            for v in dim_vals:
                line += f" {v:>8.3f}"
            line += f" {pct:>9.1f}%"
            print(line)

    # ========== Suggested eps ranges ==========
    print(f"\n{'='*90}")
    print("SUGGESTED EPS RANGES FOR LAT SWEEP")
    print(f"{'='*90}")
    print("  Based on: median global L2 ~ reasonable coverage, P75 ~ aggressive")
    print(f"\n  {'Layer':<8} {'Med_Gbl':>8} {'P75_Gbl':>8} {'Med_PosMax':>10} {'P75_PosMax':>10}")
    print(f"  {'─'*50}")
    for k in keys:
        s = stats[k]
        lbl = f"L{k}" if isinstance(k, int) else "Embed"
        med_g = np.median(s["l2_global"])
        p75_g = np.percentile(s["l2_global"], 75)
        med_pm = np.median(s["l2_per_pos_max"])
        p75_pm = np.percentile(s["l2_per_pos_max"], 75)
        print(f"  {lbl:<8} {med_g:>8.1f} {p75_g:>8.1f} {med_pm:>10.1f} {p75_pm:>10.1f}")


if __name__ == "__main__":
    main()
