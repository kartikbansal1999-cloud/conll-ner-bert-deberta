"""Evaluation: entity-level metrics, bootstrap CI, error taxonomy, and a non-neural baseline.

Usage:
    python src/evaluation.py --model_dir runs/bert-base-cased-s42/best --out_dir runs/bert-base-cased-s42/test
    python src/evaluation.py --baseline --out_dir runs/baseline
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch
from seqeval.metrics import classification_report
from seqeval.metrics.sequence_labeling import get_entities

from common import load_conll


# ---------- prediction ----------
@torch.no_grad()
def predict_words(model, tokenizer, token_lists, label_names, batch_size=64, max_length=256):
    """Return word-level tag strings (prediction of each word's first sub-token; 'O' if truncated)."""
    device = next(model.parameters()).device
    model.eval()
    out = []
    for s in range(0, len(token_lists), batch_size):
        batch = token_lists[s : s + batch_size]
        enc = tokenizer(batch, is_split_into_words=True, truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt")
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        pred_ids = logits.argmax(-1).cpu().numpy()
        for i, words in enumerate(batch):
            tags, prev = ["O"] * len(words), None
            for pos, w in enumerate(enc.word_ids(batch_index=i)):
                if w is not None and w != prev:
                    tags[w] = label_names[pred_ids[i][pos]]
                prev = w
            out.append(tags)
    return out


def most_frequent_tag_baseline(train_split, token_lists, label_names):
    """Each word gets its most frequent training tag (case-sensitive); unseen words -> 'O'."""
    counts = defaultdict(Counter)
    for toks, tags in zip(train_split["tokens"], train_split["ner_tags"]):
        for t, g in zip(toks, tags):
            counts[t][g] += 1
    table = {t: label_names[c.most_common(1)[0][0]] for t, c in counts.items()}
    return [[table.get(t, "O") for t in toks] for toks in token_lists]


# ---------- metrics ----------
def _entities(tags):
    return set(get_entities(tags))  # (type, start, end)


def bootstrap_f1(y_true, y_pred, n_boot=1000, seed=0):
    """95% CI for entity-level micro-F1 by resampling sentences."""
    tp, fp, fn = [], [], []
    for g, p in zip(y_true, y_pred):
        ge, pe = _entities(g), _entities(p)
        tp.append(len(ge & pe)); fp.append(len(pe - ge)); fn.append(len(ge - pe))
    tp, fp, fn = map(np.array, (tp, fp, fn))
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(tp), len(tp))
        t, f_p, f_n = tp[idx].sum(), fp[idx].sum(), fn[idx].sum()
        scores.append(2 * t / (2 * t + f_p + f_n) if t else 0.0)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def error_taxonomy(tokens, y_true, y_pred, max_examples=200):
    """Categorise every gold/pred entity. Returns (Counter, example list)."""
    counts, examples = Counter(), []
    for toks, g, p in zip(tokens, y_true, y_pred):
        ge, pe = get_entities(g), get_entities(p)
        pe_set, ge_set = set(pe), set(ge)

        def overlap(a, b):
            return a[1] <= b[2] and b[1] <= a[2]

        for gold in ge:
            if gold in pe_set:
                counts["correct"] += 1; continue
            ov = [x for x in pe if overlap(gold, x)]
            if not ov:
                kind = "missed"
            elif any(x[1:] == gold[1:] for x in ov):
                kind = "type_error"              # right span, wrong type
            elif any(x[0] == gold[0] for x in ov):
                kind = "boundary_error"          # right type, wrong span
            else:
                kind = "boundary_and_type_error"
            counts[kind] += 1
            if len(examples) < max_examples:
                examples.append({"kind": kind, "gold": gold[0], "gold_text": " ".join(toks[gold[1]:gold[2] + 1]),
                                 "pred": [(x[0], " ".join(toks[x[1]:x[2] + 1])) for x in ov],
                                 "sentence": " ".join(toks)})
        for pred in pe:
            if not any(overlap(pred, x) for x in ge):
                counts["spurious"] += 1
                if len(examples) < max_examples:
                    examples.append({"kind": "spurious", "pred": pred[0],
                                     "pred_text": " ".join(toks[pred[1]:pred[2] + 1]), "sentence": " ".join(toks)})
    return counts, examples


def evaluate_predictions(tokens, y_true, y_pred, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rep = classification_report(y_true, y_pred, output_dict=True, digits=4)
    lo, hi = bootstrap_f1(y_true, y_pred)
    counts, examples = error_taxonomy(tokens, y_true, y_pred)
    result = {
        "micro_f1": rep["micro avg"]["f1-score"], "micro_precision": rep["micro avg"]["precision"],
        "micro_recall": rep["micro avg"]["recall"], "macro_f1": rep["macro avg"]["f1-score"],
        "micro_f1_ci95": [lo, hi],
        "per_type": {k: v for k, v in rep.items() if k not in ("micro avg", "macro avg", "weighted avg")},
        "error_taxonomy": dict(counts),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(result, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(os.path.join(out_dir, "errors.jsonl"), "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"micro-F1 {result['micro_f1']:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print("error taxonomy:", dict(counts))
    return result


def evaluate_model(model, tokenizer, split, label_names, out_dir):
    y_true = [[label_names[t] for t in tags] for tags in split["ner_tags"]]
    y_pred = predict_words(model, tokenizer, split["tokens"], label_names)
    return evaluate_predictions(split["tokens"], y_true, y_pred, out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir")
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()
    ds, labels = load_conll()
    split = ds[a.split]
    if a.baseline:
        y_true = [[labels[t] for t in tags] for tags in split["ner_tags"]]
        y_pred = most_frequent_tag_baseline(ds["train"], split["tokens"], labels)
        evaluate_predictions(split["tokens"], y_true, y_pred, a.out_dir)
    else:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model_dir, add_prefix_space=True) \
            if "roberta" in a.model_dir.lower() else AutoTokenizer.from_pretrained(a.model_dir)
        model = AutoModelForTokenClassification.from_pretrained(a.model_dir)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        evaluate_model(model, tok, split, labels, a.out_dir)
