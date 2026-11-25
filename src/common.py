"""Shared data loading, tokenisation/label alignment and training-time metrics."""
import numpy as np
from datasets import load_dataset
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

DATASET = "eriktks/conll2003"  # CoNLL-2003, IOB2 tags
# The repo's main branch uses a loading script (unsupported in recent `datasets`),
# so we read the Parquet files the Hub auto-converts from it.
REVISION = "refs/convert/parquet"
CONLL_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def load_conll():
    ds = load_dataset(DATASET, revision=REVISION)
    feat = ds["train"].features["ner_tags"]
    label_names = feat.feature.names if hasattr(feat, "feature") and hasattr(feat.feature, "names") else CONLL_LABELS
    assert len(label_names) == 9 and label_names[0] == "O", label_names
    print(f"train/val/test sizes: {len(ds['train'])}/{len(ds['validation'])}/{len(ds['test'])}; labels: {label_names}")
    return ds, label_names


def tokenize_and_align(examples, tokenizer, max_length=256):
    """Tokenise pre-split words; label only the first sub-token of each word (-100 elsewhere)."""
    enc = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, max_length=max_length)
    all_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        prev, row = None, []
        for w in enc.word_ids(batch_index=i):
            row.append(-100 if (w is None or w == prev) else tags[w])
            prev = w
        all_labels.append(row)
    enc["labels"] = all_labels
    return enc


def make_compute_metrics(label_names):
    def compute_metrics(p):
        logits, labels = p
        preds = np.argmax(logits, axis=-1)
        y_pred, y_true = [], []
        for pr, lb in zip(preds, labels):
            mask = lb != -100
            y_pred.append([label_names[x] for x in pr[mask]])
            y_true.append([label_names[x] for x in lb[mask]])
        return {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
        }

    return compute_metrics
