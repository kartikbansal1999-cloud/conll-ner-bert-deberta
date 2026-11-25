"""Fine-tune a transformer for CoNLL-2003 NER, select by validation F1, report once on test.

Usage:
    python src/train.py --model bert-base-cased --seed 42
    python src/train.py --model microsoft/deberta-v3-base --seed 42
    python src/train.py --model bert-base-cased --max_train_samples 200 --epochs 1   # smoke test
"""
import argparse
import json
import os

import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments, set_seed)

from common import load_conll, make_compute_metrics, tokenize_and_align
from evaluation import evaluate_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="bert-base-cased")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--max_train_samples", type=int, default=None)
ap.add_argument("--output_dir", default=None)
a = ap.parse_args()

set_seed(a.seed)
run_dir = a.output_dir or os.path.join("runs", f"{a.model.split('/')[-1]}-s{a.seed}")
ds, labels = load_conll()
if a.max_train_samples:
    ds["train"] = ds["train"].select(range(a.max_train_samples))
    ds["validation"] = ds["validation"].select(range(min(200, len(ds["validation"]))))
    ds["test"] = ds["test"].select(range(min(200, len(ds["test"]))))

tok = AutoTokenizer.from_pretrained(a.model, add_prefix_space=True) if "roberta" in a.model else AutoTokenizer.from_pretrained(a.model)
model = AutoModelForTokenClassification.from_pretrained(
    a.model, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})

cols = ds["train"].column_names
tokenized = ds.map(lambda x: tokenize_and_align(x, tok), batched=True, remove_columns=cols)

cuda = torch.cuda.is_available()
bf16 = cuda and torch.cuda.is_bf16_supported()
args = TrainingArguments(
    output_dir=os.path.join(run_dir, "checkpoints"), learning_rate=a.lr, num_train_epochs=a.epochs,
    per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=64, weight_decay=0.01, warmup_ratio=0.1,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
    metric_for_best_model="f1", greater_is_better=True, seed=a.seed, fp16=cuda and not bf16, bf16=bf16,
    logging_steps=50, report_to="none")
trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
                  data_collator=DataCollatorForTokenClassification(tok), processing_class=tok,
                  compute_metrics=make_compute_metrics(labels))
trainer.train()

best = os.path.join(run_dir, "best")
trainer.save_model(best)
tok.save_pretrained(best)
with open(os.path.join(run_dir, "train_log.json"), "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

model.to("cuda" if cuda else "cpu")
print("\n=== TEST (best checkpoint by validation F1) ===")
evaluate_model(model, tok, ds["test"], labels, os.path.join(run_dir, "test"))
