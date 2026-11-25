# Fine-tuning BERT & DeBERTa for Named Entity Recognition (CoNLL-2003)

Token-classification fine-tuning of `bert-base-cased` and `microsoft/deberta-v3-base` on CoNLL-2003
(PER / ORG / LOC / MISC), with an evaluation that goes beyond a single F1 number.

## Setup

- **Data:** CoNLL-2003 (`eriktks/conll2003`), standard train/validation/test split, IOB2 tags.
- **Model:** pre-trained encoder + linear token-classification head. The label goes on the first sub-token of each word; the rest are masked (`-100`).
- **Training:** AdamW, lr 3e-5, batch 32, 3 epochs, 10% warmup, weight decay 0.01. The checkpoint is chosen by **validation** F1. **Test is scored once**, on that checkpoint.
- **Seeds:** 3 per model (42, 43, 44). Results are mean ± std.

## Evaluation

- Entity-level precision / recall / F1 (`seqeval`, exact span + type match), not token accuracy.
- Per-entity-type breakdown.
- 95% bootstrap confidence interval on micro-F1 (1,000 resamples over sentences).
- **Error taxonomy** for every gold entity: correct / wrong type / wrong boundary / both / missed, plus spurious predictions. Examples are written to `errors.jsonl`.
- **Baseline:** most-frequent-tag-per-word lookup, to show what the transformer adds over memorisation.

## Results (test set)

| Model | Micro-F1 (mean ± std, 3 seeds) | PER | ORG | LOC | MISC |
|---|---|---|---|---|---|
| Most-frequent-tag baseline | _fill in_ | | | | |
| bert-base-cased | _fill in_ | | | | |
| deberta-v3-base | _fill in_ | | | | |

**Error analysis:** _2–3 sentences on what dominates (e.g. ORG/LOC confusion, boundary errors on multi-word names, MISC) and one example each._

## Reproduce

Needs a GPU (a free Colab T4 takes ~10 min per run).

```bash
pip install -r requirements.txt
python src/evaluation.py --baseline --out_dir runs/baseline
for s in 42 43 44; do
  python src/train.py --model bert-base-cased --seed $s
  python src/train.py --model microsoft/deberta-v3-base --seed $s
done
```

Each run writes `runs/<model>-s<seed>/test/metrics.json` and `errors.jsonl`.
Quick smoke test on CPU: `python src/train.py --max_train_samples 200 --epochs 1`.

## Layout

```
src/common.py      data loading, sub-token label alignment, training metrics
src/train.py       fine-tuning + final test evaluation
src/evaluation.py  entity metrics, bootstrap CI, error taxonomy, baseline
```

## Limitations

- CoNLL-2003 is news text from 1996; results will not transfer to other domains without adaptation.
- Three seeds give a rough variance estimate, not a tight one.
- No hyperparameter search: both models use the same settings.
