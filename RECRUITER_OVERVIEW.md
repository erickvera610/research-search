# Recruiter Quick View

## What this project proves

This repository demonstrates scientific-text classification on a large real benchmark with official abstract-level splits, explainable TF-IDF/logistic regression, error analysis, robustness checks, browser/Python parity, and a separate PyTorch neural baseline.

## Best evidence to inspect first

1. `MODEL_CARD.md` — official split protocol, metrics, errors, and limitations.
2. `artifacts/model-results.json` — held-out test results and confident errors.
3. `robustness_benchmark.py` — sentence-length and token-dropout stress tests.
4. `pytorch_text_baseline.py` — lightweight EmbeddingBag neural baseline.
5. `model-demo.html` — public browser inference using exported parameters.
6. `train_model.py` — TF-IDF + multinomial logistic regression training.

## Skills shown

Python, NLP preprocessing, TF-IDF, multinomial logistic regression, PyTorch, EmbeddingBag, official train/dev/test evaluation, macro F1, confusion matrices, error analysis, reproducibility, model export, APIs, tests, and CI.

## Important interpretation

The neural baseline is intentionally lightweight and is not a transformer or pretrained language model. Its purpose is to show PyTorch training and to compare a neural representation against the stronger explainable baseline under the same dataset split discipline.

## Recommended interview demo

Run:

```bash
python -m pip install -r requirements-pytorch.txt
python train_model.py
python robustness_benchmark.py
python pytorch_text_baseline.py
```

Explain why abstracts, not individual sentences, define split independence; then inspect one high-confidence error and discuss what TF-IDF cannot represent.

## Screenshot/result checklist

Capture:

- the scientific sentence-role demo,
- overall accuracy/macro F1 versus majority baseline,
- confusion matrix or per-class results,
- a documented high-confidence error,
- robustness degradation under token dropout,
- PyTorch baseline result for comparison.

All screenshots should trace to reproducible code or JSON artifacts.
