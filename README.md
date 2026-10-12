# Research Search

> **Recruiter quick view:** real PubMed benchmark, official train/dev/test splits, explainable TF-IDF/logistic baseline, PyTorch neural baseline, robustness testing, browser/Python parity, error analysis, tests, and CI. Start with [RECRUITER_OVERVIEW.md](RECRUITER_OVERVIEW.md), [PORTFOLIO_EVIDENCE.md](PORTFOLIO_EVIDENCE.md), [MODEL_CARD.md](MODEL_CARD.md), and the [interactive model demo](https://erickvera610.github.io/research-search/model-demo.html).

## Real-data scientific text classification

The main experiment uses **PubMed 20k RCT** with official abstract-level splits:

- Train: **180,040 sentences**
- Validation: **30,212 sentences**
- Test: **30,135 sentences**
- Distinct abstracts: 15,000 / 2,500 / 2,500
- Abstract overlap across partitions: **0**

The explainable baseline uses TF-IDF with up to 1,000 word features and multinomial logistic regression. Hyperparameter selection uses validation macro F1; the official test split is evaluated after model selection.

Held-out results:

| Metric | Trained model | Majority baseline |
|---|---:|---:|
| Accuracy | **76.72%** | **32.84%** |
| Macro F1 | **0.7020** | **0.0989** |

The classifier predicts BACKGROUND, OBJECTIVE, METHODS, RESULTS, or CONCLUSIONS for individual scientific-abstract sentences.

## PyTorch neural baseline

`pytorch_text_baseline.py` adds a lightweight neural comparison using `EmbeddingBag(mean, 64) -> Linear(5)`.

```bash
python -m pip install -r requirements-pytorch.txt
python pytorch_text_baseline.py
```

The neural baseline is intentionally small. It demonstrates PyTorch training and representation learning; it is **not** a transformer, pretrained language model, or LLM. Results are written to `artifacts/pytorch-text-results.json` and should not be quoted until reproduced in the current environment.

## Robustness evaluation

`robustness_benchmark.py` reports:

- clean test accuracy and macro F1,
- performance by sentence-length group,
- deterministic token-dropout stress performance,
- prediction change rate after token removal.

```bash
python robustness_benchmark.py
```

This stress test exposes brittleness; token deletion is synthetic and does not establish clinical or out-of-domain robustness.

## Explainability and failure analysis

The main model exports:

- confusion matrix,
- per-class precision/recall/F1,
- 20 confidently wrong test predictions,
- PMID and sentence index for error inspection.

This makes failure analysis directly inspectable instead of showing only an aggregate score.

## Demo and reproducibility

**[Interactive trained-model demo](https://erickvera610.github.io/research-search/model-demo.html)** · [Model card](MODEL_CARD.md) · [Actions](https://github.com/erickvera610/research-search/actions)

```bash
python -m pip install -r requirements-model.txt
python train_model.py
python -m unittest discover -v
node test_model.cjs
python model_server.py
```

The public demo uses exported trained parameters. The Python API uses the same parameters, and parity tests compare browser and Python predictions.

## Original retrieval prototype

The repository also preserves an explainable local retrieval prototype based on TF-IDF, cosine similarity, and matched terms. It runs locally with no external API and does not claim semantic understanding, synonym resolution, or answer generation.

```bash
python app.py
```

## Skills demonstrated

Python, NLP preprocessing, TF-IDF, multinomial logistic regression, PyTorch, EmbeddingBag, official train/dev/test evaluation, macro F1, confusion matrices, error analysis, robustness testing, model export, inference APIs, tests, and CI.

## Key files

- `train_model.py` — official-split TF-IDF/logistic experiment.
- `pytorch_text_baseline.py` — PyTorch neural baseline.
- `robustness_benchmark.py` — test-set robustness checks.
- `MODEL_CARD.md` — full protocol, results, errors, and limitations.
- `artifacts/model-results.json` — held-out metrics and errors.
- `RECRUITER_OVERVIEW.md` — recruiter review path and screenshot checklist.
- `PORTFOLIO_EVIDENCE.md` — evidence summary.

## Limitations

The main model is a 1,000-feature TF-IDF classifier and does not model deep semantic context. The PyTorch baseline is deliberately lightweight. Neither model is a clinical decision system, semantic search engine, transformer, or production NLP service. Dataset abstracts may have copyright restrictions; raw source text is downloaded for reproduction and is not redistributed by the project.

## Authorship

Independent AI-assisted learning project in Erik Vera's portfolio. No employer affiliation or proprietary employer data. Code is MIT licensed; dataset provenance and rights are documented separately.
