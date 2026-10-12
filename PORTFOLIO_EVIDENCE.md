# Portfolio Evidence

This page summarizes the strongest recruiter-facing evidence in this repository and separates measured results from broader claims.

## Real scientific text data

- Dataset: **PubMed 20k RCT** from Dernoncourt and Lee (IJCNLP 2017).
- Official abstract-level partitions are preserved.
- Downloaded original-text subset:
  - **180,040 training sentences**
  - **30,212 development sentences**
  - **30,135 test sentences**
- Abstract counts:
  - 15,000 train
  - 2,500 development
  - 2,500 test
- Cross-partition abstract overlap is explicitly checked and equals zero.

## Task

Classify scientific abstract sentences into one of five roles:

- BACKGROUND
- OBJECTIVE
- METHODS
- RESULTS
- CONCLUSIONS

This is a supervised scientific-text classification problem, not a retrieval or generative-LLM benchmark.

## Model and selection protocol

- TF-IDF representation with at most 1,000 word features.
- Lowercase ASCII tokenization.
- Minimum training document frequency of 5.
- Multinomial logistic regression.
- Candidate regularization values: `C=0.5` and `C=2`.
- Model selection uses validation macro F1.
- Vocabulary and IDF are fitted on training data during selection, then refitted on train + development.
- Final test text does not influence fitting or model selection.

## Held-out results

| Metric | Trained model | Majority-class baseline |
|---|---:|---:|
| Accuracy | **76.72%** | **32.84%** |
| Macro F1 | **0.7020** | **0.0989** |

The repository also exports per-class precision, recall and F1, a confusion matrix, and **20 confidently wrong predictions** for explicit failure analysis.

## Evidence a recruiter can inspect

1. `train_model.py` — full real-data training pipeline.
2. `MODEL_CARD.md` — experiment design, dataset provenance, results, limitations, and application relevance.
3. `artifacts/model-results.json` — test metrics, confusion matrix, per-class performance, and documented errors.
4. `data/SOURCE.json` — pinned dataset source and checksums.
5. `model-demo.html` — browser inference using exported trained parameters.
6. `model_server.py` — validated Python inference API.
7. `test_model.cjs` — Python/browser numerical parity checks.
8. GitHub Actions — repeatable training and testing on supported Python versions.

## Engineering and ML skills demonstrated

Python, NLP preprocessing, TF-IDF, multiclass logistic regression, train/validation/test discipline, leakage prevention, baseline comparison, macro F1, confusion-matrix analysis, error analysis, model export, inference API design, browser/Python parity, unit testing, CI, reproducibility, and scientific-data provenance.

## Why this project matters for internships

This project demonstrates more than a toy text classifier because it uses a large public scientific dataset with preserved official partitions and explicit error analysis. It shows the ability to:

- build a complete supervised NLP pipeline,
- evaluate against a simple baseline,
- keep test data outside model selection,
- expose a trained model through an API and browser demo,
- document both strengths and failure cases.

That evidence is directly relevant to ML Engineer, Applied AI, Data Science, NLP, research-engineering, and software-engineering internships.

## Important limitations

- This is a classical statistical NLP model, not a transformer, LLM, or neural language model.
- Vocabulary is intentionally limited to 1,000 features.
- Sentence position is not used as a feature.
- Probabilities are not calibrated guarantees.
- The task is scientific sentence-role classification only; it does not retrieve full papers, verify claims, summarize evidence, or provide medical advice.
- Performance on other scientific domains or general text has not been established.

## Recruiter takeaway

The strongest evidence in this project is the combination of **large real-world scientific text data, clean experimental separation, a measurable baseline gap, explicit failure analysis, reproducible training, and deployable inference**.

The result is intentionally described as a solid classical NLP system rather than overstated as modern large-scale deep learning.
