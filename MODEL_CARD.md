# Model card: scientific text classification

## What a reviewer can verify

[Run the public demo](https://erickvera610.github.io/research-search/model-demo.html). It uses the actual exported trained parameters, without a server account. The Python API uses the same parameters. `test_model.cjs` checks Python/browser numerical parity. Full test metrics and documented errors are in [model-results.json](artifacts/model-results.json), with model SHA-256 and data checksums in [SOURCE.json](data/SOURCE.json).

## Data and experiment

PubMed 20k RCT, from Dernoncourt and Lee (IJCNLP 2017), [paper](https://aclanthology.org/I17-2052/) and [upstream repository](https://github.com/Franck-Dernoncourt/pubmed-rct). Pinned upstream commit and SHA-256 are recorded. Downloaded original-text subset contains 180,040 train, 30,212 development and 30,135 test sentences. Official abstract-level partitions contain 15000, 2500, 2500 distinct abstracts; overlap across partitions is checked and equals zero.

TF-IDF vocabulary: at most 1,000 word features, lowercase ASCII token pattern `[a-z][a-z]+`, minimum training document frequency 5. Supervised multinomial logistic regression, candidates C=0.5 and C=2, selected by validation macro F1. Vocabulary and IDF are fitted only on train during selection, then refitted on train+development. No test text influences fitting. Sentence position is not an input. This is a trained statistical model, **not a neural network, transformer or LLM**.

| Held-out metric | Trained model | Majority-class baseline |
|---|---:|---:|
| Accuracy | 76.72% | 32.84% |
| Macro F1 | 0.7020 | 0.0989 |

Per-class precision/recall/F1, confusion matrix and 20 confidently wrong predictions are exported. Error references identify PMID and zero-based sentence index in the downloaded test file. Look up PMID via `https://pubmed.ncbi.nlm.nih.gov/<PMID>/`. The demo sentences are newly written illustrative inputs, not examples falsely presented as held-out evidence.

The model routes sentences to BACKGROUND, OBJECTIVE, METHODS, RESULTS or CONCLUSIONS. It can support scanning abstracts, but does not retrieve full papers, verify scientific claims or advise treatment. Uncalibrated probabilities are not guarantees. Empty and oversized text is rejected; text with no vocabulary overlap returns an explicit out-of-vocabulary result. Scientific abbreviations, ambiguous roles and other domains can fail.

Raw abstracts are downloaded for reproduction, ignored by git and not redistributed. Upstream states some abstracts may be protected by copyright; no blanket CC license is asserted. The code repository license does not change dataset rights.

## Reproduce and inspect

```bash
python -m pip install -r requirements-model.txt
python train_model.py
python -m unittest discover -v
node test_model.cjs
python model_server.py
```

Open http://127.0.0.1:8001. `POST /api/predict` accepts a JSON object, validates inputs and returns numeric model inference. `GET /health` reports server health. The public Pages demo runs inference in the browser; the Python API is local, not an always-on hosted endpoint. Existing original demos remain available.

GitHub Actions repeats training on Python 3.11/3.12, checks all unit tests and cross-language predictions, then uploads weights/results/source metadata as downloadable artifacts. Data downloads require network access. Floating-point values and optimization can vary across environments; stored exact hashes identify each exported run, not a promise of universal bitwise reproduction.

## Relation to applications

This is an independent, AI-assisted portfolio experiment. No employer affiliation or proprietary employer data. Application documents reviewed privately are not republished.

| Application context | Evidence this project provides | Gap remaining |
|---|---|---|
| DoorDash Software Engineer Intern, requisition 8171041 | Python implementation, validated prediction API, automated unit tests, GitHub CI, executable browser interface | Large-scale distributed operations, production monitoring and deployment are not demonstrated |
| Jump AI Research Engineer – Deep Learning Intern, requisition 8052338 | Explicit training/validation/test protocol, model export, benchmark comparisons and error analysis | No PyTorch/JAX/TensorFlow, CUDA, HPC, large-scale training, transformers, latency-sensitive trading or state-of-the-art research |
| Lilly technology resume previously prepared | Reproducible Python model evaluation and quantitative research communication | Resume is application context, not verified requirements for a specific Lilly vacancy |
| BCG analytics interests discussed | Baseline comparison, model limitations and clear quantitative evidence | No consulting engagement, business impact or employer endorsement claimed |

General Motors is not an active application target following the user's reported rejection. This work supports technical discussion; it does not establish eligibility or guarantee hiring.

## Ownership and interview preparation

Development used AI assistance. Automated daily fictional exercises are separate from these real-data experiments and do not establish manual work history. Before presenting the project, reproduce the experiment, explain the features/splits, inspect a failure and be able to change the implementation. Discuss measured results and remaining gaps precisely.
