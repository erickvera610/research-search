# Local retrieval-augmented generation

Research Search now includes a real, local text-generation CLI using Google's pretrained FLAN-T5-small through PyTorch and Hugging Face Transformers. It retrieves up to two notes with the existing TF-IDF cosine retriever and generates an answer from their text. This is a new optional module; the PubMed classifier and browser demo remain separate.

## Reproduce (Python 3.12, CPU)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-genai.txt
python genai_qa.py 'Which algorithm predicts probabilities for classification?'
python evaluate_genai.py
python -m unittest discover -v
```

The first generation downloads model/tokenizer files from Hugging Face (hundreds of MB); later runs use the local cache. No paid API, API key or GPU is needed. Model revision is pinned to `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`; greedy decoding uses no sampling and at most 48 new tokens. Model loading failures raise errors, never canned fallback answers. Models are not redistributed in this repository. Tested torch version: 2.15.0+cpu.

## Concrete result

Question: **Which algorithm predicts probabilities for classification?**

Actual generated answer: **Logistic regression**.

Retrieved source: `Machine learning validation`, whose text states that logistic regression predicts probabilities for classification. The JSON output returns original source text, titles, lexical scores and matched terms. These are retrieval provenance, **not verified citations or proof that every generated claim is supported**. All generated answers are explicitly marked `generated_unverified`.

## Honest evaluation — November 19, 2026

The corpus consists of the repository's five short, authored educational notes in `core.py`, not medical abstracts or a production knowledge base. Eight hand-written development questions were fixed in `evaluate_genai.py` before running generation. There is no training, held-out test set or prompt tuning. The metric checks whether **all predefined answer fragments** occur in the normalized output. It does not assess factuality or semantic equivalence.

| Diagnostic question | Actual output | Fragment check |
|---|---|---|
| Algorithm predicting classification probabilities | Logistic regression | Pass |
| Practices improving software reliability | Continuous integration | Fail: omitted automated tests |
| Algorithms for shortest routes | Dijkstra and breadth first search | Pass |
| Database for transactions and parameterized queries | SQLite | Pass |
| Operator detecting image edges | Insufficient evidence | Fail: Sobel expected |
| Accuracy of logistic regression | Insufficient evidence | Pass: absent from notes |
| Cost of SQLite | Insufficient evidence | Pass: absent from notes |
| Unknown token `zzzxxyy` | Insufficient evidence. | Pass: deterministic retrieval gate |

**6/8 fragment checks passed.** Seven cases invoked the real model; the zero-overlap case skipped generation. This is a small development diagnostic, not a general accuracy estimate. Both failures remain in the raw [results](artifacts/genai-results.json), together with expected fragments, sources, model revision, dependency versions and SHA-256 hashes. All 16 Python tests and two JavaScript suites passed locally. Unit tests mock the generator to check orchestration; the recorded evaluation uses the actual model. No remote CI completion claimed.

## Boundaries and next steps

Lexical overlap can retrieve irrelevant evidence; no overlap can miss synonyms. No threshold has been calibrated. A prompt requesting abstention cannot guarantee it. Even retrieved correct evidence does not guarantee complete or correct generation, as the failures show. Input questions are limited to 500 characters, and contexts exceeding 512 model tokens are rejected rather than silently truncated. This demo uses trusted bundled notes; it has no adversarial-document or prompt-injection defense and must not be deployed with untrusted inputs as a reliable assistant. It is not medical advice or clinical decision support.

Next evaluation work should add a larger independent question set, source attribution assessment and adversarial/unsupported questions before claims of general quality. The present implementation demonstrates local model inference, retrieval integration, deterministic configuration, testable boundaries and transparent error analysis.

## Attribution

Independent AI-assisted portfolio work maintained by Erik Vera; integration and evaluation code are MIT licensed. The pretrained model is Google FLAN-T5-small, Apache 2.0: https://huggingface.co/google/flan-t5-small . Erik did not train or author the pretrained model. No employer data, personal data or secrets are used. The bundled authored notes remain under this repository's MIT license.
