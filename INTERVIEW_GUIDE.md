# Interview Guide

## 60-second explanation

I built a scientific-text classification experiment using the official PubMed 20k RCT train, development, and test splits. The baseline model uses TF-IDF with multinomial logistic regression and reaches 76.72% test accuracy with 0.702 macro F1, far above the majority baseline. I added a lightweight PyTorch EmbeddingBag classifier so I can compare a neural text representation with an interpretable sparse baseline using the same split discipline. The project also includes browser/Python inference parity, error analysis, robustness tests, and an API.

## Questions I should be able to answer

- Why use macro F1 for five imbalanced classes?
- Why keep abstracts separated across train/dev/test?
- How does TF-IDF work?
- What does multinomial logistic regression learn?
- How does EmbeddingBag differ from TF-IDF?
- Why might a simple TF-IDF model beat a small neural baseline?
- What would a transformer-based extension change?
- How do you avoid overstating this as semantic search or an LLM?

## Strong limitation statement

The main classifier is supervised sentence-role classification on one benchmark. The PyTorch model is intentionally lightweight and is not a transformer, pretrained language model, clinical decision system, or large-scale NLP deployment.

## 20-second recruiter version

Real scientific NLP project with official dataset splits, TF-IDF/logistic regression, a PyTorch neural baseline, macro-F1 evaluation, confusion/error analysis, robustness tests, browser/API inference, Docker, and CI.
