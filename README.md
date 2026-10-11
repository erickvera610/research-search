# Research Search

Explainable local document retrieval with TF-IDF, cosine similarity, and matched terms.

An AI-assisted educational portfolio project prepared for Erik Vera's internship learning. This is a working prototype, not production software. Authorship and skills should be represented based on what you can explain and change independently.

## Try in your browser

[Open the live demo](https://erickvera610.github.io/research-search/). The static JavaScript demo runs on the visitor's device and cannot change repository code. The Python implementation below provides the local backend. ML random-number generators differ between runtimes, so exact metrics differ. Tracker browser storage is localStorage; Python storage is SQLite.

## Run in two minutes

Requires Python 3.11 or newer; no pip dependencies.

```bash
git clone https://github.com/erickvera610/research-search.git
cd research-search
python app.py
```

Open http://127.0.0.1:8000. On Windows, use `py app.py` if `python` is unavailable. Stop with Ctrl+C. Run `python -m unittest discover -v` to execute tests.

## Architecture

- `index.html`: responsive accessible interface; renders output with textContent.
- `app.py`: local HTTP service, `/health`, and validated JSON POST `/api/run`.
- `core.py`: domain logic separated from transport and presentation.
- `test_core.py`: deterministic behavior and failure cases.
- `.github/workflows/tests.yml`: Python 3.11 and 3.12 CI.
- `TUTORIAL_ES.md`: beginner Spanish guide and interview practice.

## API

POST a JSON object to `/api/run`; inputs are the same fields used by the interface. Responses are JSON. Invalid inputs return 400; unknown paths return 404. GET `/health` returns `{"status":"ok"}`.

## Design and limits

Un token es una palabra normalizada. TF cuenta repeticiones; IDF da más peso a palabras menos frecuentes. Cada documento se convierte en un vector. La similitud del coseno compara su dirección con la consulta. El puntaje no es una probabilidad. Esta búsqueda encuentra palabras, no entiende sinónimos ni genera respuestas. Puedes ampliar DOCUMENTS con textos propios sin compartir documentos privados.

The built-in HTTP server is for a local demonstration. There is no authentication, rate limiting, or public production deployment. Never enter personal application records on a publicly exposed demo. Bind to localhost by default. For Codespaces only, run `HOST=0.0.0.0 python app.py` and keep the forwarded port private; usage may consume your GitHub quota.

## Review and contribution workflow

Create a feature branch, make one small change, run the tests, commit it, push the branch, and open a pull request. Explain the change and validation. Do not fabricate collaboration or commit history. Suggested first improvement: add a new input option with one behavior test and update this README.

## References

- https://docs.python.org/3/library/http.server.html
- https://docs.github.com/en/actions/tutorials/build-and-test-code/python

License: MIT.

## Trained model with real data

[Interactive model demo](https://erickvera610.github.io/research-search/model-demo.html) · [Model card](MODEL_CARD.md) · [Training and tests](https://github.com/erickvera610/research-search/actions)

Install `pip install -r requirements-model.txt`, run `python train_model.py`, then `python -m unittest discover -v` and `node test_model.cjs`. Run `python model_server.py` for the connected Python inference API at http://127.0.0.1:8001.

The original demonstration remains available. This new experiment trains on downloaded public data with checksum verification and separate validation/test partitions. Exported weights execute in the browser and Python API; see the model card for limitations and the exact relation to internship skills.
