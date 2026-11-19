"""Local retrieval-augmented generation; no remote inference or API key."""
import argparse
import json
from pathlib import Path
from core import DOCUMENTS, search

MODEL_ID = 'google/flan-t5-small'
REVISION = '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab'
ABSTAIN = 'Insufficient evidence.'


def retrieve(question, documents=DOCUMENTS):
    if not isinstance(question, str) or not question.strip() or len(question) > 500:
        raise ValueError('Question must contain 1–500 characters')
    # Existing lexical retriever; positive overlap is not proof of answerability.
    return search(question, documents)[:2]


def prompt_for(question, sources):
    context = '\n'.join(s['text'] for s in sources)
    return f'Answer the question using the context. If the answer is absent, say Insufficient evidence.\nContext: {context}\nQuestion: {question}\nAnswer:'


class Generator:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        torch.set_num_threads(2)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, revision=REVISION, use_safetensors=True).eval()

    def __call__(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        if inputs.input_ids.shape[1] > 512:
            raise ValueError('Context exceeds 512 tokens; shorten documents')
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=48, do_sample=False, num_beams=1)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def answer(question, generator, documents=DOCUMENTS):
    sources = retrieve(question, documents)
    if not sources:
        return {'question': question, 'answer': ABSTAIN, 'status': 'no_lexical_match', 'sources': []}
    return {'question': question, 'answer': generator(prompt_for(question, sources)),
            'status': 'generated_unverified', 'sources': sources}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    args = parser.parse_args()
    sources = retrieve(args.question)
    generator = Generator() if sources else None
    print(json.dumps(answer(args.question, generator), indent=2))
