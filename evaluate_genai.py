"""Small transparent development diagnostic, NOT a held-out benchmark."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from genai_qa import Generator, answer, MODEL_ID, REVISION

CASES = [
 ('Which algorithm predicts probabilities for classification?', ['logistic regression'], 'Machine learning validation'),
 ('Which practices improve software reliability?', ['automated tests', 'continuous integration'], 'Software engineering practices'),
 ('Which algorithms find shortest routes through graph obstacles?', ['dijkstra', 'breadth first search'], 'Graph route planning'),
 ('Which database supports transactions and parameterized queries?', ['sqlite'], 'Database application design'),
 ('Which operator detects image edges?', ['sobel'], 'Computer vision foundations'),
 ('What is the accuracy of logistic regression?', ['insufficient evidence'], None),
 ('How much does SQLite cost?', ['insufficient evidence'], None),
 ('zzzxxyy', ['insufficient evidence'], None),
]

if __name__ == '__main__':
    generator = Generator()
    rows = []
    for question, expected, source in CASES:
        result = answer(question, generator)
        normalized = result['answer'].lower().rstrip('.')
        result.update(expected_fragments=expected, expected_source=source,
                      all_expected_fragments_present=all(x in normalized for x in expected),
                      expected_source_retrieved=None if source is None else any(s['title']==source for s in result['sources']))
        rows.append(result)
    report = {'model': MODEL_ID, 'revision': REVISION, 'device': 'cpu',
              'decoding': {'do_sample':False, 'num_beams':1, 'max_new_tokens':48},
              'scope':'8 hand-written development checks, no tuning or held-out generalization claim',
              'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','sentencepiece','safetensors']},
              'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['genai_qa.py','core.py','evaluate_genai.py']},
              'fragment_pass_count':sum(r['all_expected_fragments_present'] for r in rows), 'cases':rows}
    Path('artifacts/genai-results.json').write_text(json.dumps(report, indent=2)+'\n')
    for r in rows: print(r['question'], '=>', r['answer'], r['all_expected_fragments_present'])
