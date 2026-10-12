"""Robustness checks for the trained PubMed sentence-role classifier.

The official test labels remain untouched by training. This script reports performance
by sentence length and under deterministic token dropout, which helps expose where the
TF-IDF model is brittle without claiming semantic robustness.
"""
from pathlib import Path
import json, re
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from train_model import read_split
from model import ROOT, load, predict

TOKEN_RE=re.compile(r'[a-z][a-z]+')

def token_dropout(text, every=5):
    words=text.split()
    return ' '.join(w for i,w in enumerate(words) if (i+1)%every)

def infer_many(model,texts):
    out=[]
    for text in texts: out.append(predict({'text':text},model)['prediction'])
    return np.array(out)

def main():
    model, _ = load()
    texts, y, _ = read_split('test')
    pred=infer_many(model,texts)
    lengths=np.array([len(TOKEN_RE.findall(t.lower())) for t in texts])
    groups={'short_0_12':lengths<=12,'medium_13_25':(lengths>12)&(lengths<=25),'long_26_plus':lengths>25}
    by_length={}
    for name,mask in groups.items():
        by_length[name]={'n':int(mask.sum()),'accuracy':float(accuracy_score(y[mask],pred[mask])),'macro_f1':float(f1_score(y[mask],pred[mask],average='macro'))}
    dropped=[token_dropout(t,5) for t in texts]
    pred_drop=infer_many(model,dropped)
    result={'clean':{'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro'))},
            'by_sentence_length':by_length,
            'token_dropout_20pct_approx':{'accuracy':float(accuracy_score(y,pred_drop)),'macro_f1':float(f1_score(y,pred_drop,average='macro')),
                                         'prediction_change_rate':float(np.mean(pred_drop!=pred))},
            'protocol':'Official test split only; deterministic removal of every fifth whitespace token; no robustness-specific tuning.',
            'limitations':'Token deletion is synthetic. These results do not establish clinical, semantic, or out-of-domain robustness.'}
    out=ROOT/'artifacts'; out.mkdir(exist_ok=True); (out/'robustness-results.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
