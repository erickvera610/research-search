"""Small PyTorch neural baseline for PubMed 20k RCT sentence-role classification.

This is intentionally lightweight: an EmbeddingBag encoder plus linear classifier.
It complements, rather than replaces, the stronger explainable TF-IDF/logistic model.
Model selection uses validation macro F1 and the official test split remains untouched.
"""
from pathlib import Path
from collections import Counter
import json, re
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from train_model import read_split
from model import ROOT

SEED=42
TOKEN_RE=re.compile(r'[a-z][a-z]+')
np.random.seed(SEED); torch.manual_seed(SEED)

def tokens(text): return TOKEN_RE.findall(text.lower())

def build_vocab(texts,max_words=12000,min_df=3):
    counts=Counter()
    for text in texts: counts.update(set(tokens(text)))
    words=[w for w,c in counts.most_common() if c>=min_df][:max_words-2]
    return {'<pad>':0,'<unk>':1,**{w:i+2 for i,w in enumerate(words)}}

def encode(text,vocab):
    ids=[vocab.get(w,1) for w in tokens(text)]
    return ids or [1]

def collate(batch):
    flat=[]; offsets=[0]; labels=[]
    for ids,label in batch:
        flat.extend(ids); labels.append(label); offsets.append(len(flat))
    return torch.tensor(flat,dtype=torch.long),torch.tensor(offsets[:-1],dtype=torch.long),torch.tensor(labels,dtype=torch.long)

class TextNet(nn.Module):
    def __init__(self,vocab_size,classes,dim=64):
        super().__init__(); self.emb=nn.EmbeddingBag(vocab_size,dim,mode='mean'); self.head=nn.Linear(dim,classes)
    def forward(self,ids,offsets): return self.head(self.emb(ids,offsets))

def predict(model,data,batch=512):
    loader=DataLoader(data,batch_size=batch,shuffle=False,collate_fn=collate); out=[]
    model.eval()
    with torch.no_grad():
        for ids,offsets,_ in loader: out.extend(model(ids,offsets).argmax(1).tolist())
    return np.array(out)

def main():
    tr,yt,_=read_split('train'); va,yv,_=read_split('dev'); te,ye,_=read_split('test')
    classes=sorted(set(yt)); label={c:i for i,c in enumerate(classes)}; vocab=build_vocab(tr)
    train=[(encode(t,vocab),label[y]) for t,y in zip(tr,yt)]; valid=[(encode(t,vocab),label[y]) for t,y in zip(va,yv)]; test=[(encode(t,vocab),label[y]) for t,y in zip(te,ye)]
    loader=DataLoader(train,batch_size=256,shuffle=True,collate_fn=collate)
    model=TextNet(len(vocab),len(classes)); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4); loss_fn=nn.CrossEntropyLoss()
    best=None; stale=0; history=[]
    yv_ids=np.array([label[y] for y in yv]); ye_ids=np.array([label[y] for y in ye])
    for epoch in range(30):
        model.train()
        for ids,offsets,y in loader:
            opt.zero_grad(); loss=loss_fn(model(ids,offsets),y); loss.backward(); opt.step()
        pv=predict(model,valid); score=float(f1_score(yv_ids,pv,average='macro')); history.append(score)
        if best is None or score>best[0]: best=(score,{k:v.detach().clone() for k,v in model.state_dict().items()},epoch+1); stale=0
        else:
            stale+=1
            if stale>=5: break
    model.load_state_dict(best[1]); pt=predict(model,test)
    result={'framework':torch.__version__,'architecture':'EmbeddingBag(mean,64) -> Linear(5)','vocab_size':len(vocab),'seed':SEED,
            'split':{'train':len(tr),'validation':len(va),'test':len(te)},'best_validation_macro_f1':best[0],'epochs':best[2],
            'test_accuracy':float(accuracy_score(ye_ids,pt)),'test_macro_f1':float(f1_score(ye_ids,pt,average='macro')),
            'confusion_matrix':confusion_matrix(ye_ids,pt).tolist(),
            'protocol':'Vocabulary built from train only; early stopping by validation macro F1; official test evaluated after selection.',
            'limitations':'Lightweight neural baseline, not a transformer or pretrained language model. Intended to demonstrate PyTorch training and comparison against the TF-IDF baseline.'}
    out=ROOT/'artifacts'; out.mkdir(exist_ok=True); (out/'pytorch-text-results.json').write_text(json.dumps(result,indent=2)); torch.save(model.state_dict(),out/'pytorch-text.pt'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
