import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report
from threadpoolctl import threadpool_limits
from model import *

def read_split(name):
    p=ROOT/'data'/f'{name}.txt';source=json.loads((ROOT/'data/SOURCE.json').read_text())
    if not p.exists():p.parent.mkdir(exist_ok=True);p.write_bytes(urllib.request.urlopen(source['base_url']+f'/{name}.txt').read())
    if digest(p)!=source['checksums'][name]:raise ValueError('Dataset checksum mismatch')
    texts=[];labels=[];ids=[];pid=None
    for line in p.read_text().splitlines():
        if line.startswith('###'):pid=line[3:]
        elif '\t' in line:
            label,text=line.split('\t',1);texts.append(text);labels.append(label);ids.append(pid)
    return texts,np.array(labels),ids

def train():
    tr,yt,it=read_split('train');va,yv,iv=read_split('dev');te,ye,ie=read_split('test')
    assert not(set(it)&set(iv) or set(it)&set(ie) or set(iv)&set(ie))
    vec=TfidfVectorizer(max_features=1000,min_df=5,token_pattern=r'[a-z][a-z]+',lowercase=True);xt=vec.fit_transform(tr);xv=vec.transform(va);scores={}
    with threadpool_limits(limits=1):
        for c in (.5,2):
            clf=LogisticRegression(C=c,max_iter=300).fit(xt,yt);scores[str(c)]=float(f1_score(yv,clf.predict(xv),average='macro'))
        c=float(max(scores,key=scores.get));vec.fit(tr+va);clf=LogisticRegression(C=c,max_iter=300).fit(vec.transform(tr+va),np.concatenate((yt,yv)));pred=clf.predict(vec.transform(te));probs=clf.predict_proba(vec.transform(te))
    model={'task':'research','vocabulary':{k:int(v) for k,v in vec.vocabulary_.items()},'idf':vec.idf_.tolist(),'weights':clf.coef_.tolist(),'biases':clf.intercept_.tolist(),'classes':clf.classes_.tolist(),'C':c,'token_pattern':'[a-z][a-z]+'}
    own=['We aimed to assess the effect of exercise on blood pressure.','Patients were randomly assigned to receive treatment or placebo.','The treatment group showed a significant decrease in blood pressure.','These findings support further investigation of the intervention.','Cardiovascular disease is a major public health problem.']
    assert np.allclose([list(predict({'text':t},model)['probabilities'].values()) for t in own],clf.predict_proba(vec.transform(own)))
    majority=max(set(yt),key=lambda c:int(sum(yt==c)));wrong=np.flatnonzero(pred!=ye)
    errors=[{'pmid':ie[i],'sentence_index':int(i),'actual':str(ye[i]),'predicted':str(pred[i]),'probability':float(probs[i].max())} for i in sorted(wrong,key=lambda i:probs[i].max(),reverse=True)[:20]]
    report={'task':'Scientific abstract sentence-role classification','dataset':'PubMed 20k RCT (original text)','sentences':{'train':len(tr),'validation':len(va),'test':len(te)},'abstracts':{'train':len(set(it)),'validation':len(set(iv)),'test':len(set(ie))},'abstract_id_overlap':0,'protocol':'Official abstract-level train/dev/test splits. Fit TF-IDF on train only for C selection by validation macro F1; refit on train+dev; evaluate test once. No positional features or sentence labels as inputs.','validation_macro_f1':scores,'selected_C':c,'test_accuracy':float(accuracy_score(ye,pred)),'test_macro_f1':float(f1_score(ye,pred,average='macro')),'baseline_accuracy':float(np.mean(ye==majority)),'baseline_macro_f1':float(f1_score(ye,np.full(len(ye),majority),average='macro')),'confusion_matrix':confusion_matrix(ye,pred,labels=clf.classes_).tolist(),'classes':clf.classes_.tolist(),'classification_report':classification_report(ye,pred,output_dict=True),'worst_errors':errors,'examples':[{'text':t,**predict({'text':t},model)} for t in own],'source_sha256':json.loads((ROOT/'data/SOURCE.json').read_text())['checksums'],'limits':'Supervised multinomial logistic regression with 1000 TF-IDF features; not a neural network, transformer, LLM, semantic search or clinical decision system. Single benchmark and uncalibrated probabilities. Dataset abstracts may be copyrighted; raw texts are downloaded for reproduction and not redistributed here. Demo sentences are newly written illustrative inputs, not held-out evidence.'}
    save(model,report);print(json.dumps({k:report[k] for k in ('sentences','test_accuracy','test_macro_f1','baseline_accuracy')}))
if __name__=='__main__':train()
