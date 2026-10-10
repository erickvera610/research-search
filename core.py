import re,math
from collections import Counter
DOCUMENTS=[{'title':'Machine learning validation','text':'Training validation test splits prevent data leakage. Logistic regression predicts probabilities for classification.'},{'title':'Software engineering practices','text':'Version control pull requests automated tests and continuous integration improve software reliability.'},{'title':'Computer vision foundations','text':'Image filtering convolution Sobel edges and geometric transformations support computer vision.'},{'title':'Graph route planning','text':'Shortest path algorithms Dijkstra and breadth first search find routes through graph obstacles.'},{'title':'Database application design','text':'SQLite transactions parameterized queries validation and REST APIs support application tracking.'}]
def tokens(text): return re.findall(r'[a-z0-9]+',text.lower())
def search(query,documents=DOCUMENTS):
    if not isinstance(query,str) or not query.strip(): raise ValueError('Enter a query')
    if not documents:return []
    terms=[Counter(tokens(d['text'])) for d in documents]
    vocabulary=set().union(*(set(t) for t in terms))
    idf={t:math.log((len(documents)+1)/(1+sum(t in d for d in terms)))+1 for t in vocabulary}
    def vector(counter):return {t:c*idf[t] for t,c in counter.items() if t in idf}
    q=vector(Counter(tokens(query)));qn=math.sqrt(sum(v*v for v in q.values()))
    results=[]
    for doc,counter in zip(documents,terms):
        v=vector(counter);norm=math.sqrt(sum(x*x for x in v.values()))
        score=sum(x*v.get(t,0) for t,x in q.items())/(qn*norm) if qn and norm else 0
        if score>0: results.append({**doc,'score':round(score,4),'matched_terms':sorted(set(q)&set(v))})
    return sorted(results,key=lambda r:-r['score'])
def run(payload): return {'results':search(payload.get('query','')),'method':'TF-IDF cosine similarity; lexical retrieval, not an LLM'}
