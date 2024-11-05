import pickle

import spacy
from tqdm import tqdm
import spacy_ke
docs = open('../../mqc/data/facet/cwdocs.tsv').readlines()
texts = [d.strip('\n').split('\t')[1] for d in docs]
ids = [d.strip('\n').split('\t')[0] for d in docs]
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("yake")
keyword_dict = {}
for i,text in tqdm(enumerate(texts)):
    try:
        doc = nlp(text)
        keywords = [str(i[0]) for i in doc._.extract_keywords(n=3)]
        keyword_dict[ids[i]] = ' '.join(keywords)
    except:
        print(ids[i])
keyword_dict = {i:keyword_dict[i] for i in keyword_dict if keyword_dict[i]!=''}
with open('keyword_dict.pkl','wb') as f:
    pickle.dump(keyword_dict,f,protocol=1)
print(len(keyword_dict))
print(len(set(list(keyword_dict.values()))))


