# retrieve top10 documents given the topic
import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
import numpy as np
# nltk.download('punkt')
# nltk.download('stopwords')

def stem_tokenize(text, remove_stopwords=True):
    stemmer = PorterStemmer()
    tokens = [word for sent in nltk.sent_tokenize(text) \
                                      for word in nltk.word_tokenize(sent)]
    tokens = [word for word in tokens if word not in \
          nltk.corpus.stopwords.words('english')]
    return [stemmer.stem(word) for word in tokens]


# Files paths

request_file_path = 'answers_all09.csv'
doc_bank_path = '../mqc/data/cwdocs/cwdocs09.tsv'
run_file_path = './sample_runs/dev_bm25_09_original'

# Reads files and build bm25 corpus (index)

dev = pd.read_csv(request_file_path, sep=',')
doc_bank = pd.read_csv(doc_bank_path, sep='\t').fillna('')

# doc_bank['tokenized_doc_list'] = doc_bank['doc'].map(stem_tokenize)
# tokenized_doc_str_list = []
# for d in tqdm(doc_bank['doc']):
#     tokenized_doc_str = stem_tokenize(d)
#     tokenized_doc_str_list.append(tokenized_doc_str)
import pickle
with open('tokenized_09.pkl','rb') as f:
    tokenized_doc_str_list = pickle.load(f)
# with open('tokenized_12.pkl','wb') as f:
#     pickle.dump(tokenized_doc_str_list,f,protocol=1)
doc_bank['tokenized_doc_list']  = tokenized_doc_str_list
doc_bank['tokenized_doc_str'] = doc_bank['tokenized_doc_list'].map(lambda x: ' '.join(x))

bm25_corpus = doc_bank['tokenized_doc_list'].tolist()
bm25 = BM25Okapi(bm25_corpus)

# Runs bm25 for every query and stores output in file.

with open(run_file_path, 'w') as fo:
  for tid in tqdm(dev['Input.facet_id'].unique()):
    # print(tid)
    query = dev.loc[dev['Input.facet_id']==tid, 'Input.facet'].tolist()[0]
    topic = dev.loc[dev['Input.facet_id']==tid, 'Input.query'].tolist()[0]
    scores = bm25.get_scores(stem_tokenize(topic+' '+query))

    top_n = np.argsort(scores)[::-1][:100]
    scores = [scores[i] for i in top_n]
    # +' '+query
    bm25_ranked_list = bm25.get_top_n(stem_tokenize(topic, True),
                                    bm25_corpus,
                                    n=100)
    bm25_q_list = [' '.join(sent) for sent in bm25_ranked_list]
    # print(len(bm25_q_list))
    # print(len(doc_bank['tokenized_doc_str']))
    preds = [doc_bank['doc_id'][i] for i in top_n]
    # preds = doc_bank.set_index('tokenized_doc_str').loc[bm25_q_list, 'doc_id'].tolist()
    # print(len(preds))
    for i, qid in enumerate(preds):
      fo.write('{} 0 {} {} {} {} bm25\n'.format(tid, qid, i, len(preds)-i, scores[i]))