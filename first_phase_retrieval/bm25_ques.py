# use bm25 to select the best question of each topic
import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
import numpy as np
def stem_tokenize(text, remove_stopwords=True):
  stemmer = PorterStemmer()
  tokens = [word for sent in nltk.sent_tokenize(text) \
                                      for word in nltk.word_tokenize(sent)]
  tokens = [word for word in tokens if word not in \
          nltk.corpus.stopwords.words('english')]
  return [stemmer.stem(word) for word in tokens]

request_file_path = 'Answer_data.csv'
question_bank_path = 'question_bank_MQC.tsv'
run_file_path = 'sample_runs/ques_bm25'


dev = pd.read_csv(request_file_path, sep=',')
question_bank = pd.read_csv(question_bank_path, sep=',').fillna('')

question_bank['tokenized_question_list'] = question_bank['question'].map(stem_tokenize)
question_bank['tokenized_question_str'] = question_bank['tokenized_question_list'].map(lambda x: ' '.join(x))

bm25_corpus = question_bank['tokenized_question_list'].tolist()
bm25 = BM25Okapi(bm25_corpus)

# Runs bm25 for every query and stores output in file.

with open(run_file_path, 'w') as fo:
  for tid in dev['topic_id'].unique():
    query = dev.loc[dev['topic_id']==tid, 'initial_request'].tolist()[0]
    bm25_ranked_list = bm25.get_top_n(stem_tokenize(query, True),
                                    bm25_corpus,
                                    n=30)
    bm25_q_list = [' '.join(sent) for sent in bm25_ranked_list]
    preds = question_bank.set_index('tokenized_question_str').loc[bm25_q_list, 'question_id'].tolist()
    for i, qid in enumerate(preds):
      fo.write('{} 0 {} {} {} bm25\n'.format(tid, qid, i, len(preds)-i))

