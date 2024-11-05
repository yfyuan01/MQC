import json
import csv
from copy import deepcopy
# train_file = '/home/yyuan/MQC/mqc/data/facet/test.qrel'
first_retrieval_results = '../../first_phase_retrieval/sample_runs/bm25.qrel'
qrels = '../../mqc/data/facet/qrels'
train_qrels = '../../facet_data/dev.qrel'
facets = '../../mqc/data/facet/facet.csv'
all_records = '../../first_phase_retrieval/Answer_data_clip.csv'
# all_records = '/home/yyuan/MQC/ClariQ/dev.tsv.1'
# all_records1 = '/home/yyuan/MQC/ClariQ/train.tsv'
# all_records2 = '/home/yyuan/MQC/ClariQ/test_with_labels.tsv'
all =  open(all_records,'r').readlines()
qrels = open(qrels,'r').readlines()
all_records = list(csv.DictReader(open(all_records)))
train_facet = list(set([i.split(' ')[0] for i in open(train_qrels,'r').readlines()]))
print(len(train_facet))
# all_records1 = list(csv.DictReader(open(all_records1), delimiter='\t'))
# all_records2 = list(csv.DictReader(open(all_records2), delimiter='\t'))
# all_records.extend(all_records1)
# all_records.extend(all_records2)
print(all_records[0].keys())
first_retrieval_results = open(first_retrieval_results,'r').readlines()
# facet_id = [t.split(' ')[0] for t in trains]
doc_dict = {}
qrels_dict = {}
facet_doc_score = {}
facets = list(csv.DictReader(open(facets)))
facet_topic = {f['facet_id']: f['topic'] for f in facets}
facet_question = {f['facet_id']:f['question'] for f in facets}
facet_answer = {f['facet_id']:f['answer'] for f in facets}
facet_img = {}
import pickle
# img_features = pickle.load(open('/ivi/ilps/personal/yyuan/img_feature_all.pkl','rb'))
for f in facets:
    facet_img[f['facet_id']] = []
    for i in range(1,4):
        # if f[f'image{i}'] in img_features:
        facet_img[f['facet_id']].append(f[f'image{i}'])
# facet_img = {f['facet_id']:[f['image1'],f['image2'],f['image3']] for f in facets}
for t in first_retrieval_results:
    facet_id = t.split(' ')[0]
    doc_id = t.split(' ')[2]
    if facet_id not in doc_dict:
        doc_dict[facet_id] = []
    doc_dict[facet_id].append(doc_id)
    facet_doc_score[(facet_id,doc_id)] = t.strip('\n').split(' ')[-1][:-4]
# print(doc_dict['F0957'])
for q in qrels:
    facet_id = q.split(' ')[0]
    doc_id = q.split(' ')[2]
    if facet_id not in qrels_dict:
        qrels_dict[facet_id] = []
    if int(q.split(' ')[-1])>0:
        qrels_dict[facet_id].append(doc_id)
overlap_dict = {}
for facet in doc_dict:
    overlap_dict[facet] = [i for i in doc_dict[facet] if i in qrels_dict[facet]]
for f in overlap_dict:
    overlap_dict[f] = sorted(overlap_dict[f],key=lambda x:facet_doc_score[(f,x)],reverse=True)

# overlap_dict = {f:sorted(])for f in overlap_dict}
all_records_json = []
# train_json = json.load(open('train.json'))
# all_records_json1 = []
for k,a in enumerate(all_records):
    record = {}
    facet = a['facet_id']
    record['facet_id'] = facet
    record['related_dict'] = overlap_dict[facet]
    record['topic'] = facet_topic[facet]
    record['question'] = a['question']
    record['answer'] = a['answer']
    record['img_ids']=[a['image1'],a['image2'],a['image3']]
    # record['img_ids'] = facet_img[facet]
    # if len(overlap_dict[facet])>0:
    #     all_records_json1.append(record)
    record1 = deepcopy(record)
    record['id'] = k
    # record['img2'] = facet_img[facet][1]
    # record['img3'] = facet_img[facet][2] and record1 not in train_json:
    if len(overlap_dict[facet])>0 and facet in train_facet:
        all_records_json.append(record)

# print(len(all_records_json))
# all_records_json = [i for i in all_records_json1 if i not in train_json]
print(len(all_records_json))

# for facet in doc_dict:
#     record = {}
#     record['facet_id'] = facet
#     record['related_dict'] = overlap_dict[facet]
#     record['topic'] = facet_topic[facet]
#     record['question'] = facet_question[facet]
#     record['answer'] = facet_answer[facet]
#     record['img_ids'] = facet_img[facet]
    # record['img2'] = facet_img[facet][1]
    # record['img3'] = facet_img[facet][2]
    # if len(overlap_dict[facet])>0:
    #     all_records.append(record)
# print(len(all_records))
with open('dev_clip.json','w') as f:
    json.dump(all_records_json,f)
