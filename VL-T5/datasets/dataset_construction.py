import json
import csv
#This is the file for ground-truth training/testing/validation qrels 
train_file = '/Users/yuanyifei/MQC_NEW/facet_data/dev.qrel'
qrels = '/Users/yuanyifei/MQC_NEW/facet/all.qrel'
# facets = '/home/yyuan/MQC/facet_data/faceted_all_records.csv'
#This is the file that stores all the data including image & qa pairs & topics
facets = '/Users/yuanyifei/MQC_NEW/mqc/data/facet/facet.csv'
trains =  open(train_file,'r').readlines()
qrels = open(qrels,'r').readlines()
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

for t in trains:
    facet_id = t.split(' ')[0]
    doc_id = t.split(' ')[2]
    if facet_id not in doc_dict:
        doc_dict[facet_id] = []
    doc_dict[facet_id].append(doc_id)
    facet_doc_score[(facet_id,doc_id)] = t.strip('\n').split(' ')[-1][:-4]
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
l = [d for d in overlap_dict if len(overlap_dict[d])>0]
all_records = []
for k,facet in enumerate(doc_dict):
    record = {}
    record['facet_id'] = facet
    record['related_dict'] = overlap_dict[facet]
    record['topic'] = facet_topic[facet]
    record['question'] = facet_question[facet]
    record['answer'] = facet_answer[facet]
    record['img_ids'] = facet_img[facet]
    record['id'] = facet_img[facet]
    # record['img2'] = facet_img[facet][1]
    # record['img3'] = facet_img[facet][2]
    if len(overlap_dict[facet])>0:
        all_records.append(record)
print(len(all_records))
with open('/Users/yuanyifei/MQC_NEW/facet_data/dev.json','w') as f:
    json.dump(all_records,f)
