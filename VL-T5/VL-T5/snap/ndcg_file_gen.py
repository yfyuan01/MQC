# turn the pkl file into ndcg required file
import pickle
import json
# result_file = 'result_vlt5_rand_img.pkl'
result_file = 'result_vlt5_clari.pkl'
test_file = '/home/yyuan/MQC/facet_data/train.json'
keyword_file = '/home/yyuan/MQC/VL-T5/datasets/keyword_dict.pkl'
qrels = 'all_qrels'
results = pickle.load(open(result_file,'rb'))
tests = json.load(open(test_file))
keywords = pickle.load(open(keyword_file,'rb'))
qrels = open(qrels).readlines()
qrels_dict = {(i.split(' ')[0],i.split(' ')[2]):int(i.strip('\n').split(' ')[-1]) for i in qrels}
keywords1 = {}
for k in keywords:
    if keywords[k] not in keywords1:
        keywords1[keywords[k]] = []
    keywords1[keywords[k]].append(k)
# keywords = {keywords[d]:d for d in keywords}
result_index = {t['topic']+' '+t['question']+' '+t['answer']:t['facet_id'] for t in tests}
with open('result_vlt5_clari.run','w') as f:
    for i in range(len(results['questions'])):
        facet = results['ids'][i]
        facet = str(int(facet[1:]))
        for j,p in enumerate(results['predictions'][i]):
            try:
                doc_id = keywords1[p]
            except:
                if len([i for i in keywords1 if i.find(p)==0])==1:
                    doc_id = keywords1[[i for i in keywords1 if i.find(p)==0][0]]
            if len(doc_id)==1:
                doc_id = doc_id[0]
            else:
                score = []
                for d in doc_id:
                    if (facet,d) not in qrels_dict:
                        score.append(-2)
                    else:
                        score.append(qrels_dict[(facet,d)])
                doc_id = doc_id[score.index(max(score))]
                print(doc_id)
            score = results['scores'][i][j].cpu().item()
            f.write(facet+' Q0 '+doc_id+' '+str(j+1)+' '+str(score)+' run\n')
# qrels = '/home/yyuan/MQC/qrels/all_facet.qrel'
# qrels = open(qrels).readlines()
# qrels = [q.split(' ') for q in qrels]
# with open('all_qrels','w') as f:
#     for q in qrels:
#         f.write(str(int(q[0][1:]))+' '+' '.join(q[1:]))
