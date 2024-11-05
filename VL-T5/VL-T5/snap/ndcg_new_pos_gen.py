# turn the pkl file into ndcg required file
import pickle
import json
result_file = 'result_vlt5_ours.pkl'
test_file = '/home/yyuan/MQC/facet_data/train.json'
keyword_file = '/home/yyuan/MQC/VL-T5/datasets/keyword_dict.pkl'
qrels = 'all_qrels'
# results = pickle.load(open(result_file,'rb'))
import pickle
import json
# t5 = pickle.load(open('result_t5.pkl','rb'))
vlt5_img1 = pickle.load(open('neg.pkl','rb'))
vlt5_img2 = pickle.load(open('neg1.pkl','rb'))
vlt5_img3 = pickle.load(open('neg2.pkl','rb'))
# vlt5_img1 = pickle.load(open('result_vlt5_img1.pkl','rb'))
# vlt5_img2 = pickle.load(open('result_vlt5_img2.pkl','rb'))
# vlt5_img3 = pickle.load(open('result_vlt5_img3.pkl','rb'))
# predicts_t5 = t5['predictions']
# answers_t5 = t5['targets']
predicts_vlt5_img1 = vlt5_img1['predictions']
answers_vlt5_img1 = vlt5_img1['targets']
predicts_vlt5_img2 = vlt5_img1['predictions']
answers_vlt5_img2 = vlt5_img2['targets']
predicts_vlt5_img3 = vlt5_img3['predictions']
answers_vlt5_img3 = vlt5_img3['targets']
pos_list = []
p1_vlt5 = []
p1_t5 = []
p1 = 0.
p3 = 0.
p5 = 0.
results = {'predictions':[],'scores':[],'ids':[],'questions':[]}
tags = []
for i in range(len(vlt5_img1['predictions'])):
    # predict_t5 = predicts_t5[i]
    # answer_t5 = answers_t5[i].split(' [SEP] ')
    predict_vlt5_img1 = predicts_vlt5_img1[i]
    answer_vlt5_img1 = answers_vlt5_img1[i].split(' [SEP] ')
    predict_vlt5_img2 = predicts_vlt5_img2[i]
    answer_vlt5_img2 = answers_vlt5_img2[i].split(' [SEP] ')
    predict_vlt5_img3 = predicts_vlt5_img3[i]
    answer_vlt5_img3 = answers_vlt5_img3[i].split(' [SEP] ')
    p11 = len([i for i in predict_vlt5_img1[:1] if i in answer_vlt5_img1]) / 1.0
    p31 = len([i for i in predict_vlt5_img1[:3] if i in answer_vlt5_img1]) / 3.0
    p51 = len([i for i in predict_vlt5_img1[:5] if i in answer_vlt5_img1]) / 5.0
    p12 = len([i for i in predict_vlt5_img2[:1] if i in answer_vlt5_img2]) / 1.0
    p32 = len([i for i in predict_vlt5_img2[:3] if i in answer_vlt5_img2]) / 3.0
    p52 = len([i for i in predict_vlt5_img2[:5] if i in answer_vlt5_img2]) / 5.0
    p13 = len([i for i in predict_vlt5_img3[:1] if i in answer_vlt5_img3]) / 1.0
    p33 = len([i for i in predict_vlt5_img3[:3] if i in answer_vlt5_img3]) / 3.0
    p53 = len([i for i in predict_vlt5_img3[:5] if i in answer_vlt5_img3]) / 5.0
    p1 += max([p11,p12,p13])
    p3 += max([p31,p32,p33])
    p5 += max([p51,p52,p53])
    print(p51,p52,p53)
    a = [p51,p52,p53]
    b = [p31,p32,p33]
    c = [p11,p12,p13]
    tag1 = [i for i, x in enumerate(a) if x == max(a)]
    tag2 = [i for i, x in enumerate(b) if x == max(b)]
    tag3 = [i for i, x in enumerate(c) if x == max(c)]
    # tag = [i for i in tag2 if i in tag3]
    tags.append(tag3)
with open('tag_neg.pkl','wb') as f:
    pickle.dump(tags,f,protocol=1)



# tests = json.load(open(test_file))
# keywords = pickle.load(open(keyword_file,'rb'))
# qrels = open(qrels).readlines()
# qrels_dict = {(i.split(' ')[0],i.split(' ')[2]):int(i.strip('\n').split(' ')[-1]) for i in qrels}
# keywords1 = {}
# for k in keywords:
#     if keywords[k] not in keywords1:
#         keywords1[keywords[k]] = []
#     keywords1[keywords[k]].append(k)
# # keywords = {keywords[d]:d for d in keywords}
# result_index = {t['topic']+' '+t['question']+' '+t['answer']:t['facet_id'] for t in tests}
# with open('test_t5_change.run','w') as f:
#     for i in range(len(results['questions'])):
#         facet = results['ids'][i]
#         facet = str(int(facet[1:]))
#         for j,p in enumerate(results['predictions'][i]):
#             try:
#                 doc_id = keywords1[p]
#             except:
#                 if len([i for i in keywords1 if i.find(p)==0])==1:
#                     doc_id = keywords1[[i for i in keywords1 if i.find(p)==0][0]]
#             if len(doc_id)==1:
#                 doc_id = doc_id[0]
#             else:
#                 score = []
#                 for d in doc_id:
#                     if (facet,d) not in qrels_dict:
#                         score.append(-2)
#                     else:
#                         score.append(qrels_dict[(facet,d)])
#                 doc_id = doc_id[score.index(max(score))]
#                 print(doc_id)
#             score = results['scores'][i][j].cpu().item()
#             f.write(facet+' Q0 '+doc_id+' '+str(j+1)+' '+str(score)+' run\n')
# qrels = '/home/yyuan/MQC/qrels/all_facet.qrel'
# qrels = open(qrels).readlines()
# qrels = [q.split(' ') for q in qrels]
# with open('all_qrels','w') as f:
#     for q in qrels:
#         f.write(str(int(q[0][1:]))+' '+' '.join(q[1:]))
