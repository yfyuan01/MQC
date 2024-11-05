import pickle
import json
t5 = pickle.load(open('result_t5.pkl','rb'))
# vlt5 = pickle.load(open('result_vlt5_ours.pkl','rb'))
vlt5_img1 = pickle.load(open('result_vlt5_img1.pkl','rb'))
vlt5_img2 = pickle.load(open('result_vlt5_img2.pkl','rb'))
vlt5_img3 = pickle.load(open('result_vlt5_img3.pkl','rb'))
predicts_t5 = t5['predictions']
answers_t5 = t5['targets']
# predicts_vlt5 = vlt5['predictions']
# answers_vlt5 = vlt5['targets']
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
for i in range(len(t5['predictions'])):
    predict_t5 = predicts_t5[i]
    answer_t5 = answers_t5[i].split(' [SEP] ')
    # p1 = len([i for i in predict_t5[:1] if i in answer_t5])/1.0
    # p3 = len([i for i in predict_t5[:3] if i in answer_t5]) / 3.0
    # p5 = len([i for i in predict_t5[:5] if i in answer_t5])/5.0
    # predict_vlt5 = predicts_vlt5[i]
    # answer_vlt5 = answers_vlt5[i].split(' [SEP] ')
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
    p1 += sorted([p11,p12,p13])[1]
    p3 += sorted([p31,p32,p33])[1]
    p5 += sorted([p51,p52,p53])[1]
    # print(predict_vlt5)
    # print(answer_vlt5)
    # break
    # p11 = len([i for i in predict_vlt5[:1] if i in answer_vlt5]) / 1.0
    # p31 = len([i for i in predict_vlt5[:3] if i in answer_vlt5]) / 3.0
    # p51 = len([i for i in predict_vlt5[:5] if i in answer_vlt5]) / 5.0
    # # if (p5+p3+p1)/3 >= (p51+p31+p11)/3:
    # if p5<=p51 and p3<=p31 and p1<=p11 and (p3+p5+p1)/3!=(p31+p51+p11)/3:
    #     pos_list.append(t5['ids'][i])
    # p1_t5.append(p5)
    # p1_vlt5.append(p51)
print(p1/len(t5['predictions']))
print(p3/len(t5['predictions']))
print(p5/len(t5['predictions']))
print(len(pos_list))
print(pos_list[0])
print(sum(p1_vlt5)/len(p1_vlt5))
print(sum(p1_t5)/len(p1_t5))
# train = json.load(open('/home/yyuan/MQC/VL-T5/datasets/test.json'))
# for t in train:
#     if t['facet_id'] in pos_list:
#         t['tag'] = 0
#     else:
#         t['tag'] = 1
# with open('/home/yyuan/MQC/VL-T5/datasets/test_new.json','w') as f:
#     json.dump(train,f)
val = json.load(open('/home/yyuan/MQC/VL-T5/datasets/test.json'))
input_dict = {val[k]['facet_id']:k for k in range(len(val))}

contents = [val[input_dict[p]] for p in pos_list]
print(contents[4])
print(p1_vlt5[0])
print(p1_t5[0])
# import csv
# all_records = '/home/yyuan/MQC/first_phase_retrieval/Answer_data.csv'
# val = [i for i in val if i['id'] in pos_list]
# import random
# random_list = random.sample(range(len(val)),100)
# val = [val[i] for i in random_list]
# facets = '/home/yyuan/MQC/mqc/data/facet/facets.tsv'
# facets = list(csv.DictReader(open(facets)))
# facet_topic = {f['facet_id']: f['facet'] for f in facets}
# with open('pos_all.csv', 'w', newline='') as csvfile:
#     fieldnames = ['facet','query','question','image1','image2','image3']
#     # fieldnames = ['facet_id', 'facet','topic','question','answer','img1','img2','img3']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     for v in val:
#         d = {}
#         # d['facet_id'] = v['facet_id']
#         d['query'] = v['topic']
#         d['question'] = v['question']
#         # d['answer'] = v['answer']
#         d['image1'] = v['img_ids'][0]
#         d['image2'] = v['img_ids'][1]
#         if len(v['img_ids'])>=3:
#             d['image3'] = v['img_ids'][2]
#         else:
#             d['image3'] = ''
#         d['facet'] = facet_topic[v['facet_id']]
#         writer.writerow(d)
#
