import pickle
import json
import numpy
t5 = pickle.load(open('result_t5.pkl','rb'))
# vlt5 = pickle.load(open('result_vlt5_ours.pkl','rb'))
vlt5 = pickle.load(open('test_2_img.pkl','rb'))
predicts_t5 = t5['predictions']
answers_t5 = t5['targets']
predicts_vlt5 = vlt5['predictions']
answers_vlt5 = vlt5['targets']
scores_t5 = t5['scores']
scores_vlt5 = vlt5['scores']
pos_list = []
p_vlt5 = []
p_t5 = []
s_vlt5 = []
s_t5 = []
p1 = 0.
print(len(t5['predictions']))
for i in range(len(t5['predictions'])):
    predict_t5 = predicts_t5[i]
    answer_t5 = answers_t5[i].split(' [SEP] ')
    score_t5 = numpy.mean(scores_t5[i].cpu().numpy())
    p1 = len([i for i in predict_t5[:1] if i in answer_t5])/1.0
    p3 = len([i for i in predict_t5[:3] if i in answer_t5]) / 3.0
    p5 = len([i for i in predict_t5[:5] if i in answer_t5])/5.0
    predict_vlt5 = predicts_vlt5[i]
    answer_vlt5 = answers_vlt5[i].split(' [SEP] ')
    score_vlt5 = numpy.mean(scores_vlt5[i].cpu().numpy())
    p11 = len([i for i in predict_vlt5[:1] if i in answer_vlt5]) / 1.0
    p31 = len([i for i in predict_vlt5[:3] if i in answer_vlt5]) / 3.0
    p51 = len([i for i in predict_vlt5[:5] if i in answer_vlt5]) / 5.0
    # p1 += sorted([p11,p12,p13])[1]
    # p3 += sorted([p31,p32,p33])[1]
    # p5 += sorted([p51,p52,p53])[1]
    # print(predict_vlt5)
    # print(answer_vlt5)
    # break
    # p11 = len([i for i in predict_vlt5[:1] if i in answer_vlt5]) / 1.0
    # p31 = len([i for i in predict_vlt5[:3] if i in answer_vlt5]) / 3.0
    # p51 = len([i for i in predict_vlt5[:5] if i in answer_vlt5]) / 5.0
    # if (p5+p3+p1)/3 >= (p51+p31+p11)/3:
    if p5<=p51 and p3<=p31 and p1<=p11 and (p3+p5+p1)/3!=(p31+p51+p11)/3:
        pos_list.append(i)
        p_t5.append((p3+p5+p1)/3)
        p_vlt5.append((p31+p51+p11)/3)
        s_vlt5.append(score_vlt5)
        s_t5.append(score_t5)
# print(p1/len(t5['predictions']))
# print(p3/len(t5['predictions']))
# print(p5/len(t5['predictions']))
print(len(pos_list))
# print(pos_list[0])
# print(sum(p1_vlt5)/len(p1_vlt5))
# print(sum(p1_t5)/len(p1_t5))
# train = json.load(open('/home/yyuan/MQC/VL-T5/datasets/test.json'))
# for t in train:
#     if t['facet_id'] in pos_list:
#         t['tag'] = 0
#     else:
#         t['tag'] = 1
# with open('/home/yyuan/MQC/VL-T5/datasets/test_new.json','w') as f:
#     json.dump(train,f)
val = json.load(open('../../datasets/test.json'))
# input_dict = {val[k]['facet_id']:k for k in range(len(val))}
#
# contents = [val[input_dict[p]] for p in pos_list]
# print(contents[4])
# print(p1_vlt5[0])
# print(p1_t5[0])
import csv
all_records = '../../../first_phase_retrieval/Answer_data.csv'
print(len(val))
# val = pos_list
val = [val[i] for i in pos_list]
# import random
# random_list = random.sample(range(len(val)),100)
# val = [val[i] for i in random_list]
facets = '../../../mqc/data/facet/facet.csv'
facets = list(csv.DictReader(open(facets)))
facet_topic = {f['facet_id']: f['facet'] for f in facets}
all_records = list(csv.DictReader(open(all_records)))
topic_id = {f['topic']: f['topic_id'] for f in all_records}
with open('pos_all.csv', 'w', newline='') as csvfile:
    fieldnames = ['facet','facet_id','topic','topic_id','answer','question','image1','image2','image3','score_uni','score_multi','confidence_uni','confidence_multi']
    # fieldnames = ['facet_id', 'facet','topic','question','answer','img1','img2','img3']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for k,v in enumerate(val):
        d = {}
        d['facet_id'] = v['facet_id']
        d['topic'] = v['topic']
        d['topic_id'] = topic_id[v['topic']]
        d['question'] = v['question']
        d['answer'] = v['answer']
        d['image1'] = v['img_ids'][0]
        d['image2'] = v['img_ids'][1]
        if len(v['img_ids'])>=3:
            d['image3'] = v['img_ids'][2]
        else:
            d['image3'] = ''
        d['facet'] = facet_topic[v['facet_id']]
        d['score_uni'] = p_t5[k]
        d['score_multi'] = p_vlt5[k]
        d['confidence_uni'] = s_t5[k]
        d['confidence_multi'] = s_vlt5[k]
        writer.writerow(d)

