import pickle
tests = open('/home/yyuan/MQC/qulac/src/result_ql.qrel').readlines()#bm25_test_original.qrel
import json
test_json = json.load(open('test.json'))
facets = [t['facet_id'] for t in test_json]
# with open('bm25_test_original.qrel','w') as f:
#     for t in tests:
#         if t.split(' ')[0] in facets and int(t.split(' ')[5])<=10:
#             f.write(str(int(t.split(' ')[0][1:]))+' Q'+' '.join(t.split(' ')[1:3])+' '+str(int(t.split(' ')[4])+1)+' '+t.split(' ')[-2]+' run\n')
result = '/home/yyuan/MQC/VL-T5/VL-T5/snap/result_t5.pkl'
result = pickle.load(open(result,'rb'))
result_index = {t['topic']+' '+t['question']+' '+t['answer']:str(int(t['facet_id'][1:])) for t in test_json}
answer_index = {result_index[result['questions'][i]]:result['targets'][i].split(' [SEP] ') for i in range(len(result['predictions']))}
topic_dict = pickle.load(open('/home/yyuan/MQC/VL-T5/datasets/keyword_dict.pkl','rb'))
bm25_result = {}
for t in tests:
    if t.split(' ')[0] not in bm25_result:
        bm25_result[t.split(' ')[0]] = []
    if t.split(' ')[2] in topic_dict:
        bm25_result[t.split(' ')[0]].append(topic_dict[t.split(' ')[2]])
p1 = 0.
p3 = 0.
p5 = 0.
p10 = 0.
print(list(answer_index.items())[0])
print(list(bm25_result.items())[0])
for b in bm25_result:
    p1 += len([k for k in bm25_result[b][:1] if k in answer_index[b]])/1.0
    p3 += len([k for k in bm25_result[b][:3] if k in answer_index[b]]) / 3.0
    p5 += len([k for k in bm25_result[b][:5] if k in answer_index[b]]) / 5.0
    p10 += len([k for k in bm25_result[b][:10] if k in answer_index[b]]) / 10.0
p1 = p1/len(bm25_result)
p3 = p3/len(bm25_result)
p5 = p5/len(bm25_result)
p10 = p10/len(bm25_result)
print(p1)
print(p3)
print(p5)
print(p10)