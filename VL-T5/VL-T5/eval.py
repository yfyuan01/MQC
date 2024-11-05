import pickle
p1 = 0
p3 = 0
p5 = 0
p10 = 0
result = pickle.load(open('snap/result_t5_val.pkl','rb'))
result1 = pickle.load(open('snap/result_vlt5.pkl','rb'))
prediction = result['predictions']
ground_truth = result['targets']
id = result1['img_ids']
prediction1 = result1['predictions']
ground_truth1 = result1['targets']
for k in range(len(prediction)):
    answer = ground_truth[k].split(' [SEP] ')
    p1+=len([i for i in answer if i in prediction[k][:1]])/1.0
    p3 += len([i for i in answer if i in prediction[k][:3]]) / 3.0
    p5 += len([i for i in answer if i in prediction[k][:5]]) / 5.0
    p10 += len([i for i in answer if i in prediction[k][:10]]) / 10.0
print(len(prediction))
print(p1/len(prediction))
print(p3/len(prediction))
print(p5/len(prediction))
print(p10/len(prediction))
p1 = 0
p3 = 0
p5 = 0
p10 = 0
for k in range(len(prediction1)):
    answer = ground_truth1[k].split(' [SEP] ')
    p1+=len([i for i in answer if i in prediction1[k][:1]])/1.0
    p3 += len([i for i in answer if i in prediction1[k][:3]]) / 3.0
    p5 += len([i for i in answer if i in prediction1[k][:5]]) / 5.0
    p10 += len([i for i in answer if i in prediction1[k][:10]]) / 10.0
print(p1/len(prediction1))
print(p3/len(prediction1))
print(p5/len(prediction1))
print(p10/len(prediction1))
print(len(prediction1))
print('-------------')
# origin test
import csv
p1 = 0
p3 = 0
p5 = 0
p10 = 0
l=0
p11 = 0
p31 = 0
p51 = 0
p101 = 0
l1 = 0
all = list(csv.DictReader(open('/home/yyuan/MQC/first_phase_retrieval/Answer_data.csv')))
print(len(all))
id = result['ids']
for k,r in enumerate(id):
    ques_number = all[r]['question_id']
    if int(ques_number[1:])<=3941:
        answer = ground_truth[k].split(' [SEP] ')
        p1 += len([i for i in answer if i in prediction[k][:1]]) / 1.0
        p3 += len([i for i in answer if i in prediction[k][:3]]) / 3.0
        p5 += len([i for i in answer if i in prediction[k][:5]]) / 5.0
        p10 += len([i for i in answer if i in prediction[k][:10]]) / 10.0
        l+=1
    else:
        answer = ground_truth[k].split(' [SEP] ')
        p11 += len([i for i in answer if i in prediction[k][:1]]) / 1.0
        p31 += len([i for i in answer if i in prediction[k][:3]]) / 3.0
        p51 += len([i for i in answer if i in prediction[k][:5]]) / 5.0
        p101 += len([i for i in answer if i in prediction[k][:10]]) / 10.0
        l1 += 1
print(p1/float(l))
print(p3/float(l))
print(p5/float(l))
print(p10/float(l))
print('')
print(p11/float(l1))
print(p31/float(l1))
print(p51/float(l1))
print(p101/float(l1))
print('-------------------------')


p1 = 0
p3 = 0
p5 = 0
p10 = 0
l=0
p11 = 0
p31 = 0
p51 = 0
p101 = 0
l1 = 0
for k,r in enumerate(id):
    ques_number = all[r]['question_id']
    if int(ques_number[1:])<=3941:
        answer = ground_truth1[k].split(' [SEP] ')
        p1 += len([i for i in answer if i in prediction1[k][:1]]) / 1.0
        p3 += len([i for i in answer if i in prediction1[k][:3]]) / 3.0
        p5 += len([i for i in answer if i in prediction1[k][:5]]) / 5.0
        p10 += len([i for i in answer if i in prediction1[k][:10]]) / 10.0
        l+=1
    else:
        answer = ground_truth1[k].split(' [SEP] ')
        p11 += len([i for i in answer if i in prediction1[k][:1]]) / 1.0
        p31 += len([i for i in answer if i in prediction1[k][:3]]) / 3.0
        p51 += len([i for i in answer if i in prediction1[k][:5]]) / 5.0
        p101 += len([i for i in answer if i in prediction1[k][:10]]) / 10.0
        l1 += 1
print(p1/float(l))
print(p3/float(l))
print(p5/float(l))
print(p10/float(l))
print('')
print(p11/float(l1))
print(p31/float(l1))
print(p51/float(l1))
print(p101/float(l1))
print('---------')

# answers from ClariQ
result2 = pickle.load(open('snap/result_vlt5_clari.pkl','rb'))
prediction2 = result2['predictions']
ground_truth2 = result2['targets']
p1 = 0
p3 = 0
p5 = 0
p10 = 0
for k in range(len(prediction2)):
    answer = ground_truth2[k].split(' [SEP] ')
    p1+=len([i for i in answer if i in prediction2[k][:1]])/1.0
    p3 += len([i for i in answer if i in prediction2[k][:3]]) / 3.0
    p5 += len([i for i in answer if i in prediction2[k][:5]]) / 5.0
    p10 += len([i for i in answer if i in prediction2[k][:10]]) / 10.0
print(p1/len(prediction2))
print(p3/len(prediction2))
print(p5/len(prediction2))
print(p10/len(prediction2))

results_val = pickle.load(open('snap/result_t5_val.pkl','rb'))
results_val1 = pickle.load(open('snap/result_vlt5_val.pkl','rb'))
pos = []
neg = []
prediction1 = results_val['predictions']
prediction2 = results_val1['predictions']
targets1 = results_val['targets']
targets2 = results_val1['targets']
id = results_val['ids']
for i in range(len(prediction1)):
    pre = prediction1[i]
    tar = targets1[i]
    ans = tar.split(' [SEP] ')
    p1 = len([i for i in ans if i in pre[:1]]) / 1.0
    p3 = len([i for i in ans if i in pre[:3]]) / 3.0
    p5 = len([i for i in ans if i in pre[:5]]) / 5.0
    p10 = len([i for i in ans if i in pre[:10]]) / 10.0
    pre = prediction2[i]
    tar = targets2[i]
    ans = tar.split(' [SEP] ')
    p11 = len([i for i in ans if i in pre[:1]]) / 1.0
    p31 = len([i for i in ans if i in pre[:3]]) / 3.0
    p51 = len([i for i in ans if i in pre[:5]]) / 5.0
    p101 = len([i for i in ans if i in pre[:10]]) / 10.0
    if (p1>p11 and p3>p31 and p5>p51 and p10>p101):
        neg.append(id[i])
    elif (p1<p11 and p3<p31 and p5<p51 and p10<p101):
        pos.append(id[i])
print(pos[:20])
print(neg[:20])
print(all[pos[0]])
print(all[pos[1]])
print(all[neg[0]])
print(all[neg[1]])








