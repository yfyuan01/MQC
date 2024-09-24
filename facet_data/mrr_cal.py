import json
ground_truth = '/home/yyuan/MQC/mqc/data/facet/qrels'
ground_truth = open(ground_truth).readlines() #str(int([1:]))
ground_truth_dict = {}
# ground_truth_dict = {g.split(' ')[0]:g.split(' ')[2] for g in ground_truth if int(g.split(' ')[4])==0}
for q in ground_truth:
    facet_id = str(int(q.split(' ')[0][1:]))
    doc_id = q.split(' ')[2]
    if facet_id not in ground_truth_dict:
        ground_truth_dict[facet_id] = []
    if int(q.split(' ')[-1])>0:
        ground_truth_dict[facet_id].append(doc_id)
# print(ground_truth_dict['790'])
# test_file = '/home/yyuan/MQC/mqc/cedr/models/bert_1/test1.run'
# test_file = '/home/yyuan/MQC/qulac/src/result_ql.qrel'
# test_file = '/home/yyuan/MQC/VL-T5/VL-T5/snap/test_t5.run1'
test_file1 = '/home/yyuan/MQC/VL-T5/VL-T5/snap/result_vlt5_img1.run'
test_file2 = '/home/yyuan/MQC/VL-T5/VL-T5/snap/result_vlt5_img2.run'
test_file = '/home/yyuan/MQC/VL-T5/VL-T5/snap/result_vlt5_img3.run'


test_file = open(test_file).readlines()
test_file1 = open(test_file1).readlines()
test_file2 = open(test_file2).readlines()

test_result = {}
test_result1 = {}
test_result2 = {}

for t in test_file:
    facet = t.split(' ')[0]
    if facet not in test_result:
        test_result[facet] = []
        test_result[facet] = [t.split(' ')[2]]
    # test_result[facet].append(t.split(' ')[2])
for t in test_file1:
    facet = t.split(' ')[0]
    if facet not in test_result1:
        test_result1[facet] = []
        test_result1[facet] = [t.split(' ')[2]]

for t in test_file2:
    facet = t.split(' ')[0]
    if facet not in test_result2:
        test_result2[facet] = []
        test_result2[facet] = [t.split(' ')[2]]
mrr_all = 0
mrr_list = []
k = 0
for t in test_result:
    mrr = 0
    for j,tt in enumerate(test_result[t]):
        if tt in ground_truth_dict[t]:
            # k+=1
            mrr+=1.0/(j+1)
    if test_result1[t][0] in ground_truth_dict[t]:
        mrr1 = 1.0
    else:mrr1 = 0.
    if test_result2[t][0] in ground_truth_dict[t]:
        mrr2 = 1.0
    else:
        mrr2 = 0.
    mrr = max([mrr,mrr1,mrr2])
    # mrr = mrr/len(test_result[t])
    print(mrr)
    mrr_all+=mrr

# print(mrr/i)
print(mrr_all/len(test_result))
# print
