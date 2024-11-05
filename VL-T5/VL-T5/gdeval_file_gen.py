import pickle
result = pickle.load(open('snap/test_2_img.pkl','rb'))
keyword_dict = pickle.load(open('../datasets/keyword_dict.pkl','rb'))
keyword_dict = {keyword_dict[i]:i for i in keyword_dict}
with open('result.qrels','w') as f:
    for i in range(len(result['ids'])):
        facet = result['ids'][i][1:]
        for k,titles in enumerate(result['predictions'][i]):
            if titles in keyword_dict:
                doc_id = keyword_dict[titles]
                rank = str(k+1)
                score = float(result['scores'][i][k].cpu())
                f.write(facet+' Q0 '+doc_id+' '+rank+' '+str(score)+' result\n')
