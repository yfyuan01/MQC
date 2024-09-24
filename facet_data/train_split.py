import random
def data_split(full_list, ratio, shuffle=True):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    left = int((n_total-offset)/2)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:offset+left]
    sublist_3 = full_list[offset+left:]
    return sublist_1, sublist_2, sublist_3

all = open('all.qrel').readlines()
facets = list(set([i.split(' ')[0] for i in all]))
train, dev, test = data_split(facets, 0.8, shuffle=True)
with open('train.qrel','w') as f:
    for i in all:
        if i.split(' ')[0] in train:
            f.write(i)
with open('dev.qrel','w') as f:
    for i in all:
        if i.split(' ')[0] in dev:
            f.write(i)
with open('test.qrel','w') as f:
    for i in all:
        if i.split(' ')[0] in test:
            f.write(i)