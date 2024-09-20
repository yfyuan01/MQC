from tqdm import tqdm
file = open('../data/docs.tsv','rt',encoding = 'utf-8')
for i,line in tqdm(enumerate(file), desc='loading datafile (by line)', leave=False):
    if len(line.split('\t'))!=3:
        print(i)
        print(len(line.split('\t')))
