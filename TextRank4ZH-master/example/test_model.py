from gensim.models import word2vec
import csv

model = word2vec.Word2Vec.load("word2vec.model")
words = model.wv.vocab.keys()
words = list(words)
# for i in words:
#     print(i)
# list = model["马"]
# trial = model.most_similar(positive=["特朗普", "中国"], negative=["习近平"])
# similarity = model.similarity('', '')
similarity2 = model.similarity('传递', '传达')
# print(trial)
print(similarity2)

worddic = []
csvfile = open('F:\\语料\\动词字典.csv', 'r')
reader = csv.DictReader(csvfile)
for row in reader:
    dict = {}
    dict['name'] = row['name']
    dict['number'] = row['number']
    worddic.append(dict)
    # print(row['name'],row['number'])
# print(worddic)
reader_words = []
for r in worddic:
    reader_words.append(r['name'])
dictionary = []
for i in reader_words:
    for j in reader_words:
        if i in model and j in model:
            dict = {}
            dict['i'] = i
            dict['j'] = j
            dict['similarity'] = model.similarity(i, j)
            dictionary.append(dict)
            # if dict['similarity'] < 1:
            #     print(i, j, dict['similarity'])
outfile = open('F:\\语料\\比对.csv', 'w')
header = ['i', 'j', 'similarity']
dictionary
f_csv = csv.DictWriter(outfile, header)
f_csv.writeheader()  # 写入列名
f_csv.writerows(dictionary)