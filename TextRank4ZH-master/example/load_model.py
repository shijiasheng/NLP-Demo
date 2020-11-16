from gettext import install
from gensim.models import fasttext
from gensim.models import word2vec
import pandas as pd
import logging
import jieba
from appdirs import unicode
import re


with open("F:\\语料\\文章.txt", "rb") as file:
    txt = file.read()
    txt = unicode(txt, "utf-8")
    # print(txt)
    sentences = re.split('([。！!.？?])',txt)
    # print(sentences)
    new_sents = []
    for i in range(int(len(sentences) / 2)):
        sent = sentences[2 * i] + sentences[2 * i + 1]
        new_sents.append(sent)
    # print(new_sents)
    final_sents = []
    for item in new_sents:
        item = item.strip()
        # final_sents.append(item)
        s = jieba.lcut(item)
        final_sents.append(s)
        # for ss in s:
        #     final_sents.append(ss)
    # print(final_sents)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(final_sents, sg=1, size=100, window=5, min_count=1, negative=3, sample=0.001, hs=1, workers=1)
    model.save("word2vec.model")

    words = model.wv.vocab.keys()
    words = list(words)
    # for i in words:
    #     print(i)
    list = model.most_similar("中国", topn=10)
    print(list)