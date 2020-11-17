from gettext import install
from gensim.models import fasttext
from gensim.models import word2vec
import pandas as pd
import logging
import jieba
from appdirs import unicode
import re

model = word2vec.Word2Vec.load("word2vec.model")
words = model.wv.vocab.keys()
words = list(words)
# for i in words:
#     print(i)
# list = model["马"]
# trial = model.most_similar(positive=["美国", "特朗普"], negative=["中国"])
similarity = model.similarity('发射', '试射')
# similarity2 = model.similarity('美国', '美国')
print(similarity)
# print(similarity2)
