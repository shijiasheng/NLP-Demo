# -*- encoding:utf-8 -*-
from __future__ import print_function
from imp import reload
import sys
from bs4 import BeautifulSoup
import os
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass
import codecs
from textrank4zh import TextRank4Keyword, TextRank4Sentence
os.chdir('F:\\语料\\文章_事件结果（存在不完整事件）')
directory = os.listdir()
i = 0
for file in directory:
    i = i + 1
    text = open(file, 'rb').read()
    # print(text)

    clean_text = text.split(b'NULL', -1)
    # print(clean_text)
    with open("F:\\语料\\测试\\"+str(i)+".txt", "wb") as out_file:
        out_file.write(clean_text[6])
        out_file.write(bytes("\n", encoding="utf8"))
        out_file.write(clean_text[8])
        tr4w = TextRank4Keyword()
        in_text = clean_text[6]
        tr4w.analyze(text=in_text, lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象
        out_file.write(bytes("\n", encoding="utf8"))
        out_file.write(bytes("关键词：\n", encoding="utf8"))
        for item in tr4w.get_keywords(20, word_min_len=1):
            out_file.write(bytes(str(item.word)+" "+str(item.weight)+"\n", encoding="utf8"))

        # print()
        # print('关键短语：')
        # for phrase in tr4w.get_keyphrases(keywords_num=20, min_occur_num=2):
        #     print(phrase)
        #
        # tr4s = TextRank4Sentence()
        # tr4s.analyze(text=text, lower=True, source='all_filters')
        #
        # print()
        # print('摘要：')
        # for item in tr4s.get_key_sentences(num=3):
        #     print(item.index, item.weight, item.sentence)