# -*- encoding:utf-8 -*-
from __future__ import print_function
from imp import reload
import sys
from bs4 import BeautifulSoup
import os
import csv
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
header = ['name', 'score']
dictionaries = []
with open("F:\\语料\\文章.txt", "wb") as out_file:
    for file in directory:
        i = i + 1
        text = open(file, 'rb').read()
        # print(text)

        clean_text = text.split(b'NULL', -1)
        # print(clean_text)

        keywords = clean_text[8]
        clean_keywords = keywords.split(b'\n', -1)

        size_clean_keywords = int((len(clean_keywords) - 11) / 9 + 1)

        # print(size_clean_keywords)
        eventtext = []
        source = []
        target = []
        out_file.write(clean_text[6])
        # if i == 100000:
        #     break
