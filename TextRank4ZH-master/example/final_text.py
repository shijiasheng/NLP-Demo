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
with open("F:\\语料\\test.csv", 'w', newline='', encoding='utf-8')as f:
    f_csv = csv.DictWriter(f, header)
    f_csv.writeheader()  # 写入列名

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



        with open("F:\\语料\\测试\\"+str(i)+".txt", "wb") as out_file:
            out_file.write(clean_text[6])
            out_file.write(bytes("\n", encoding="utf8"))
            out_file.write(clean_text[8])
            tr4w = TextRank4Keyword()
            in_text = clean_text[6]
            tr4w.analyze(text=in_text, lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象
            out_file.write(bytes("\n", encoding="utf8"))
            out_file.write(bytes("关键词：\n", encoding="utf8"))
            # print(tr4w.get_keywords(20, word_min_len=1))
            for item in tr4w.get_keywords(20, word_min_len=1):
                out_file.write(bytes(str(item.word)+" "+str(item.weight)+"\n", encoding="utf8"))
            # r = list(set(a).intersection())
            for j in range(size_clean_keywords):
                # if j == size_clean_keywords - 1:
                #     break
                # eventtext[j] = clean_keywords[-1+j*9].replace(bytes('eventtext\t'.encode('utf-8')), bytes(''.encode('utf-8')))
                eventtext.append(clean_keywords[8 + j * 9].replace(bytes('eventtext\t'.encode('utf-8')), bytes(''.encode('utf-8'))))
                out_file.write(clean_keywords[8 + j * 9].replace(bytes('eventtext\t'.encode('utf-8')), bytes(''.encode('utf-8'))))
                # print(j)
            score = 0.00000000000
            count = 0.00000000000
            score = float(score)
            count = float(count)
            for item in tr4w.get_keywords(20, word_min_len=1):
                for et in eventtext:
                    # print(item.word)
                    et = et.decode("utf8")
                    et = et.strip('\t')
                    et = et.strip('\r')
                    # print(et)
                    et = str(et)
                    if item.word == et:
                        score = score + item.weight
                        count = count + 1
                        # print("我们一样我们一样我们一样我们一样")
            if score == 0:
                score = 0
            else:
                score = score/count
            # print(score)
            # out_file.write(score)
            dict ={}
            dict['name'] = str(i)
            dict['score'] = str(score)
            dictionaries.append(dict)
            print(dict)
        if i == 1000:
            break

    f_csv.writerows(dictionaries)
            # eventtext = clean_keywords[8].replace(bytes('eventtext\t'.encode('utf-8')), bytes(''.encode('utf-8')))

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