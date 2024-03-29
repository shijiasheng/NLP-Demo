from gensim.models import word2vec
import os
import csv
from textrank4zh import TextRank4Keyword
model = word2vec.Word2Vec.load("word2vec.model")
os.chdir('F:\\语料\\文章_事件结果（存在不完整事件）')
directory = os.listdir()
i = 0
header = ['name', 'score', 'event_text_count', 'count', 'event_text', 'mine', 'text']
dictionaries = []
with open("F:\\语料\\test2.csv", 'w', newline='', encoding='utf-8')as f:
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
        source = []
        target = []

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
                source.append(clean_keywords[6 + j * 9].replace(bytes('Source\t'.encode('utf-8')), bytes(''.encode('utf-8'))))
                target.append(clean_keywords[7 + j * 9].replace(bytes('Target\t'.encode('utf-8')), bytes(''.encode('utf-8'))))
                # out_file.write(clean_keywords[8 + j * 9].replace(bytes('eventtext\t'.encode('utf-8')), bytes(''.encode('utf-8'))))
                # print(j)
            score = 0.00000000000
            count = 0.00000000000
            score = float(score)
            count = float(count)
            event_text_count = 0
            for et in eventtext:
                event_text_count = event_text_count + 1
            for item in tr4w.get_keywords(50, word_min_len=1):
                for et in eventtext:
                    # print(item.word)
                    et = et.decode("utf8")
                    et = et.strip('\t')
                    et = et.strip('\r')
                    # print(et)
                    et = str(et)
                    if item.word in model and et in model:
                        similarity = model.similarity(item.word, et)
                        if similarity >= 0.5:
                            score = score + item.weight * similarity
                            count = count + 1
                    else:
                        if item.word == et:
                            score = score + item.weight
                            count = count + 1
                # for sr in source:
                #     # print(item.word)
                #     sr = sr.decode("utf8")
                #     sr = sr.strip('\t')
                #     sr = sr.strip('\r')
                #     # print(et)
                #     sr = str(sr)
                #     if item.word == sr:
                #         score = score + item.weight
                #         count = count + 1
                # for tg in target:
                #     # print(item.word)
                #     tg = tg.decode("utf8")
                #     tg = tg.strip('\t')
                #     tg = tg.strip('\r')
                #     # print(et)
                #     tg = str(tg)
                #     if item.word == tg:
                #         score = score + item.weight
                #         count = count + 1
            if event_text_count == 0:
                score = 0
            else:
                score = score/event_text_count
            # print(score)
            # out_file.write(score)
            dict ={}
            dict['name'] = str(i)
            dict['score'] = str(score)
            dict['event_text_count'] = str(event_text_count)
            dict['count'] = str(count)
            eventtext_new = []
            for et in eventtext:
                et = et.decode("utf8")
                et = et.strip('\t')
                et = et.strip('\r')
                et = str(et)
                eventtext_new.append(et)
            dict['event_text'] = str(eventtext_new)

            mine = []
            for item in tr4w.get_keywords(50, word_min_len=1):
                mine.append(item.word)
            dict['mine'] = str(mine)
            dict['text'] = str(clean_text[6].decode("utf8"))
            dictionaries.append(dict)
            print(dict)
        # if i == 1000:
        #     break

    f_csv.writerows(dictionaries)