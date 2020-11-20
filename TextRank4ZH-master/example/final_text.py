from gensim.models import word2vec
import os
import csv
from textrank4zh import TextRank4Keyword

def get_information(file_in):
    text_in = open(file_in, 'rb').read()
    # 文章截取出来
    clean_text = text_in.split(b'NULL', -1)
    keywords = clean_text[8]
    clean_keywords = keywords.split(b'\n', -1)
    size_clean_keywords = int((len(clean_keywords) - 11) / 9 + 1)
    eventtext_in = []
    source_in = []
    target_in = []
    event_root_in = []
    for j in range(size_clean_keywords):
        eventtext_in.append(clean_keywords[8 + j * 9].replace(bytes('eventtext\t'.encode('utf-8')), bytes(''.encode('utf-8'))))
        source_in.append(clean_keywords[6 + j * 9].replace(bytes('Source\t'.encode('utf-8')), bytes(''.encode('utf-8'))))
        target_in.append(clean_keywords[7 + j * 9].replace(bytes('Target\t'.encode('utf-8')), bytes(''.encode('utf-8'))))
        event_root_in.append(clean_keywords[9 + j * 9].replace(bytes('eventroot\t'.encode('utf-8')), bytes(''.encode('utf-8'))))
    article = clean_text[6]
    result_in = clean_text[8]
    text = article.decode("utf8").split('\u3000\u3000')
    title_in = text[1]
    # print(article.decode("utf8"))
    # text = text.decode("utf8").split(' ')
    # print(article.decode("utf8"))
    # print(article.decode("utf8").strip('\t').strip('\r').replace('\u3000\u3000', ' '))
    clean_article_in = article.decode("utf8").strip('\t').strip('\r').replace('\u3000\u3000', ' ')
    return clean_article_in, article, title_in, result_in, eventtext_in, source_in, target_in, event_root_in

def get_score(eventtext, article_textrank, title_textrank):
    score = 0.00000000000
    count = 0
    score = float(score)
    event_text_count = len(eventtext)
    clean_article_textrank = []
    clean_title_textrank = []
    for item in title_textrank:
        for et in eventtext:
            et = et.decode("utf8")
            et = et.strip('\t')
            et = et.strip('\r')
            et = str(et)
            if item.word in model and et in model:
                similarity = model.similarity(item.word, et)
                if similarity >= 0.5:
                    score = score + item.weight * similarity
                    count = count + 1
                    clean_title_textrank.append(item)
            else:
                if item.word == et:
                    score = score + item.weight
                    count = count + 1
                    clean_title_textrank.append(item)
    for item in article_textrank:
        for et in eventtext:
            et = et.decode("utf8")
            et = et.strip('\t')
            et = et.strip('\r')
            et = str(et)
            if item.word in model and et in model:
                similarity = model.similarity(item.word, et)
                if similarity >= 0.5:
                    score = score + item.weight * similarity
                    count = count + 1
                    clean_article_textrank.append(item)
            else:
                if item.word == et:
                    score = score + item.weight
                    count = count + 1
                    clean_article_textrank.append(item)
    if event_text_count == 0:
        score = 0
    else:
        score = score / event_text_count
    return score, event_text_count, count, clean_title_textrank, clean_article_textrank

def get_textrank(txt):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=txt, lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象
    len_textrank = round(len(txt) / 1000 * 10) + 1
    return tr4w.get_keywords(len_textrank, word_min_len=1)

def get_title_textrank(txt):
    tr4w = TextRank4Keyword()
    tr4w.analyze(text=txt, lower=True, window=2)  # py2中text必须是utf8编码的str或者unicode对象，py3中必须是utf8编码的bytes或者str对象
    return tr4w.get_keywords(3, word_min_len=1)

def form_dictionary(name, score, event_text_count, article_textrank, count, eventtext, title_textrank, event_root, title, in_text):
    dict = {}
    dict['name'] = str(name)
    dict['score'] = str(score)
    dict['event_text_count'] = str(event_text_count)
    dict['textrank_count'] = str(len(article_textrank))
    dict['count'] = str(count)
    eventtext_new = []
    for et in eventtext:
        et = et.decode("utf8")
        et = et.strip('\t')
        et = et.strip('\r')
        et = str(et)
        eventtext_new.append(et)
    dict['event_text'] = str(eventtext_new)
    textrank = []
    for item in article_textrank:
        textrank.append(item.word)
    dict['textrank'] = str(textrank)
    title_textrank_new = []
    for item in title_textrank:
        title_textrank_new.append(item.word)
    dict['title_textrank'] = str(title_textrank_new)
    event_root_new = []
    for er in event_root:
        et = er.decode("utf8")
        et = et.strip('\t')
        et = et.strip('\r')
        et = str(et)
        event_root_new.append(et)
    dict['event_root'] = str(event_root_new)

    dict['title'] = str(title)
    dict['text'] = str(in_text.decode("utf8"))
    return dict
# 加载model
model = word2vec.Word2Vec.load("word2vec.model")
# 打开文件夹
os.chdir('F:\\语料\\文章_事件结果（存在不完整事件）')
directory = os.listdir()
# 文件名按照自然数排序
i = 0
# 输出字典的header
header = ['name', 'score', 'event_text_count', 'count', 'textrank_count', 'event_text', 'event_root', 'textrank', 'title_textrank', 'title', 'text']
dictionaries = []
with open("F:\\语料\\test2.csv", 'w', newline='', encoding='utf-8')as f:
    f_csv = csv.DictWriter(f, header)
    f_csv.writeheader()  # 写入列名

    for file in directory:
        i = i + 1
        # 文章截取出来
        clean_article, in_text, title, result, eventtext, source, target, event_root = get_information(file)

        article_textrank = get_textrank(in_text)
        title_textrank = get_title_textrank(title.encode('utf-8'))
        with open("F:\\语料\\测试\\"+str(i)+".txt", "wb") as out_file:
            out_file.write(in_text)
            out_file.write(bytes("\n", encoding="utf8"))
            out_file.write(result)
            out_file.write(bytes("\n", encoding="utf8"))
            out_file.write(bytes("关键词：\n", encoding="utf8"))
            # for item in article_textrank:
            #     out_file.write(bytes(str(item.word)+" "+str(item.weight)+"\n", encoding="utf8"))
        score, event_text_count, count, clean_title_textrank, clean_article_textrank = get_score(eventtext, article_textrank, title_textrank)
        article_textrank = clean_article_textrank
        title_textrank = clean_title_textrank

        dict = form_dictionary(i, score, event_text_count, article_textrank, count, eventtext, title_textrank, event_root, title, in_text)
        dictionaries.append(dict)
        print(dict)
        if i == 100:
            break

    f_csv.writerows(dictionaries)