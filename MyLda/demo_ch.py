# 1.加载数据
import codecs

from sklearn.externals import joblib

corpus = []
file = codecs.open("sample.txt","r","utf-8")
for line in file.readlines():
    corpus.append(line.strip())

stripcorpus = corpus.copy()
print(stripcorpus)


# # 2.文本预处理, 可选项
import jieba.posseg as pseg
def textPrecessing(text):
    onlycorpus = []
    for string in text:
        if(string == ''):
            continue
        else:
            if(len(string)<5):
                continue
            else:
                onlycorpus.append(string)
    cutcorpusiter = onlycorpus.copy()
    cutcorpus = onlycorpus.copy()
    cixingofword = []  # 储存分词后的词语对应的词性
    wordtocixing = []  # 储存分词后的词语
    for i in range(len(onlycorpus)):
        cutcorpusiter[i] = pseg.cut(onlycorpus[i])
        cutcorpus[i] = ""
        for every in cutcorpusiter[i]:
            cutcorpus[i] = (cutcorpus[i] + " " + str(every.word)).strip()
            cixingofword.append(every.flag)
            wordtocixing.append(every.word)
    # 自己造一个{“词语”:“词性”}的字典，方便后续使用词性
    word2flagdict = {wordtocixing[i]:cixingofword[i] for i in range(len(wordtocixing))}
    return word2flagdict,cutcorpus

word2flagdict,cutcorpus=textPrecessing(stripcorpus)
print(word2flagdict)
#该区域仅首次运行，进行文本预处理，第二次运行起注释掉
# docList = []
# for desc in data_samples:
#     docList.append(textPrecessing(desc).encode('utf-8'))
# with open('E:/NLP/NLP/MyLda/20newsgroups(2000).txt', 'a') as f:
#     for line in docList:
#         f.writelines(str(line) +
#                      '\n')

# ==============================================================================
# 从第二次运行起，直接获取预处理过的docLst，前面load数据、预处理均注释掉
# docList = []
# with open('E:/NLP/NLP/MyLda/20newsgroups(2000).txt', 'r') as f:
#     for line in f.readlines():
#         if line != '':
#             docList.append(line.strip())
# ==============================================================================


# # 3.统计词频
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
transformer = TfidfTransformer()#该类会统计每个词语的tf-idf权值
#第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(cutcorpus))
#获取词袋模型中的所有词语
word = vectorizer.get_feature_names()
print(word)
#将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()
wordflagweight = [1 for i in range(len(word))]   #这个是词性系数，需要调整系数来看效果
print(wordflagweight)

for i in range(len(word)):
    print(word[i])
    if(word[i]=="n"):  # 这里只是举个例子，名词重要一点，我们就给它1.1
        wordflagweight[i] = 1.2
    elif(word[i]=="vn"):
        wordflagweight[i] = 1.1
    elif(word[i]=="m"):  # 只是举个例子，这种量词什么的直接去掉，省了一步停用词词典去除
        wordflagweight[i] = 0
    else:                                         # 权重数值还要根据实际情况确定，更多类型还请自己添加
        continue

import numpy as np
wordflagweight = np.array(wordflagweight)
newweight = weight.copy()
for i in range(len(weight)):
    for j in range(len(word)):
        newweight[i][j] = weight[i][j]*wordflagweight[j]

print(newweight)
# # ==============================================================================


# 4.LDA主题模型训练
#API: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
from sklearn.decomposition import LatentDirichletAllocation

# # ===============================================================================
# 训练并存储为模型，初次执行取消注释
# lda = LatentDirichletAllocation(n_components=20,  # 文章表示成20维的向量
#                                 max_iter=200,
#                                 learning_method='batch',
#                                 verbose=True)
# lda.fit(newweight)  # newweight即为Document_word Sparse Matrix
# joblib.dump(lda, 'E:/NLP/NLP/MyLda/model/LDA_model/LDA_sklearn_cn.model')
# # ===============================================================================
#加载lda模型，初次请执行上面的训练过程，注释加载模型
lda = joblib.load('E:/NLP/NLP/MyLda/model/LDA_model/LDA_sklearn_cn.model')

# print(lda.perplexity(tf))  # 收敛效果

# 5.LDA主题模型测试

corpus = []
file = codecs.open("sample.txt","r","utf-8")
for line in file.readlines():
    corpus.append(line.strip())

stripcorpus = corpus.copy()

#texts = [
#    "上面是一个简单的绘制效果的函数，下面调用一下这个函数，我们随意取一对参数值eps = 0.95, min_samples = 6, 使用我们的新特征newweight进行训练，代码和得到的结果如下所示：g down and without taking too long at it. Following are golden rules that will ensure your success in business.Map it outMap where you want to head. Plant goals and results all across that mental map and keep checking it off once you start achieving them one by one.Care for your peoplePeople are your biggest asset. They are the ones who will drive your business to the top. Treat them well and they will treat you well, too.Aim for greatness.Build a great company. Build great services or products. Instil a fun culture at your workplace. Inspire innovation. Inspire your people to keep coming with great ideas, because great ideas bring great changes.Be wary.Keep a close eye on the people who you partner with. It doesn’t mean you have to be sceptical of them. But you shouldn’t naively believe everything you hear. Be smart and keep your eyes and ears open all the time.Commit and stick to it.Once you make a decision, commit to it and follow through. Give it your all. If for some reason that decision doesn’t work, retract, go back to the drawing board and pick an alternate route. In business, you will have to make lots of sacrifices. Be prepared for that. It will all be worth it in the end.Be proa"]

# 文本先预处理，再在词频模型中结构化，然后将结构化的文本list传入LDA主题模型，判断主题分布。
processed_texts = []


temp = textPrecessing(stripcorpus)
print(temp)
processed_texts.append(temp)

vectorizer_texts = processed_texts
print(vectorizer_texts)
print(lda.transform(vectorizer_texts))  # 获得分布矩阵


# def print_top_words(model, feature_names, n_top_words):
#     # 打印每个主题下权重较高的term
#     for topic_idx, topic in enumerate(model.components_):
#         print("Topic #%d:" % topic_idx)
#         print(" ".join([feature_names[i]
#                         for i in topic.argsort()[:-n_top_words - 1:-1]]))
#     print()
#     # 打印主题-词语分布矩阵
#     print(model.components_)
#
#
# n_top_words = 20
# tf_feature_names = tf_vectorizer.get_feature_names()
# print_top_words(lda, tf_feature_names, n_top_words)