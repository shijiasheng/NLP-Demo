# 1.加载数据
import codecs

# from sklearn.externals import joblib

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
print(cutcorpus)
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib  # 也可以选择pickle等保存模型，请随意
#
#构建词汇统计向量并保存，仅运行首次 API: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
tf_vectorizer = CountVectorizer(
                                max_features=1500,
                                stop_words='english')
tf = tf_vectorizer.fit_transform(cutcorpus)
joblib.dump(tf_vectorizer, 'E:/NLP/NLP/MyLda/model/vec_model/vectorizer_sklearn_ch.model')
# # ==============================================================================
#得到存储的tf_vectorizer,节省预处理时间
# tf_vectorizer = joblib.load('E:/NLP/NLP/MyLda/model/vec_model/vectorizer_sklearn_ch.model')
# tf = tf_vectorizer.fit_transform(cutcorpus)
print(tf)

# 4.LDA主题模型训练
#API: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
from sklearn.decomposition import LatentDirichletAllocation

# # ===============================================================================
# 训练并存储为模型，初次执行取消注释
lda = LatentDirichletAllocation(n_components=20,  # 文章表示成20维的向量
                                max_iter=200,
                                learning_method='batch',
                                verbose=True)
lda.fit(tf)  # newweight即为Document_word Sparse Matrix
joblib.dump(lda, 'E:/NLP/NLP/MyLda/model/LDA_model/LDA_sklearn_cn.model')
# # ===============================================================================
#加载lda模型，初次请执行上面的训练过程，注释加载模型
lda = joblib.load('E:/NLP/NLP/MyLda/model/LDA_model/LDA_sklearn_cn.model')

# print(lda.perplexity(tf))  # 收敛效果

# 5.LDA主题模型测试



#texts = [
#  "上面是一个简单的绘制效果的函数，下面调用一下这个函数，我们随意取一对参数值eps = 0.95, min_samples = 6, 使用我们的新特征newweight进行训练，代码和得到的结果如下所示：g down and without taking too long at it. Following are golden rules that will ensure your success in business.Map it outMap where you want to head. Plant goals and results all across that mental map and keep checking it off once you start achieving them one by one.Care for your peoplePeople are your biggest asset. They are the ones who will drive your business to the top. Treat them well and they will treat you well, too.Aim for greatness.Build a great company. Build great services or products. Instil a fun culture at your workplace. Inspire innovation. Inspire your people to keep coming with great ideas, because great ideas bring great changes.Be wary.Keep a close eye on the people who you partner with. It doesn’t mean you have to be sceptical of them. But you shouldn’t naively believe everything you hear. Be smart and keep your eyes and ears open all the time.Commit and stick to it.Once you make a decision, commit to it and follow through. Give it your all. If for some reason that decision doesn’t work, retract, go back to the drawing board and pick an alternate route. In business, you will have to make lots of sacrifices. Be prepared for that. It will all be worth it in the end.Be proa"]

corpus = []
file = codecs.open("F:\\语料\\sample.txt","r","utf-8")
for line in file.readlines():
    corpus.append(line.strip())

stripcorpus1 = corpus.copy()

print(stripcorpus1)
# 文本先预处理，再在词频模型中结构化，然后将结构化的文本list传入LDA主题模型，判断主题分布。
processed_texts = []

for text in stripcorpus1:
    temp, temp1 = textPrecessing(stripcorpus1)
    vectorizer_texts = tf_vectorizer.transform(temp1)

print(vectorizer_texts)
print(lda.transform(vectorizer_texts)) # 获得分布矩阵


def print_top_words(model, feature_names, n_top_words):
    # 打印每个主题下权重较高的term
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    # 打印主题-词语分布矩阵
    print(model.components_)


n_top_words = 20
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)