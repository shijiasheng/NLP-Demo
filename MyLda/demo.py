# 1.加载数据
# 该语料库包含商业、科技、运动、航空航天等多领域新闻资料
# from sklearn.datasets import fetch_20newsgroups
#
# dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
# data_samples = dataset.data[:2000]  # 截取需要的量，2000条
# print(data_samples)



# # 2.文本预处理, 可选项
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# 每次访问数据需要添加数据至路径当中
def textPrecessing(text):
    # 小写化
    text = text.lower()
    # 去除特殊标点
    for c in string.punctuation:
        text = text.replace(c, ' ')
    # 分词
    wordLst = nltk.word_tokenize(text)
    # 去除停用词
    filtered = [w for w in wordLst if w not in stopwords.words('english')]
    # 仅保留名词或特定POS
    refiltered = nltk.pos_tag(filtered)
    filtered = [w for w, pos in refiltered if pos.startswith('NN')]
    # 词干化
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]

    return " ".join(filtered)


# 该区域仅首次运行，进行文本预处理，第二次运行起注释掉
# docList = []
# for desc in data_samples:
#     docList.append(textPrecessing(desc).encode('utf-8'))
# with open('E:/NLP/NLP/MyLda/20newsgroups(2000).txt', 'a') as f:
#     for line in docList:
#         f.writelines(str(line) +
#                      '\n')

# ==============================================================================
# 从第二次运行起，直接获取预处理过的docLst，前面load数据、预处理均注释掉
docList = []
with open('E:/NLP/NLP/MyLda/20newsgroups(2000).txt', 'r') as f:
    for line in f.readlines():
        if line != '':
            docList.append(line.strip())
# ==============================================================================


# # 3.统计词频
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib  # 也可以选择pickle等保存模型，请随意
#
# #构建词汇统计向量并保存，仅运行首次 API: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
#                                 max_features=1500,
#                                 stop_words='english')
# tf = tf_vectorizer.fit_transform(docList)
# joblib.dump(tf_vectorizer, 'E:/NLP/NLP/MyLda/model/vec_model/vectorizer_sklearn.model')
# ==============================================================================
#得到存储的tf_vectorizer,节省预处理时间
tf_vectorizer = joblib.load('E:/NLP/NLP/MyLda/model/vec_model/vectorizer_sklearn.model')
tf = tf_vectorizer.fit_transform(docList)
# print(tf)
# # ==============================================================================


# 4.LDA主题模型训练
#API: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
from sklearn.decomposition import LatentDirichletAllocation

# # ===============================================================================
# # 训练并存储为模型，初次执行取消注释
# lda = LatentDirichletAllocation(n_components=20,  # 文章表示成20维的向量
#                                 max_iter=200,
#                                 learning_method='batch',
#                                 verbose=True)
# lda.fit(tf)  # tf即为Document_word Sparse Matrix
# joblib.dump(lda, 'E:/NLP/NLP/MyLda/model/LDA_model/LDA_sklearn.model')
# # ===============================================================================
#加载lda模型，初次请执行上面的训练过程，注释加载模型
lda = joblib.load('E:/NLP/NLP/MyLda/model/LDA_model/LDA_sklearn.model')

# print(lda.perplexity(tf))  # 收敛效果

texts = [
    "In this morning's TechBytes, we look back at the technology that changes the world in the past decade.\"5,4...\" As we counted down to 2000, fears Y2K would crash the world's computers had many questioning if we become too dependent on technology. Most of us had no idea just how hooked we get.Google was just a few years old then, a simple search engine with a loyal following. A few months later, it would explode into the world's largest. Today, it is the most visited site on the web, with over 1 billion searches everyday.\"The iPod, it's cute.\" MP3 players were nothing new when the first iPod was introduced in the fall of 2001, but this player from Apple was different.\"You can download 1,000 of your favourite songs from your Apple computer in less than 10 minutes.\"TV was revolutionized, too. HDTV, huge flat screens but the most life changing development— TiVo and the DVR. Now we can watch shows on our time and rewind to see something we missed. Today, more than 38 million US households have a DVR.\"People for 2001 are gonna wanna take it on the roads to see something like the Blackberry.""From this to this tiny thing?""Well...\" Little devices called Blackberries became Crackberries. Now, the office is always at your fingertips.And the decade brought friends closer together. Friendster and MySpace got it started, but Facebook took it mainstream.\"It's everyone's, like Santa, like life.\"At first, it was all college kids, but soon their parents and even grandparents followed. Today, Facebook is the second most visited site on the web with 350 million users.That was a look at some of the biggest tech stories of the past decade. For the latest tech news, log on to the technology page of abcnews.com. Those are your TechBytes. I'm Winnie Tanare.",
    "Movement is usually the sign of a person healthy, because only people who love sports will be healthy. I am a love sports, so I was born to now only had a disease. Of the many sports I like table tennis best.Table tennis is a sport, it does not hurt our friendship don't like football, in front of the play is a pair of inseparable friends, when the play is the enemy, the enemy after the play. When playing table tennis, as long as you aim at the ball back and go. If the wind was blowing when playing, curving, touch you, you can only on the day scold: \"it doesn't help me also. If is another person with technical won, you can only blame yourself technology is inferior to him. Table tennis is also a not injured movement, not like basketball, in play when it is pulled down, injured, or the first prize. When playing table tennis, even if be hit will not feel pain. I'm enjoying this movement at the same time, also met many table tennis masters, let my friends every day.",
    "While starting out on a business endeavour, following a set of rules is crucial for finding success.Without proper rules a business can go spiralling down and without taking too long at it. Following are golden rules that will ensure your success in business.Map it outMap where you want to head. Plant goals and results all across that mental map and keep checking it off once you start achieving them one by one.Care for your peoplePeople are your biggest asset. They are the ones who will drive your business to the top. Treat them well and they will treat you well, too.Aim for greatness.Build a great company. Build great services or products. Instil a fun culture at your workplace. Inspire innovation. Inspire your people to keep coming with great ideas, because great ideas bring great changes.Be wary.Keep a close eye on the people who you partner with. It doesn’t mean you have to be sceptical of them. But you shouldn’t naively believe everything you hear. Be smart and keep your eyes and ears open all the time.Commit and stick to it.Once you make a decision, commit to it and follow through. Give it your all. If for some reason that decision doesn’t work, retract, go back to the drawing board and pick an alternate route. In business, you will have to make lots of sacrifices. Be prepared for that. It will all be worth it in the end.Be proactive.Be proactive. Just having goals and not doing anything about them will not get you anywhere. If you don’t act, you will not get the results you’re looking for.Perfect timing.Anticipation is the key to succeed in business. You should have the skills to anticipate changes in the market place and, the changing consumer preferences. You have to keep a tab on all this. Never rest on your past laurels and always look to inject newness into your business processes.Not giving up.That’s the difference between those who succeed and those who don’t. As a businessman you should never give up, no matter what the circumstance. Keep on persevering. You will succeed sooner or later. The key is to never quit trying.Follow these rules and you'll find yourself scaling up the ladder of succcess."]

# 文本先预处理，再在词频模型中结构化，然后将结构化的文本list传入LDA主题模型，判断主题分布。
processed_texts = []
for text in texts:
    temp = textPrecessing(text)
    processed_texts.append(temp)
vectorizer_texts = tf_vectorizer.transform(processed_texts)
# print(vectorizer_texts)
# print(lda.transform(vectorizer_texts))  # 获得分布矩阵


# 5.结果
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