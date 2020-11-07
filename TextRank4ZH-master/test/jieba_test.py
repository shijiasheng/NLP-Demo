#-*- encoding:utf-8 -*-
from __future__ import print_function
from imp import reload
import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import jieba.posseg as pseg
words = pseg.cut("七月三十一日在台北“驻美代表处”举行记者会。代表团指出，此行与副助理国务卿、副助理国防部长、众院外交、军事等委员会主席、学者专家等人深入讨论，获致的结论是“美方没有冻结对台军售”。　　冻结对台军售 要求不要再提　　林郁方说，美方对台湾有些误解，“可能是用心，或者是台湾方面没有说实话”。例如有些美国学者说“台湾从来没有表示要买F-16C/D”。林郁方说，其实台湾“连续提了三次意向书(Letter of Request)，可是美国拒绝接受”。　　稍后林郁方举出具体日期，分别是二○○六年七月廿七日、○七年二月十三日以及○七年六月廿二日。他表示，美国三次退回，并且要求台湾不要再提，“从头到尾，我们没有错；但是美国也没有错，因为美国从来没有说过要卖F-16C/D给我们”。　　林郁方表示，美方或许在奥运过后会把军售案提交国会审议，但他认为潜舰和F-16C/D战机的希望不大。前者不在遭到延搁项目之中，后者则是货源无着。	")
for w in words:
    # print(w.word)
    print('{0} {1}'.format(w.word, w.flag))
    print(type(w.word))  # in py2 is unicode, py3 is str

