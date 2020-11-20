import numpy as np


def bert(sentence, tigger_pos=-1):
    '''
    sentecn is a string
    tigger_pos is pos of tig ,a index
    return the {hi……} in 100 p in LMST 论文
    '''
    from bert_serving.client import BertClient
    bc = BertClient(ip='100.64.250.144')
    res = bc.encode([sentence])
    ans = []
    for i in range(len(res[0])):
        ans.append(res[0][i])  # erase the read only

    return np.array(ans)



print(bert("测试"))