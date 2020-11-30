# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def save_ebedding(file_name):
    #file_name = "R.txt"
    ans = []
    senlen = 150  # is the verable from constant
    batch_size = 20  # same is from constant
    save_name = file_name[0] + "_wordEmb.npy"
    tigger_pos=[]
    kind=[]
    with open(file_name, encoding="utf-8") as f:
        for line in f:
            #print(line)
            if "content" in line:
                ans.append(line[10:-2])
            #这里应该是分词才对，但是这里实际上是采用了，对于tigger的字采取一种预处理的方式。
            if "eventtext" in line:
                idx=len(ans)-1
                #print("find "+line[11:-1]+" last ans is "+ans[idx])
                if len(line)<12:
                    tigger_pos.append(-1)
                    continue
                tigger=line[11:-1]
                tot_pos=0
                sum=0
                cc=0#从零开的位置编码生活
                for word in ans[idx]:
                    if word in tigger:
                        sum+=1
                        tot_pos+=cc
                    cc+=1
                tot_pos//=sum
                tigger_pos.append([tot_pos,cc])

            if "eventroot" in line:
                print(line.split()[1])
                kind.append(int(line.split()[1]))

    #print(tigger_pos)

    print(tigger_pos)


    print("save name is " + save_name)

    import numpy as np
    a = np.zeros([senlen])
    embedd_dir = "C:\\Users\\31577\\Desktop\\BERT_CACHE\\bert-base-uncased\\" + "vocab.txt"
    test = []
    embedd_index = {}
    num = 0
    with open(embedd_dir, encoding="utf-8") as ff:#pre deal
        for line in ff:
            # print(line)
            embedd_index[line[0:-1]] = num
            num += 1
    # print(embedd_index['姜'])   and this verable work fine

    final_res = np.zeros(shape=[batch_size, senlen], dtype=int)
    i = 0
    print(len(embedd_index))
    for sen in ans:
        ids = 0
        for word in sen:
            #print(word)
            if word not in embedd_index.keys():
                print("error"+word)
                continue
            mark = int(embedd_index[word])

            #pp = input()
            final_res[i][ids] = mark
            ids += 1
        i += 1
    #print(final_res)
    np.save(save_name, final_res)


    final_tigger_pos=np.zeros([batch_size,senlen],dtype=int)
    for i in range(batch_size):
        for j in range(senlen):
            if i<len(tigger_pos) and j<tigger_pos[i][0]:
                final_tigger_pos[i][j]=1

    Rmask=np.zeros([batch_size,senlen],dtype=int)
    for i in range(batch_size):
        for j in range(senlen):
            if i<len(tigger_pos) and j>tigger_pos[i][0] and j<tigger_pos[i][1]:
                Rmask[i][j]=1
    #print(final_tigger_pos)
    np.save(file_name[0]+"_maskL.npy",final_tigger_pos)
    np.save(file_name[0]+"_maskR.npy",Rmask)

    mask=np.zeros([batch_size,senlen],dtype=int)
    for i in range(batch_size):
        for j in range(senlen):
            if i<len(tigger_pos)  and j<tigger_pos[i][1]:
                mask[i][j]=1


    #print(mask)
    np.save(file_name[0]+"_inMask.npy",mask)


    labels=np.zeros([batch_size])
    for i in range(batch_size):
        if i<len(kind):
            labels[i]=kind[i]
    print(labels)
    np.save(file_name[0]+"_label.npy",labels)


def test_embedding(file_name):
    '''
    这个部分的函数暂时弃用，因为暂时还不需要使用
    :param file_name:
    :return:
    '''
    #file_name = "R.txt"
    ans = []
    senlen = 150  # is the verable from constant
    batch_size = 20  # same is from constant
    save_name = "TestA"+ "_wordEmb.npy"
    tigger_pos=[]
    kind=[]
    with open(file_name, encoding="utf-8") as f:
        for line in f:
            #print(line)
            if "content" in line:
                ans.append(line[10:-2])
            #这里应该是分词才对，但是这里实际上是采用了，对于tigger的字采取一种预处理的方式。
            if "eventtext" in line:
                idx=len(ans)-1
                #print("find "+line[11:-1]+" last ans is "+ans[idx])
                if len(line)<12:
                    tigger_pos.append(-1)
                    continue
                tigger=line[11:-1]
                tot_pos=0
                sum=0
                cc=0#从零开的位置编码生活
                for word in ans[idx]:
                    if word in tigger:
                        sum+=1
                        tot_pos+=cc
                    cc+=1
                tot_pos//=sum
                tigger_pos.append([tot_pos,cc])

            if "eventroot" in line:
                print(line.split()[1])
                kind.append(int(line.split()[1]))

    #print(tigger_pos)

    print(tigger_pos)


    print("save name is " + save_name)

    import numpy as np
    a = np.zeros([senlen])
    embedd_dir = "C:\\Users\\31577\\Desktop\\BERT_CACHE\\bert-base-uncased\\" + "vocab.txt"
    test = []
    embedd_index = {}
    num = 0
    with open(embedd_dir, encoding="utf-8") as ff:#pre deal
        for line in ff:
            # print(line)
            embedd_index[line[0:-1]] = num
            num += 1
    # print(embedd_index['姜'])   and this verable work fine

    final_res = np.zeros(shape=[batch_size, senlen], dtype=int)
    i = 0
    print(len(embedd_index))
    for sen in ans:
        ids = 0
        for word in sen:
            #print(word)
            if word not in embedd_index.keys():
                print("error"+word)
                continue
            mark = int(embedd_index[word])

            #pp = input()
            final_res[i][ids] = mark
            ids += 1
        i += 1
    #print(final_res)
    np.save(save_name, final_res)


    final_tigger_pos=np.zeros([batch_size,senlen],dtype=int)
    for i in range(batch_size):
        for j in range(senlen):
            if i<len(tigger_pos) and j<tigger_pos[i][0]:
                final_tigger_pos[i][j]=1

    Rmask=np.zeros([batch_size,senlen],dtype=int)
    for i in range(batch_size):
        for j in range(senlen):
            if i<len(tigger_pos) and j>tigger_pos[i][0] and j<tigger_pos[i][1]:
                Rmask[i][j]=1
    #print(final_tigger_pos)
    np.save("TestA"+"_maskL.npy",final_tigger_pos)
    np.save("TestA"+"_maskR.npy",Rmask)

    mask=np.zeros([batch_size,senlen],dtype=int)
    for i in range(batch_size):
        for j in range(senlen):
            if i<len(tigger_pos)  and j<tigger_pos[i][1]:
                mask[i][j]=1


    #print(mask)
    np.save("TestA"+"_inMask.npy",mask)


    labels=np.zeros([batch_size])
    for i in range(batch_size):
        if i<len(kind):
            labels[i]=kind[i]
    print(labels)
    np.save("TestA"+"_label.npy",labels)
# Press the green button in the gutter to run the script.


def prepare_data():
    l=339764
    r=349900

    embedd_index = {}
    num = 0
    embedd_dir = "C:\\Users\\31577\\Desktop\\BERT_CACHE\\bert-base-uncased\\" + "vocab.txt"
    with open(embedd_dir, encoding="utf-8") as ff:  # pre deal
        for line in ff:
            # print(line)
            embedd_index[line[0:-1]] = num
            num += 1

    ans=[]
    tigger_pos = []
    kind = []
    for i in range(l,r):
        file_name="event\\"+str(i)+"——evets.result_before_merge.txt"
        import os
        if not os.path.exists(file_name):
            continue
        with open(file_name, encoding="utf-8") as f:
            for line in f:

                #print(line)
                if "content" in line:
                    ans.append(line[10:-2])
                # 这里应该是分词才对，但是这里实际上是采用了，对于tigger的字采取一种预处理的方式。
                if "eventtext" in line:
                    idx = len(ans) - 1
                    # print("find "+line[11:-1]+" last ans is "+ans[idx])
                    if len(line) < 12:
                        tigger_pos.append(-1)
                        continue
                    tigger = line[11:-1]
                    pos=ans[idx].find(tigger)
                    tigger_pos.append(pos)

                if "eventroot" in line:
                    if len(line)<=12:
                        kind.append(0)
                        continue
                    num=line.split()[1]
                    if ":" in num:
                        num=num.split(":")[0]
                    while len(num)>0 and num[0]=='0':
                        num=num[1:]
                    kind.append(int(num))
    #here you get ans,pos and kind
    print(len(ans),len(kind),len(tigger_pos))

    #if event has some break,then break it

    from constant import SenLen
    Rwordemb=[]
    Uwordemb=[]
    RinMask=[]
    UinMask=[]
    Rlable=[]
    Ulable=[]
    RlMask=[]
    RrMask=[]
    UlMask=[]
    UrMask=[]
    cnt=0
    for sentence,kd,tpos in zip(ans,kind,tigger_pos):
        senemb=[]
        for item in sentence:
            if item not in embedd_index.keys():
                continue
            mark = int(embedd_index[item])
            senemb.append(mark)

        mylen=len(senemb)
        if len(senemb)>SenLen:
            continue

        while len(senemb)<SenLen:
            senemb.append(0)

        #try rand

        if tpos%2==0 or tpos==-1:
            Uwordemb.append(senemb)
            Ulable.append(0)
            UinMask.append([int(i<mylen) for i in range(0,SenLen)])
            UlMask.append([int(i<tpos) for i in range(0,SenLen)])
            UrMask.append([int(i>tpos) for i in range(0,SenLen)])
        else:
            Rwordemb.append(senemb)
            Rlable.append(kd)
            RinMask.append([int(i<mylen) for i in range(0, SenLen)])
            RlMask.append([int(i<tpos) for i in range(0, SenLen)])
            RrMask.append([int(i>tpos) for i in range(0, SenLen)])
        cnt+=1

    import numpy as np
    np.save("R_wordEmb.npy",np.array(Rwordemb))
    np.save("R_inMask.npy",np.array(RinMask))
    np.save("R_label.npy",np.array(Rlable))
    np.save("R_maskL.npy",np.array(RlMask))
    np.save("R_maskR.npy",np.array(RrMask))

    np.save("U_inMask.npy",np.array(UinMask))
    np.save("U_label.npy",np.array(Ulable))
    np.save("U_maskL.npy",np.array(UlMask))
    np.save("U_wordEmb.npy",np.array(Uwordemb))
    np.save("U_maskR.npy",np.array(UrMask))




if __name__ == '__main__':
    #import torch
    #print(torch.cuda.is_available())#我这里不知道为什么，但是这段代码是必要的

    #first is content
    '''
    we have five thing to do
    self.words=np.load(Tag+"_wordEmb.npy")
    self.pos=np.load(Tag+"_inMask.npy")
    self.label=np.load(Tag+"_label.npy")
    self.maskL=np.load(Tag+"_maskL.npy")
    self.maskR=np.load(Tag+"_maskR.npy")
    
    tag="R" and "U"
    '''
    #do the wordEmb first
    prepare_data()

    import  numpy as np
    from train import rt_ans
    rt_ans()




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
