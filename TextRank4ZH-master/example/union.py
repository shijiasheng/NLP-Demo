import os
from textrank4zh import TextRank4Keyword

num1 = []
num2 = []
file1 = open('F:\\语料\\字典结果.txt', 'rb')
file2 = open('F:\\语料\\模型结果.txt', 'rb')
numbers1 = file1.readlines()
numbers2 = file2.readlines()
numbersall = []
count = 0
for item1 in numbers1:
    for item2 in numbers2:
        if item1 == item2:
            count = count + 1
            numbersall.append(item1.decode("utf-8"))
print(numbersall)
print(count)