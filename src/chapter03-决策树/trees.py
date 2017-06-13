#coding:utf-8
from math import log

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):    #函数计算数据集的熵，传入的是list
    numEntries = len(dataSet)   #获取数据的个数
    labelCount = {}     #得到所有分类加入labelCount，及yes和no
    for i in dataSet:
        currentLabel = i[-1]    #取每个数据的所属分类
        if currentLabel not in labelCount:
            labelCount[currentLabel] = 1
        else:
            labelCount[currentLabel] += 1
    shannonEnt = 0.     #初始熵为0
    for key in labelCount:
        #得到每个分类的概率
        prob = float(labelCount[key])/numEntries
        shannonEnt -= prob * log(prob,2)    #求熵
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for i in dataSet:
        if i[axis] == value:
            tem = i[:axis] #取axis之前的数据
            tem.extend(i[axis+1:]) #取axis之后的数据
            retDataSet.append(tem)
    return retDataSet

if __name__ == '__main__':
    dataSet, labels = createDataSet()
#     print(dataSet)
#     print(labels)
#     print(calcShannonEnt(dataSet))
    print(splitDataSet(dataSet, 0, 0))
