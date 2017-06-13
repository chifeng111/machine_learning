#coding:utf-8
from numpy import array, tile
from matplotlib import pyplot
from matplotlib.pyplot import text

# 创建训练样本集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  #获得训练样本的个数，及训练样本矩阵的行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5    #到此，距离已经计算出来了
    sortedDistIndicies = distances.argsort()    #返回距离从小到大排序的索引
    classCount = {} #用字典记录最近的k个距离的标签出现的次数
    for i in range(k):
        if labels[sortedDistIndicies[i]] not in classCount:
            classCount[labels[sortedDistIndicies[i]]] = 1
        else:
            classCount[labels[sortedDistIndicies[i]]] += 1
    sortedClassCount = sorted(classCount.items(), key= lambda item:item[1], reverse=True)
    return sortedClassCount[0][0]
        

if __name__ == '__main__':
    group, labels = createDataSet()
    print(classify0([0.4,0.5], group, labels, 3))
    #做出训练集图像
    pyplot.figure()
    pyplot.scatter(group[:,0], group[:,1])
    for i in range(len(group)):
        text(group[i][0]+0.02, group[i][1], labels[i])
    pyplot.scatter(0.4, 0.5, c='r')
    pyplot.show()

