#coding:utf-8
from numpy import tile, array


#从文件导入数据
def file2Matrix(fileName):
    fr = open(fileName, 'r')
    returnMat = []
    classLableVector = []
    for line in fr:
        line = line.strip()
        listFromLine = line.split('\t')
        listFromLine = [float(i) for i in listFromLine]
        returnMat.append(listFromLine[:3])
        classLableVector.append(int(listFromLine[-1]))
    returnMat = array(returnMat)
    return returnMat, classLableVector

def autoNorm(dataSet):
    attr = dataSet.shape[1]  #获取训练集的属性个数，及矩阵列数
    for i in range(attr):   
        dataSet[:,i] = (dataSet[:,i] - min(dataSet[:,i]))/(max(dataSet[:,i])-min(dataSet[:,i]))
    return dataSet

#K进邻算法分类器

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  #获得训练样本的个数，及训练样本矩阵的行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5   #到此，距离已经计算出来了
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
    #训练集数据
    dataSet, labels = file2Matrix('datingTestSet2.txt')
    dataSet = autoNorm(dataSet)
    #测试集数据
    testDataSet, testLabels = file2Matrix('datingTestSet3.txt')
    testDataSet = autoNorm(testDataSet)
#     print(classify0(testDataSet[10], dataSet, labels, 3))
    totalTestNum = testDataSet.shape[0]
    wrongNum = 0
    for i in range(totalTestNum):
        print("the classifier came back with: %d, the real answer is: %d" \
              %(classify0(testDataSet[i], dataSet, labels, 10), testLabels[i]))
        if classify0(testDataSet[i], dataSet, labels, 10) != testLabels[i]:
            wrongNum += 1
        print("the total error rate is: %f" %(wrongNum/totalTestNum))
    

