# coding: utf-8
from numpy import array

def loadDataSet():
    dataSet = []
    label = []
    with open("testSet.txt", 'r') as f:
        for line in f.readlines():
            lineList = line.strip().split("\t")
            dataSet.append([float(lineList[0]), float(lineList[1])])
            label.append(int(lineList[2]))
    return array(dataSet), label


if __name__ == '__main__':
    dataSet, label = loadDataSet()   
    print(dataSet)   
    print(label)      