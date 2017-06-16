# coding: utf-8
from numpy import array, mat, linalg, linspace, dot, tile
from matplotlib import pyplot

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) - 1 #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return array(dataMat), array(labelMat)

def standRegres(xArr,yArr): #计算回归参数
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:  #计算xTx的行列式的值
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws

if __name__ == '__main__':
    data, label = loadDataSet('ex0.txt')
    ws = standRegres(data, label)
    print(ws)
    #作图显示训练集数据
    pyplot.figure()
    for i in range(len(label)):
        pyplot.scatter(data[i,1], label[i], c = 'black')
    
    #作图画出回归线
    w = array(ws)
    xx = linspace(0, 1)
    yy = w[0] + w[1]*xx
    print(xx,yy)
    pyplot.plot(xx, yy, "-r")
    pyplot.show()
    
    
    