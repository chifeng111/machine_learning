# coding: utf-8
from sklearn import svm
from matplotlib import pyplot
from numpy import array, linspace

def createData():
    data = array([[2,0],[1,1],[2,3]])
    label = [0,0,1]
    return data, label

if __name__ == '__main__':
    data, label = createData()
    clf = svm.SVC(kernel='linear')
    clf.fit(data, label)
    print(clf)
    print(clf.support_vectors_) #打印支持向量
#     print(clf.predict([-0.8,-1]))   #进行预测
    
    #获得超平面的参数w,b
    w = clf.coef_[0]
    b = clf.intercept_[0]
    print(w, b)
    pyplot.figure()
    pyplot.scatter(data[:,0], data[:,1], c=label)
    #画出超平面
    xx = linspace(0,3,100)
    yy = xx*(-w[0]/w[1])-(b/w[1])
    pyplot.plot(xx,yy,'-r')
    #标出支持向量
#     pyplot.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=80,)
    pyplot.show()
    
    
    
    
    
    
    
    
    
    