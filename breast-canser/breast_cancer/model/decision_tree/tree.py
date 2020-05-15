from math import log
from operator import itemgetter


def filetoDataSet(filename):
    fr = open(filename,'r')
    all_lines = fr.readlines()
    featname = all_lines[0].strip().split(',')[1:-1]
    print(featname)
    dataSet = []
    for line in all_lines[1:]:
        line = line.strip()
        lis = line.split(',')[1:]
        dataSet.append(lis)
    fr.close()
    return dataSet,featname

def calcEnt(dataSet):           #计算香农熵
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        label = featVec[-1]
        if label not in labelCounts.keys():
            labelCounts[label] = 0
        labelCounts[label] += 1
    Ent = 0.0
    for key in labelCounts.keys():
        p_i = float(labelCounts[key]/numEntries)
        Ent -= p_i * log(p_i,2)
    return Ent
def splitDataSet(dataSet, axis, value):   #划分数据集,找出第axis个属性为value的数据
    returnSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            retVec = featVec[:axis]
            retVec.extend(featVec[axis+1:])
            returnSet.append(retVec)
    return returnSet
def chooseBestFeat(dataSet):
    numFeat = len(dataSet[0])-1
    Entropy = calcEnt(dataSet)
    DataSetlen = float(len(dataSet))
    bestGain = 0.0
    bestFeat = -1
    for i in range(numFeat):
        allvalue = [featVec[i] for featVec in dataSet]
        specvalue = set(allvalue)
        nowEntropy = 0.0
        for v in specvalue:
            Dv = splitDataSet(dataSet,i,v)
            p = len(Dv)/DataSetlen
            nowEntropy += p * calcEnt(Dv)
        if Entropy - nowEntropy > bestGain:
            bestGain = Entropy - nowEntropy
            bestFeat = i
    return bestFeat
def Vote(classList):
    classdic = {}
    for vote in classList:
        if vote not in classdic.keys():
            classdic[vote] = 0
        classdic[vote] += 1
    sortedclassDic = sorted(classdic.items(),key=itemgetter(1),reverse=True)
    return sortedclassDic[0][0]
def createDecisionTree(dataSet,featnames):
    featname = featnames[:]              ################
    classlist = [featvec[-1] for featvec in dataSet]  #此节点的分类情况
    if classlist.count(classlist[0]) == len(classlist):  #全部属于一类
        return classlist[0]
    if len(dataSet[0]) == 1:         #分完了,没有属性了
        return Vote(classlist)       #少数服从多数
    # 选择一个最优特征进行划分
    bestFeat = chooseBestFeat(dataSet)
    bestFeatname = featname[bestFeat]
    del(featname[bestFeat])     #防止下标不准
    DecisionTree = {bestFeatname:{}}
    # 创建分支,先找出所有属性值,即分支数
    allvalue = [vec[bestFeat] for vec in dataSet]
    specvalue = sorted(list(set(allvalue)))  #使有一定顺序
    for v in specvalue:
        copyfeatname = featname[:]
        DecisionTree[bestFeatname][v] = createDecisionTree(splitDataSet(dataSet,bestFeat,v),copyfeatname)
    return DecisionTree

if __name__ == '__main__':
    filename = "E:\\execise-items\\breast-canser\\data\\breast-cancer-wisconsin.data"
    DataSet,featname = filetoDataSet(filename)
    #print(DataSet)
    #print(featname)
    Tree = createDecisionTree(DataSet,featname)
    print(Tree)