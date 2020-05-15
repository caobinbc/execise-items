# coding: utf-8

from math import log
import operator

class DecisionTreeModel():

    # 计算给定数据集的香农熵
    def calcShannonEnt(self, dataSet):
        numEntries = len(dataSet)
        labelCounts = {}
        # 对所有可能分类创建字典
        for featVec in dataSet:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key])/numEntries
            shannonEnt -= prob * log(prob, 2)  # 以2为底求对数
        return shannonEnt

    # 按照给定特征划分数据集
    def splitDataSet(self, dataSet, axis, value):
        retDataset = []
        for featVec in dataSet:
            if featVec[axis] == value:
                reduceFeatVec = featVec[:axis]
                reduceFeatVec.extend(featVec[axis+1:1])
                retDataset.append(reduceFeatVec)
        return retDataset

    def chooseBestFeatureToSplit(self, dataSet):
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = self.calcShannonEnt(dataSet)
        bestInfoGain = 0.0
        bestFeature = -1
        for i in range(numFeatures):
            # 创建唯一的分类标签列表
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntropy = 0.0
            for value in uniqueVals: # 计算每种划分方式的信息熵
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * self.calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
            if (infoGain > bestInfoGain):    # 计算最好的信息增益
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature

    def majorityCnt(self, classList):
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys(): classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def createTree(self, dataSet, labels):
        classList = [example[-1] for example in dataSet]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        # 遍历完所有特征时返回出现次数最多的类别
        if len(dataSet[0]) == 1:
            return self.majorityCnt(classList)
        bestFeat = self.chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}
        # 得到列表中所有的属性值
        del (labels[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value), subLabels)
        return myTree

    def classify(self, inputTree, featLabels, testVec):
        firstStr = inputTree.keys()[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        key = testVec[featIndex]
        valueOfFeat = secondDict[key]
        if isinstance(valueOfFeat, dict):
            classLabel = self.classify(valueOfFeat, featLabels, testVec)
        else:
            classLabel = valueOfFeat
        return classLabel

    def storeTree(self, inputTree, filename):
        import pickle
        fw = open(filename, 'w')
        pickle.dump(inputTree, fw)
        fw.close()

    def grabTree(self, filename):
        import pickle
        fr = open(filename)
        return pickle.load(fr)

