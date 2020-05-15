# coding: utf-8

import os
import sys
import pickle
import pandas as pd
from sklearn import tree as stree
from sklearn.model_selection import cross_val_score



sys.path.append('.')

from tree import filetoDataSet, createDecisionTree


def classify(Tree, featnames, X):
    classLabel = "未知"
    root = list(Tree.keys())[0]
    firstGen = Tree[root]
    featindex = featnames.index(root)  #根节点的属性下标
    for key in firstGen.keys():   #根属性的取值,取哪个就走往哪颗子树
        if X[featindex] == key:
            if type(firstGen[key]) == type({}):
                classLabel = classify(firstGen[key],featnames,X)
            else:
                classLabel = firstGen[key]
    return classLabel


def StoreTree(Tree,filename):
    fw = open(filename,'wb')
    pickle.dump(Tree,fw)
    fw.close()

# 5折交叉验证
def cross_val_score_n(dataSet, featnames):

    scores= []
    # 将数据分为5份
    dataList = []
    pice = len(dataSet) // 5
    for i in range(5):
        n = pice * i
        m = n + pice
        if m >= len(dataSet):
            dataList.append(dataSet[n:])
            print('最后的数据截断是：', n)
        dataList.append(dataSet[n:m])

    # print(pice)
    # for y in range(5):
    #     print(len(dataList[y]))
    for i in range(5):
        cnt = 0
        data = []
        datatest = []
        for t in range(5):
            if t == i:
                datatest = dataList[t]
            data.extend(dataList[t])

        Tree = createDecisionTree(data, featnames)

        for lis in datatest:
            judge = classify(Tree, featnames, lis[:-1])
            shouldbe = lis[-1]
            if judge == shouldbe:
                cnt += 1
            i += 1
        scores.append(cnt / float(i))
    return scores


if __name__ == '__main__':

    filename = "E:\\execise-items\\breast-canser\\data\\breast-cancer-wisconsin.data"

    dataSet = filetoDataSet(filename)
    featnames = [
        'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
        'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'
    ]

    print(featnames)
    scores = cross_val_score_n(dataSet, featnames)
    print(scores)

    clf = stree.DecisionTreeClassifier()

    # X = [d[:-1] for d in dataSet]
    X = []
    for d in dataSet:
        X_covert = []
        try:
            for i in range(len(d)):
                if d[i] == '?':
                    X_covert.append('0')
                    continue
                X_covert.append(d[i])

            X.append(X_covert)
        except Exception as e:
            print(e)
            break
    Y = [d[-1] for d in dataSet]

    clf.fit(X, Y)
    scores = cross_val_score(clf, X, Y, cv=5)
    print(scores)




