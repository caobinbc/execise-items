# coding: utf-8

import os
import sys
import pickle
import pandas as pd

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


if __name__ == '__main__':

    filename = "E:\\execise-items\\breast-canser\\data\\breast-cancer-wisconsin.data"


    dataSet, featnames = filetoDataSet(filename)
    Tree = createDecisionTree(dataSet[:620], featnames)
    storetree = "E:\\execise-items\\breast-canser\\data\\decTree.dect"
    StoreTree(Tree, storetree)
    # Tree = ReadTree(storetree)
    i = 1
    cnt = 0
    for lis in dataSet[620:]:
        judge = classify(Tree, featnames, lis[:-1])
        shouldbe = lis[-1]
        if judge == shouldbe:
            cnt += 1
        print("Test %d was classified %s, it's class is %s %s" % (
        i, judge, shouldbe, "=====" if judge == shouldbe else ""))
        i += 1
    print("The Tree's Accuracy is %.3f" % (cnt / float(i)))