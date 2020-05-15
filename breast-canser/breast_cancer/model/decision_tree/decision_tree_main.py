# coding: utf-8

import os
import sys
import pandas as pd

sys.path.append('.')

from tree import DecisionTreeModel

# 处理数据
data = pd.read_csv('E:\\execise-items\\breast-canser\\data\\breast-cancer-wisconsin.data', header=None)
print(data)