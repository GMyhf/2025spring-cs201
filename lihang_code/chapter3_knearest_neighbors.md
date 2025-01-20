# 第3章 K近邻

Updated 1123 GMT+8 Jan 20 2025.
2025 winter, Complied by Hongfei Yan



Logs:

> 我们现有的知识，也可以去修正网上流行的代码了。不用那么复杂，用heapq保证knn_list中始终有n个最小距离就可以的。
>







## 3.1 模型

**模型定义**





# k_neighbors_classifier.py

```python
import math
from itertools import combinations


def L(x, y, p=2):
    # x1 = [1, 1], x2 = [5,1]
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i] - y[i]), p)
        return math.pow(sum, 1 / p)
    else:
        return 0


# x1 = [1, 1]
# x2 = [5, 1]
# x3 = [4, 4]
# # x1, x2
# for i in range(1, 5):
#     r = {'1-{}'.format(c): L(x1, c, p=i) for c in [x2, x3]}
#     print(min(zip(r.values(), r.keys())))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
import heapq

# data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

#print(df)

# plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
# plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend()
# plt.show()

data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:, :-1], data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


class KNN:
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        """
        parameter: n_neighbors 临近点个数
        parameter: p 距离度量
        """
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    #     def predict(self, X):
    #         # 取出n个点
    #         knn_list = []
    #         for i in range(self.n):
    #             dist = np.linalg.norm(X - self.X_train[i], ord=self.p)  # 欧氏距离
    #             knn_list.append((dist, self.y_train[i]))
    #
    #         for i in range(self.n, len(self.X_train)):
    #             max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
    #             dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
    #             if knn_list[max_index][0] > dist:
    #                 knn_list[max_index] = (dist, self.y_train[i])
    #
    #         # 统计
    #         knn = [k[-1] for k in knn_list]
    #         count_pairs = Counter(knn)
    # #         max_count = sorted(count_pairs, key=lambda x: x)[-1]
    #         max_count = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]
    #         return max_count

    # 可以用heapg保证knn list中始终有n个最小距离。
    def predict(self, X):
        # 初始化最大堆（注意：由于Python只有最小堆，这里我们将距离取负）
        knn_heap = []

        for i in range(len(self.X_train)):
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            if len(knn_heap) < self.n:
                # 如果堆还没有满，直接添加
                heapq.heappush(knn_heap, (-dist, self.y_train[i]))
            else:
                # 如果新点的距离更近，则替换堆顶元素
                if -knn_heap[0][0] > dist:
                    heapq.heapreplace(knn_heap, (-dist, self.y_train[i]))

        # 统计
        knn = [k[1] for k in knn_heap]
        count_pairs = Counter(knn)
        max_count = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]

        return max_count

    def score(self, X_test, y_test):
        right_count = 0
        n = 10
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)


clf = KNN(X_train, y_train)
clf.score(X_test, y_test)

test_point = [6.0, 3.0]
print('Test Point: {}'.format(clf.predict(test_point)))
# Test Point: 1.0

plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.plot(test_point[0], test_point[1], 'bo', label='test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()


from sklearn.neighbors import KNeighborsClassifier
clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)
print('sklearn model score: {}'.format(clf_sk.score(X_test, y_test)))
# sklearn model score: 1.0

```



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202501201131083.png" alt="image-20250120113125707" style="zoom: 67%;" />







# kd树

### sklearn.neighbors.KNeighborsClassifier

- n_neighbors: 临近点个数

- p: 距离度量

- algorithm: 近邻算法，可选{'auto', 'ball_tree', 'kd_tree', 'brute'}

- weights: 确定近邻的权重

  

**kd**树是一种对k维空间中的实例点进行存储以便对其进行快速检索的树形数据结构。

**kd**树是二叉树，表示对k维空间的一个划分（partition）。构造**kd**树相当于不断地用垂直于坐标轴的超平面将k维空间切分，构成一系列的k维超矩形区域。kd树的每个结点对应于一个k维超矩形区域。

> 【2.2 kd树是如何构造的？-哔哩哔哩】 https://b23.tv/KdKiJUb
>
> 【2.3 kd树的搜索过程-哔哩哔哩】 https://b23.tv/xzctdn1

构造**kd**树的方法如下：

构造根结点，使根结点对应于k维空间中包含所有实例点的超矩形区域；通过下面的递归方法，不断地对k维空间进行切分，生成子结点。在超矩形区域（结点）上选择一个坐标轴和在此坐标轴上的一个切分点，确定一个超平面，这个超平面通过选定的切分点并垂直于选定的坐标轴，将当前超矩形区域切分为左右两个子区域 （子结点）；这时，实例被分到两个子区域。这个过程直到子区域内没有实例时终止（终止时的结点为叶结点）。在此过程中，将实例保存在相应的结点上。

通常，依次选择坐标轴对空间切分，选择训练实例点在选定坐标轴上的中位数 （median）为切分点，这样得到的**kd**树是平衡的。注意，平衡的**kd**树搜索时的效率未必是最优的。

## 构造平衡kd树算法

输入：k维空间数据集$T＝\{x1，x2,…,xN\}$，

其中$x_{i}=\left(x_{i}^{(1)}, x_{i}^{(2)}, \cdots, x_{i}^{(k)}\right)^{\mathrm{T}}$

输出：**kd**树。

（1）开始：构造根结点，根结点对应于包含T的k维空间的超矩形区域。

选择 $x^{(1)}$ 为坐标轴，以T中所有实例的 $x^{(1)}$ 坐标的中位数为切分点，将根结点对应的超矩形区域切分为两个子区域。切分由通过切分点并与坐标轴 $x^{(1)}$ 垂直的超平面实现。

由根结点生成深度为1的左、右子结点：左子结点对应坐标 $x^{(1)}$ 小于切分点的子区域， 右子结点对应于坐标 $x^{(1)}$ 大于切分点的子区域。

将落在切分超平面上的实例点保存在根结点。

（2）重复：对深度为j的结点，选择 $x^{(1)}$ 为切分的坐标轴，$l＝j(modk)+1$，以该结点的区域中所有实例的$x^{(1)}$坐标的中位数为切分点，将该结点对应的超矩形区域切分为两个子区域。切分由通过切分点并与坐标轴x(1)垂直的超平面实现。

由该结点生成深度为 $j+1$ 的左、右子结点：左子结点对应坐标 $x^{(1)}$ 小于切分点的子区域，右子结点对应坐标 $x^{(1)}$ 大于切分点的子区域。

将落在切分超平面上的实例点保存在该结点。

（3）直到两个子区域没有实例存在时停止。从而形成**kd**树的区域划分。



## kd_tree_demo.py

```python
#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: kd_tree_demo.py
@time: 2021/8/3 17:08
@project: statistical-learning-method-solutions-manual
@desc: 习题3.2 kd树的构建与求最近邻点
"""

import numpy as np
from sklearn.neighbors import KDTree

# 构造例题3.2的数据集
train_data = np.array([[2, 3],
                       [5, 4],
                       [9, 6],
                       [4, 7],
                       [8, 1],
                       [7, 2]])
# （1）使用sklearn的KDTree类，构建平衡kd树
# 设置leaf_size为2，表示平衡树
tree = KDTree(train_data, leaf_size=2)

# （2）使用tree.query方法，设置k=1，查找(3, 4.5)的最近邻点
# dist表示与最近邻点的距离，ind表示最近邻点在train_data的位置
dist, ind = tree.query(np.array([[3, 4.5]]), k=1)
node_index = ind[0]

# 得到最近邻点
x1 = train_data[node_index][0][0]
x2 = train_data[node_index][0][1]
print("x点的最近邻点是({0}, {1})".format(x1, x2))
# 输出结果为：x点的最近邻点是(2, 3)
```

