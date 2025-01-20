# 第3章 $k$近邻

Updated 1123 GMT+8 Jan 20 2025.
2025 winter, Complied by Hongfei Yan



Logs:

> 我们现有的知识，也可以去修正网上流行的代码了。不用那么复杂，用heapq保证knn_list中始终有n个最小距离就可以的。
>



$k$-近邻法（$k$-nearest neighbor, $k$-NN）是一种基本的分类与回归方法。$k$近邻法是一种基于实例的学习方法，通过查找训练数据集中与新实例最接近的 $k$ 个实例来预测新实例的类别。算法的核心在于选择合适的 $k$ 值、距离度量和分类决策规则。



## 3.1 $k$近邻算法

$k$ 近邻算法简单、直观：给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的 $k$ 个实例，这 $k$ 个实例的多数属于某个类，就把该输入实例分为这个类。

**算法3.1** （$k$近邻法）

输入：训练数据集 $ T = \{(x_1, y_1), (x_2, y_2), \cdots, (x_N, y_N)\} $

其中，$ x_i \in \mathcal{X} \subseteq \mathbb{R}^n $ 为实例的特征向量，$ y_i \in \mathcal{Y} = \{c_1, c_2, \cdots, c_K\} $ 为实例的类别，$ i = 1, 2, \cdots, N $。

输出：实例 $ x $ 所属的类 $ y $。

（1）根据给定的距离度量，在训练集 $ T $ 中找出与 $ x $ 最邻近的 $ k $ 个点，涵盖这 $ k $ 个点的 $ x $ 的邻域记作 $ N_k(x) $。

（2）在 $ N_k(x) $ 中根据分类决策规则（如多数表决）决定 $ x $ 的类别 $ y $：
$
y = \arg \max_{c_j} \sum_{x_i \in N_k(x)} I(y_i = c_j), \quad i = 1, 2, \cdots, N, j = 1, 2, \cdots, K
$
式 (3.1) 中，$ I $ 为指示函数，即当 $ y_i = c_j $ 时 $ I $ 为 1，否则 $ I $ 为 0。

$k$近邻法的特殊情况是 $k = 1$ 的情形，称为最近邻算法。对于输入的实例点（特征向量）$ x $，最近邻法将训练数据集中与 $ x $ 最邻近点的类作为 $ x $ 的类。

$k$紧邻法没有显示的学习过程。





## 3.2 $k$近邻模型

### 3.2.1 模型

### 3.2.2 距离度量

欧氏距离（Euclidean distance），曼哈顿距离（Manhattan distance）。



下面例子说明，由不同距离度量下最近邻点是不同的。

**例 3.1** 已知二维空间的 3 个点 $ x_1 = (1, 1)^T $，$ x_2 = (5, 1)^T $，$ x_3 = (4, 4)^T $，试求在 $ p $ 取不同值时，$ L_p $ 距离下 $ x_1 $ 的最近邻点。

**解**

因为 $ x_1 $ 和 $ x_2 $ 只有第一维的值不同，所以 $ p $ 为任何值时，$ L_p(x_1, x_2) = 4 $。

而 $ x_1 $ 和 $ x_3 $ 在不同 $ p $ 值下的 $ L_p $ 距离：

$ L_1(x_1, x_3) = 6 $，$ L_2(x_1, x_3) = 4.24 $，$ L_3(x_1, x_3) = 3.78 $，$ L_4(x_1, x_3) = 3.57 $

于是得到：$ p$  等于 1  或 2 时，$ x_2 $ 是 $ x_1 $ 的最近邻点；当 $ p \geqslant 3 $ 时，$ x_3 $ 是 $ x_1 $ 的最近邻点。



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


x1 = [1, 1]
x2 = [5, 1]
x3 = [4, 4]
# x1, x2
for i in range(1, 5):
    r = {'1-{}'.format(c): L(x1, c, p=i) for c in [x2, x3]}
    print(min(zip(r.values(), r.keys())))

"""
(4.0, '1-[5, 1]')
(4.0, '1-[5, 1]')
(3.7797631496846193, '1-[4, 4]')
(3.5676213450081633, '1-[4, 4]')
"""
```



展示如何手动实现一个简单的 KNN 分类器，并将其应用于 Iris 数据集的一个子集上。此外，它还演示了如何使用 Python 的标准库和第三方库来进行数据处理、可视化以及机器学习模型的构建与评估。同时，通过与 scikit-learn 提供的专业实现进行对比，可以验证自定义模型的有效性或发现可能存在的优化空间。

k_neighbors_classifier.py

```python
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



> 加载数据到构建和评估一个自定义的 $k $近邻分类器的过程，并且与 `scikit-learn` 库中的 KNN 分类器进行了比较。以下是该段代码的主要功能解析：
>
> 1. **数据准备**
>
> - **导入必要的库**：包括用于数值计算的 NumPy，用于数据处理的 Pandas，用于绘图的 Matplotlib，以及 Scikit-Learn 的一些模块。
> - **加载 Iris 数据集**：使用 `sklearn.datasets.load_iris()` 函数加载著名的鸢尾花数据集。该数据集包含150个样本，每个样本有4个特征（萼片长度、萼片宽度、花瓣长度、花瓣宽度），并分为3个类别。
> - **创建 DataFrame**：将数据集转换为 Pandas 的 DataFrame 格式，方便后续的数据操作。
> - **选择部分数据**：为了简化问题，只选择了前两个类别（即前100个样本），并且只保留了两个特征（萼片长度和萼片宽度）用于可视化和训练模型。
>
> **2. 可视化**
>
> - **绘制散点图**：通过 Matplotlib 绘制了两类样本在二维空间中的分布情况，帮助直观理解数据结构。
>
> **3. 模型实现**
>
> - **定义 KNN 类**：编写了一个名为 `KNN` 的类来实现 \( k \)-近邻算法。这个类包含了初始化方法 (`__init__`)、预测方法 (`predict`) 和评分方法 (`score`)。
>   - 在 `predict` 方法中，使用了最大堆（通过将距离取负数模拟最小堆）来高效地找到最近的 \( k \) 个邻居，并根据多数表决规则确定新实例的类别。
>   - `score` 方法用来评估模型在测试集上的准确率。
>
> **4. 模型训练与评估**
>
> - **实例化 KNN 对象**：用训练数据初始化 KNN 模型。
> - **评估模型性能**：调用 `score` 方法计算模型在测试集上的正确率。
> - **测试单个点**：对给定的一个测试点 `[6.0, 3.0]` 进行分类预测，并打印结果。
> - **可视化测试点**：在之前的散点图基础上添加测试点的位置。
>
> **5. 使用 Scikit-Learn 的 KNN 模型对比**
>
> - **导入并配置 sklearn 的 KNN 分类器**：创建并训练了一个来自 `scikit-learn.neighbors` 模块的标准 KNN 分类器。
> - **评估 sklearn 模型**：输出 sklearn 版本的 KNN 分类器在测试集上的得分。



### 3.2.3 $k$值的选择

$k$值一般取一个比较小的数值。通常采用交叉验证法来选取最优的$k$值。







### 3.3.4 分类决策规则

多数表决规则（majority voting rule）。







## 3.3 $k$近邻法的实现：$kd$树



**sklearn.neighbors.KNeighborsClassifier**

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



### 3.3.1 构造平衡kd树算法

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



### 3.3.2 搜索$kd$树

**最近邻搜索**

`find_nearest` 函数：

- 递归搜索 k-d 树，寻找与目标点最近的样本点。
- 核心逻辑：
  1. **递归到叶节点**：确定目标点所在的子空间。
  2. **更新最近邻信息**：从叶节点向上回溯，更新最近邻点和距离。
  3. 剪枝优化：判断超球体（目标点为球心，当前最近距离为半径）是否与分割超平面相交。
     - 如果不相交，则无需访问另一子空间。
  4. **检查另一子空间**：如果超球体与分割超平面相交，递归检查另一子空间，更新最近邻信息。



### sklearn_kd_tree_demo.py

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



### my_kd_tree.py

```python
from math import sqrt
from collections import namedtuple
import time
from random import random

# 定义一个namedtuple,分别存放最近坐标点、最近距离和访问过的节点数
result = namedtuple("Result_tuple",
                    "nearest_point  nearest_dist  nodes_visited")


# kd-tree每个结点中主要包含的数据结构如下
class KdNode:
    def __init__(self, dom_elt, split, left, right):
        self.dom_elt = dom_elt  # k维向量节点(k维空间中的一个样本点)
        self.split = split  # 整数（进行分割维度的序号）
        self.left = left  # 该结点分割超平面左子空间构成的kd-tree
        self.right = right  # 该结点分割超平面右子空间构成的kd-tree


class KdTree:
    def __init__(self, data):
        k = len(data[0])  # 数据维度

        def create_node(split, data_set):  # 按第split维划分数据集,创建KdNode
            if not data_set:  # 数据集为空
                return None

            data_set.sort(key=lambda x: x[split])  # 按要进行分割的那一维数据排序
            split_pos = len(data_set) // 2  # 整数除法得到中间位置
            median = data_set[split_pos]  # 中位数分割点
            split_next = (split + 1) % k  # cycle coordinates

            # 递归的创建kd树
            return KdNode(
                median,
                split,
                create_node(split_next, data_set[:split_pos]),  # 创建左子树
                create_node(split_next, data_set[split_pos + 1:]))  # 创建右子树

        self.root = create_node(0, data)  # 从第0维分量开始构建kd树,返回根节点


# KDTree的前序遍历
def preorder(root):
    print(root.dom_elt)
    if root.left:  # 节点不为空
        preorder(root.left)
    if root.right:
        preorder(root.right)


# 对构建好的kd树进行搜索，寻找与目标点最近的样本点：


def find_nearest(tree, point):
    k = len(point)  # 数据维度

    def travel(kd_node, target, max_dist):
        if kd_node is None:
            return result([0] * k, float("inf"), 0)

        nodes_visited = 1

        s = kd_node.split  # 进行分割的维度
        pivot = kd_node.dom_elt  # 进行分割的“轴”

        if target[s] <= pivot[s]:  # 如果目标点第s维小于分割轴的对应值(目标离左子树更近)
            nearer_node = kd_node.left  # 下一个访问节点为左子树根节点
            further_node = kd_node.right  # 同时记录下右子树
        else:  # 目标离右子树更近
            nearer_node = kd_node.right  # 下一个访问节点为右子树根节点
            further_node = kd_node.left

        temp1 = travel(nearer_node, target, max_dist)  # 进行遍历找到包含目标点的区域

        nearest = temp1.nearest_point  # 以此叶结点作为“当前最近点”
        dist = temp1.nearest_dist  # 更新最近距离

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist  # 最近点将在以目标点为球心，max_dist为半径的超球体内

        temp_dist = abs(pivot[s] - target[s])  # 第s维上目标点与分割超平面的距离
        if max_dist < temp_dist:  # 判断超球体是否与超平面相交
            return result(nearest, dist, nodes_visited)  # 不相交则可以直接返回，不用继续判断

        # ----------------------------------------------------------------------
        # 计算目标点与分割点的欧氏距离
        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))

        if temp_dist < dist:  # 如果“更近”
            nearest = pivot  # 更新最近点
            dist = temp_dist  # 更新最近距离
            max_dist = dist  # 更新超球体半径

        # 检查另一个子结点对应的区域是否有更近的点
        temp2 = travel(further_node, target, max_dist)

        nodes_visited += temp2.nodes_visited
        if temp2.nearest_dist < dist:  # 如果另一个子结点内存在更近距离
            nearest = temp2.nearest_point  # 更新最近点
            dist = temp2.nearest_dist  # 更新最近距离

        return result(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))  # 从根节点开始递归


data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
kd = KdTree(data)
preorder(kd.root)
"""
[7, 2]
[5, 4]
[2, 3]
[4, 7]
[9, 6]
[8, 1]
"""

ret = find_nearest(kd, [3, 4.5])
print(ret)
# Result_tuple(nearest_point=[2, 3], nearest_dist=1.8027756377319946, nodes_visited=4)

# 产生一个k维随机向量，每维分量值在0~1之间
def random_point(k):
    return [random() for _ in range(k)]


# 产生n个k维随机向量
def random_points(k, n):
    return [random_point(k) for _ in range(n)]


N = 400000
# 在开始时记录进程时间
start_cpu_time = time.process_time()

kd2 = KdTree(random_points(3, N))  # 构建包含四十万个3维空间样本点的kd树
ret2 = find_nearest(kd2, [0.1, 0.5, 0.8])  # 四十万个样本点中寻找离目标最近的点

# 在结束时再次记录进程时间
end_cpu_time = time.process_time()

# 计算并打印所用的CPU时间
elapsed_cpu_time = end_cpu_time - start_cpu_time
print(f"Elapsed CPU time: {elapsed_cpu_time:0.4f} seconds")

print(ret2)
# Elapsed CPU time: 3.9399 seconds
# Result_tuple(nearest_point=[0.09951475212182137, 0.4971758210372218, 0.8019299872473542], nearest_dist=0.0034548955254863362, nodes_visited=46)

```

