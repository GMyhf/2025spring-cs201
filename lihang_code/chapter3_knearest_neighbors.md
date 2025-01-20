# 第3章 K近邻

Updated 1123 GMT+8 Jan 20 2025.
2025 winter, Complied by Hongfei Yan



Logs:

> 我们现有的知识，也可以去修正网上流行的代码了。不用那么复杂，用heapq保证knn_list中始终有n个最小距离就可以的。
>







## 3.1 模型

**模型定义**







# chapter3.py代码

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





```python

```

