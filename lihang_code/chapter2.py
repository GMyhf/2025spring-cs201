import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target

df.columns = [
    'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
]
print(df.label.value_counts())
"""
label
0    50
1    50
2    50
Name: count, dtype: int64
"""
# plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
# plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend()
# plt.show()

data = np.array(df.iloc[:100, [0, 1, -1]])  # 取前100行，第0、1、-1列
X, y = data[:, :2], data[:, -1]  # X为前两列，y为最后一列
y = np.array([1 if i == 1 else -1 for i in y])  # 将标签转换为1和-1


# 数据线性可分，二分类数据
# 此处为一元一次线性方程
class Model:
    def __init__(self):
        self.w = np.ones(len(data[0]) - 1, dtype=np.float32)
        self.b = 0
        self.l_rate = 0.1

    def sign(self, x, w, b):
        y = np.dot(x, w) + b    # 点积
        return y

    # 随机梯度下降法（SGD）
    def fit(self, X_train, y_train):
        not_converged = True
        while not_converged:
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self.w, self.b) <= 0:
                    self.w = self.w + self.l_rate * np.dot(y, X)
                    self.b = self.b + self.l_rate * y
                    wrong_count += 1
            if wrong_count == 0:
                not_converged = False
        return 'Perceptron Model!'

    def score(self):
        pass


perceptron = Model()
perceptron.fit(X, y)

x_points = np.linspace(4, 7, 10)    # 生成4到7之间的10个数
y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]   # 二维坐标系中的直线方程
plt.plot(x_points, y_)

plt.plot(data[:50, 0], data[:50, 1], 'o', color='blue', label='0')
plt.plot(data[50:100, 0], data[50:100, 1], 'o', color='orange', label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()
