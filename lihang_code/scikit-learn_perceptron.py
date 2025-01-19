import sklearn
from sklearn.linear_model import Perceptron

print(sklearn.__version__)

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

data = np.array(df.iloc[:100, [0, 1, -1]])  # 取前100行，第0、1、-1列
X, y = data[:, :2], data[:, -1]  # X为前两列，y为最后一列
y = np.array([1 if i == 1 else -1 for i in y])  # 将标签转换为1和-1


clf = Perceptron(fit_intercept=True,
                 max_iter=1000,
                 shuffle=True)
clf.fit(X, y)

# Weights assigned to the features.
print(clf.coef_)

# 画布大小
plt.figure(figsize=(10,10))

# 中文标题
#plt.rcParams['font.sans-serif']=['SimHei']
#plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 使用系统自带的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.title('鸢尾花线性数据示例')

plt.scatter(data[:50, 0], data[:50, 1], c='b', label='Iris-setosa',)
plt.scatter(data[50:100, 0], data[50:100, 1], c='orange', label='Iris-versicolor')

# 画感知机的线
x_ponits = np.arange(4, 8)
y_ = -(clf.coef_[0][0]*x_ponits + clf.intercept_)/clf.coef_[0][1]
plt.plot(x_ponits, y_)

# 其他部分
plt.legend()  # 显示图例
plt.grid(False)  # 不显示网格
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
plt.show()