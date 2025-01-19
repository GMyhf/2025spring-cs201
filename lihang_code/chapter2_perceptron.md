# 第2章 感知机学习算法

Updated 1916 GMT+8 Jan 19 2025.
2025 winter, Complied by Hongfei Yan



Logs:

> 结合AI把程序每一句都看懂。
>



感知机（perceptron）是二分类的线性分类模型。旨在求出将训练数据进行线性划分的分离超平面，为此，导入基于误分类的损失函数，利用梯度下降法对损失函数进行极小化，求得感知机模型。感知机学习算法具有简单而易于实现的特点，分为原始形式和对偶形式。感知机1957年由Rosenblatt提出，是神经网络与支持向量机的基础。



## 2.1 感知机模型

**模型定义**

假设输入空间（特征空间）是 $\mathcal{X} \subseteq \mathbb{R}^n$，输出空间是 $\mathcal{Y} = \{+1, -1\}$。输入 $x \in \mathcal{X}$ 表示实例的特征向量，对应于输入空间（特征空间）的点；输出 $y \in \mathcal{Y}$ 表示实例的类别。由输入空间到输出空间的如下函数：

$ f(x) = \text{sign}(w \cdot x + b) \tag{2.1} $

称为感知机。其中，$w$ 和 $b$ 为感知机模型参数，$w \in \mathbb{R}^n$ 叫作权值（weight）或权值向量（weight vector），$b \in \mathbb{R}$ 叫作偏置（bias），$w \cdot x$ 表示 $w$ 和 $x$ 的内积。sign 是符号函数，即

$ \text{sign}(x) = \begin{cases} 
+1, & x \geqslant 0 \\
-1, & x < 0 
\end{cases} \tag{2.2} $



> 感知机是根据输入实例的特征向量 $x$ 进行二类分类的线性分类模型。



感知机是一种线性分类模型，属于判别模型。感知机模型的假设空间是定义在特征空间中的所有线性分类模型（linear classification model）或线性分类器（linear classifier），即函数集合 $\{f | f(x) = w \cdot x + b\}$。

感知机有如下几何解释：线性方程

$ w \cdot x + b = 0 \tag{2.3} $

对应于特征空间 $\mathbb{R}^n$ 中的一个超平面 $S$，其中 $w$ 是超平面的法向量，$b$ 是超平面的截距。这个超平面将特征空间划分为两个部分。位于两部分的点（特征向量）分别被分为正、负两类。因此，超平面 $S$ 称为分离超平面（separating hyperplane）。



**学习策略** 

感知机学习的目标是最小化损失函数： 

$$ \min_{w, b} L(w, b) = -\sum_{x_i \in M} y_i (w \cdot x_i + b) $$ 

这里的损失函数表示所有误分类点到分离超平面的总距离。 



**算法实现** 

感知机学习算法基于随机梯度下降法对损失函数进行最优化，有原始形式和对偶形式。在原始形式中，算法流程如下： 

1. 任意选取一个初始的超平面。 

2. 随机选择一个误分类点。 

3. 根据以下规则更新权重 $w$ 和偏置 $b$：   

   $w = w + \eta y_i x_i$  

   $b = b + \eta y_i$ 

4. 重复步骤2和3，直到没有误分类点或达到预定的迭代次数。 

当训练数据集线性可分时，感知机学习算法是收敛的。其误分类次数 $k$ 满足不等式： 

$$ k \leqslant \left(\frac{R}{\gamma}\right)^2 $$ 

这里 $R$ 是训练样本集中距离原点最远的点到原点的距离，$\gamma$ 是最小间隔。

当训练数据集线性可分时，感知机学习算法存在无穷多个解，这些解可能因为不同的初值或不同的迭代顺序而有所差异。 



**示例应用** 

以Iris数据集为例，我们可以取出两个分类的数据，并使用`[sepal length, sepal width]`作为特征来进行感知机模型的训练和测试。



## 2.2 感知机学习策略

### 2.2.1 数据集的线性可分性

**定义 2.2（数据集的线性可分性）**给定一个数据集

$ T = \{(x_1, y_1), (x_2, y_2), \cdots, (x_N, y_N)\} $

其中，$ x_i \in \mathcal{X} = \mathbb{R}^n $，$ y_i \in \mathcal{Y} = \{+1, -1\} $，$ i = 1, 2, \cdots, N $，如果存在某个超平面 $ S $

$ w \cdot x + b = 0 $

能够将数据集的正实例点和负实例点完全正确地划分到超平面的两侧

即对所有 $ y_i = +1 $ 的实例 $i $，有 $ w \cdot x_i + b > 0 $，对所有 $ y_i = -1 $ 的实例 $ i $，有 $ w \cdot x_i + b < 0 $，则称数据集 $ T $ 为线性可分数据集（linearly separable data set）；否则，称数据集 $ T $ 线性不可分。



### 2.2.2 感知机学习策略

损失函数是误分类点到超平面 $S$ 的总距离。

给定训练数据集

$ T = \{(x_1, y_1), (x_2, y_2), \cdots, (x_N, y_N)\} $

其中，$ x_i \in \mathcal{X} = \mathbb{R}^n $，$ y_i \in \mathcal{Y} = \{+1, -1\} $，$ i = 1, 2, \cdots, N $。感知机 $ \text{sign}(w \cdot x + b) \tag{2.1} $ 学习的损失函数定义为

$ L(w, b) = - \sum_{x_i \in M} y_i (w \cdot x_i + b) \tag{2.4} $

其中 $ M $ 为误分类点的集合。这个损失函数就是感知机学习的经验风险函数。

显然，损失函数 $ L(w, b) $ 是非负的。如果没有误分类点，损失函数值是 0。而且，误分类点越少，误分类点离超平面越近，损失函数值就越小。一个特定的样本点的损失函数：在误分类时是参数 $ w, b $ 的线性函数，在正确分类时是 0。因此，给定训练数据集 \( T \)，损失函数 $ L(w, b) $ 是 $ w, b $ 的连续可导函数。

感知机学习的策略是在假设空间中选取使损失函数最小的模型参数 $ w, b $，即感知机模型。



## 2.3 感知机学习算法

感知机学习问题转化为求解损失函数的最优化问题，最优化的方法是随机梯度下降法。感知机学习的具体算法，包括原始形式和对偶形式，并证明在训练数据线性可分条件下感知机学习算法的收敛性。

### 2.3.1 感知机学习算法的原始形式

感知机学习算法是误分类驱动的，具体采用随机梯度下降法（stochastic gradient descent）。首先，任意选取一个超平面 $ w_0, b_0 $，然后用梯度下降法不断地极小化目标函数。极小化过程中不是一次使误分类点集合  M  中所有误分类点的梯度下降，而是一次随机选取一个误分类点使其梯度下降。

假设误分类点集合 $ M $ 是固定的，那么损失函数 $ L(w, b) $ 的梯度由

$ \nabla_w L(w, b) = - \sum_{x_i \in M} y_i x_i $

$ \nabla_b L(w, b) = - \sum_{x_i \in M} y_i $

给出。

随机选取一个误分类点 $(x_i, y_i)$，对 $ w, b $ 进行更新：

$ w \leftarrow w + \eta y_i x_i \tag{2.6} $

$ b \leftarrow b + \eta y_i \tag{2.7} $

式中 $\eta$ $(0 < \eta \leq 1)$ 是步长，在统计学习中又称为学习率（learning rate）。这样，通过迭代可以期待损失函数 $ L(w, b) $ 不断减小，直到为 0。综上所述，得到如下算法：

**算法 2.1（感知机学习算法的原始形式）**

输入：训练数据集 $ T = \{(x_1, y_1), (x_2, y_2), \cdots, (x_N, y_N)\} $，其中 $ x_i \in \mathcal{X} = \mathbb{R}^n $，$ y_i \in \mathcal{Y} = \{-1, +1\} $，$ i = 1, 2, \cdots, N $；学习率 $\eta$ $(0 < \eta \leq 1)$；

输出：$ w, b $；感知机模型 $ f(x) = \text{sign}(w \cdot x + b) $。

1. 选取初值 $ w_0, b_0 $；
2. 在训练集中选取数据 $(x_i, y_i)$；
3. 如果 $ y_i (w \cdot x_i + b) \leq 0 $，
   $ w \leftarrow w + \eta y_i x_i $
   $ b \leftarrow b + \eta y_i $
4. 转至 (2)，直至训练集中没有误分类点。



这种学习算法直观上有如下解释：当一个实例点被误分类，即位于分离超平面的错误一侧时，则调整 $ w, b $ 的值，使分离超平面向该误分类点的一侧移动，以减少该误分类点与超平面间的距离，直至超平面越过该误分类点使其被正确分类。

算法 2.1 是感知机学习的基本算法，对应于后面的对偶形式，称为原始形式。感知机学习算法简单且易于实现。



### 2.3.2 算法的收敛性

对于线性可分数据集感知机学习算法原始形式收敛，即经过有限次迭代可以得到一个将训练数据集正确划分的分离超平面及感知机模型。



### 2.3.3 感知机学习算法的对偶形式

感知机学习算法的原始形式和对偶形式与支持向量机学习算法的原始形式和对偶形式相对应。



# 附录 Iris 数据集

Iris 数据集是机器学习和统计学领域中非常著名的一个多变量数据集，由英国统计学家及生物学家罗纳德·费雪（Ronald Fisher）于1936年引入。它也被称为 Fisher's Iris 数据集或 Anderson's Iris 数据集，因为最初的数据是由 Edgar Anderson 收集的。

**Iris 数据集的特点：**

- **样本数量**：共有150个样本。
- **特征数量**：每个样本有四个特征，分别是萼片长度 (sepal length)、萼片宽度 (sepal width)、花瓣长度 (petal length) 和花瓣宽度 (petal width)，单位为厘米。
- **类别数量**：分为三个不同的类别，每类50个样本，分别对应三种不同种类的鸢尾花：
  - Iris Setosa（山鸢尾）
  - Iris Versicolor（变色鸢尾）
  - Iris Virginica（维吉尼亚鸢尾）

**使用场景：**

Iris 数据集常被用于演示监督学习算法，尤其是分类任务。由于其简单性和代表性，它是初学者学习数据分析、模式识别以及机器学习的理想选择。此外，该数据集还经常出现在学术文献中作为基准测试数据集来评估新提出的算法性能。

**数据集获取：**

Iris 数据集可以通过多种方式获得，包括但不限于以下几种途径：

- **Python 的 Scikit-learn 库**：可以直接从 sklearn.datasets 模块加载 iris 数据集。
- **UCI Machine Learning Repository**：可以从 UCI 网站下载原始数据文件。
- **其他开源平台**：如 Kaggle 等平台上也可以找到这个经典的数据集。

**示例代码（使用 Python 和 Scikit-learn 加载 Iris 数据集）：**

```python
from sklearn.datasets import load_iris
import pandas as pd

# 加载数据集
iris = load_iris()

# 将数据转换为 DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# 查看前几行数据
print(df.head())
```

这段代码展示了如何使用 Python 的 Scikit-learn 库轻松加载 Iris 数据集，并将其转换为 Pandas 的 DataFrame 格式以方便后续分析。通过这种方式，你可以快速开始探索和实验各种机器学习模型。





# chapter2.py代码

```python
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

```



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250119164059015.png" alt="image-20250119164059015" style="zoom: 67%;" />



也可以直接用scikit-learn包中的 Perceptron，代码 scikit-learn_perceptron.py

```python
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
```

