# 第1章 统计学习及监督学习概论

Updated 2033 GMT+8 Jan 17 2025.
2025 winter, Complied by Hongfei Yan



Logs:

这一章概念很多。



## 1.1 统计学习

机器学习，往往是指统计机器学习。

统计学习关于数据的基本假设是同类数据具有一定的统计规律性。

统计学习由监督学习（supervised learning）、无监督学习（unsupervised learning ）和强化学习（reinforcement learning ）等组成。

## 1.2 统计学习的分类

### 1.2.1 基本分类

#### 1 监督学习

输入变量与输出变量均为连续变量的预测问题称为回归问题；输出变量为有限个离散变量的预测问题称为分类问题；输入变量与输出变量均为变量序列的预测问题称为标注问题。

训练数据与测试数据被看作是依联合概率分布`P(X,Y)`独立同分布产生的。

监督学习的模型可以是概率模型或非概率模型，由条件概率分布`P(X｜Y)`或决策函数（decision funciton）`Y=f(X)`表示。

#### 2 无监督学习

预测模型表示数据的类别、转换或概率。模型可以实现对数据的聚类、降维或概率估计。包含所有可能的模型的集合称为假设空间。

#### 3 强化学习

本质是学习最优的序列决策。智能系统的目标不是短期奖励的最大化，而是长期累积奖励的最大化。

### 1.2.2 按模型分类

概率模型是生成模型，非概率模型是判别模型。

条件概率分布最大化后得到函数，函数归一化得到条件概率分布。

概率模型一定可以表示为联合概率分布的形式。无论概率模型如何复杂，均可以用最基本的加法规则和乘法规则进行概率推理。

### 1.2.3 按算法分类

分为在线学习（online learning）与批量学习（batch learning）。

在线学习是指每次接受一个样本，进行预测。

### 1.2.4 按技巧分类

#### 1 贝叶斯学习

模型的先验分布是贝叶斯学习的特点。

贝叶斯估计与极大似然估计在思想上有很大的不同，代表着统计学中贝叶斯学派和频率学派对统计的不同认识。

#### 2 核方法

核方法（kernel method）是使用核函数表示和学习非线性的一种机器学习，可以用于监督学习和无监督学习。有一些线性模型的学习方法基于相似度计算，更具体地，向量内积计算。核方法可以把它们扩展到非线性模型的学习，使其应用范围更广泛。

## 1.3 统计学习方法三要素

统计学习方法都是由模型、策略和算法构成。

### 1.3.1 模型

模型的假设空间（hypothesis space）包含所有可能的条件概率分布或决策函数。

### 1.3.2 策略

损失函数度量模型一次预测的好坏，风险函数度量平均意义下模型预测的好坏。

#### 1 损失函数和风险函数

监督学习问题是在假设空间 $\mathcal{F}$ 中选取模型 f 作为决策函数。对于给定的输入 X，由 f(X) 给出相应的输出 Y。这个输出的预测值 f(X) 与真实值 Y 可能一致也可能不一致。用一个损失函数（loss function）或代价函数（cost function）来度量预测错误的程度。损失函数是 f(X) 和 Y 的非负实值函数，记作 `L(Y, f(X))`。

模型 f(X) 关于联合分布 P(X,Y) 的平均意义下的损失、称为风险函数（risk function）或期望损失（expected loss）。

模型 f(X) 关于训练数据集的评价损失称为经验风险（empirical risk）或经验损失（empirical cost）。

根据大数定律，当样本容量趋于无穷时，经验风险 $R_{emp}(f)$ 趋于期望风险 $R_{exp}(f)$。所以一个很自然的想法是用经验风险估计期望风险。但是，由于现实中训练样本数目有限，甚至很小，所以用经验风险估计期望风险常常并不理想，要对经验风险进行一定的矫正。这就关系到监督学习的两个基本策略：经验风险最小化和结构风险最小化。

#### 2 经验风险最小化与结构风险最小化

结构风险最小化（structural risk minimization, SRM）是为了防止过拟合而提出来的策略。结构风险最小化等价于正则化（regularization）。结构风险在经验风险上加上表示模型复杂度的正则表达式项（regularizer）或罚项（penalty term）。

### 1.3.3 算法

统计学习问题归结为最优化问题，统计学习的算法成为求解最优化问题的算法。

## 1.4 模型评估与模型选择

### 1.4.1 训练误差与测试误差

### 1.4.2 过拟合与模型选择

在多项式函数拟合中可以看到，随着多项式次数（模型复杂度）的增加，训练误差会减小，直至趋于0，但是测试误差却不如此，它会随着多项式次数（模型复杂度）的增加先减小而后增大。而最终的目的是使测试误差达到最小。

## 1.5 正则化与交叉验证

### 1.5.1 正则化

模型选择的典型方法是正则化（regularization）。正则化是结构风险最小化策略的实现，是在经验风险上加上一个正则化项（regularizer）或罚项（penalty term）。

正则化复合奥卡姆剃刀（Occam's razor）原理。在所有可能选择的模型中，能够很好地解释已知数据并且十分简单才是最好的模型，也就是应该选择的模型。从贝叶斯估计的角度来看，正则化项对应于模型的先验概率。

### 1.5.2 交叉验证

## 1.6 泛化能力

### 1.6.1 泛化误差

学习方法的泛化能力（generalization ability）是指由该方法学习到的模型对未知数据的预测能力，是学习方法本质上重要的性质。

泛化误差（generalization error）是所学习到的模型的期望风险。

### 1.6.2 泛化误差上界

## 1.7 生产模型与判别模型

## 1.8 监督学习应用

### 1.8.1 分类问题

### 1.8.2 标注问题

### 1.8.3 回归问题

许多领域的任务可以形式化为回归问题，比如，回归可以用于商务领域，作为市场趋势预测、产品质量管理、客户满意度调查、投资风险分析的工具。作为例子，简单介绍股价预测问题。假设知道某一公司在过去不同时间点(比如，每天)的市场上的股票价格（比如，股票平均价格），以及在各个时间点之前可能影响该公司股价的信息（比如，该公司前一周的营业额、利润）。目标是从过去的数据学习一个模型，使它可以基于当前的信息预测该公司下一个时间点的股票价格。可以将这个问题作为回归问题解决。具体地，将影响股价的信息视为自变量（输入的特征），而将股价视为因变量（输出的值）。将过去的数据作为训练数据就可以学习一个回归模型，并对未来的股进行预测。可以看出这是一个困难的预测问题。因为影响股价的因素非常多，我们未必能判断到哪些信息（输入的特征）有用并能得到这些信息。



# 本章概要

1.统计学习或机器学习是关于计算机基于数据构建概率统计模型并运用模型对数据进行分析与预测的一门学科。统计学习包括监督学习、无监督学习和强化学习。

2.统计学习方法三要素 —— 模型、策略、算法，对理解统计学习方法起到提纲挈领的作用。

3.本书第1 篇主要讨论监督学习，监督学习可以概括如下：从给定有限的训练数据出发，假设数据是独立同分布的，而且假设模型属于某个假设空间，应用某一评价准则，从假设空间中选取一个最优的模型，使它对已给训练数据及未知测试数据在给定评价标准意义下有最准确的预测。

4.统计学习中，进行模型选择或者说提高学习的泛化能力是一个重要问题。如果只考虑减少训练误差，就可能产生过拟合现象。模型选择的方法有正则化与交叉验证。学习方法泛化能力的分析是统计学习理论研究的重要课题。

5.分类问题、标注问题和回归问题都是监督学习的重要问题。本书第1 篇介绍的统计学习方法包括感知机、k 近邻法、朴素贝叶斯法、决策树、逻辑斯谛回归与最大熵模型、支持向量机、提升方法、EM算法、隐马尔可夫模型和条件随机场。这些方法是主要的分类、标注以及回归方法。它们又可以归类为生成方法与判别方法。





# chapter1.py代码

```python
import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

#print(np.poly1d([1,2,3]))

# 目标函数
def real_func(x):
    return np.sin(2*np.pi*x)

# 多项式
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)

# 残差
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret

# 十个点
x = np.linspace(0, 1, 10)
x_points = np.linspace(0, 1, 1000)
# 加上正态分布噪音的目标函数的值
y_ = real_func(x)
y = [np.random.normal(0, 0.1) + y1 for y1 in y_]


def fitting(M=0):
    """
    M    为 多项式的次数
    """
    # 随机初始化多项式参数
    p_init = np.random.rand(M + 1)
    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    print('Fitting Parameters:', p_lsq[0])

    # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    return p_lsq

# M=0
p_lsq_0 = fitting(M=0)
plt.show()
# M=1
p_lsq_1 = fitting(M=1)
plt.show()
# M=3
p_lsq_3 = fitting(M=3)
plt.show()
# M=9
p_lsq_9 = fitting(M=9)
plt.show()


# 正则化
regularization = 0.0001


def residuals_func_regularization(p, x, y):
    ret = fit_func(p, x) - y
    ret = np.append(ret,
                    np.sqrt(0.5 * regularization * np.square(p)))  # L2范数作为正则化项
    return ret

# 最小二乘法,加正则化项
p_init = np.random.rand(9 + 1)
p_lsq_regularization = leastsq(
    residuals_func_regularization, p_init, args=(x, y))

plt.plot(x_points, real_func(x_points), label='real')
plt.plot(x_points, fit_func(p_lsq_9[0], x_points), label='fitted curve')
plt.plot(
    x_points,
    fit_func(p_lsq_regularization[0], x_points),
    label='regularization')
plt.plot(x, y, 'bo', label='noise')
plt.legend()
plt.show()
```

> The `fit_func(p_lsq_regularization[0], x_points)` function returns the fitted y-values corresponding to the x-values in `x_points`. The regularization term is not included in the output of `fit_func`, so it does not affect the correspondence between `x_points` and `fit_func(p_lsq_regularization[0], x_points)`.
>
> In other words, `x_points` and `fit_func(p_lsq_regularization[0], x_points)` are still one-to-one corresponding pairs of x and y values, respectively. The regularization term only affects the fitting process but not the final fitted values.
>
> Here is a brief explanation:
> - `x_points` is an array of x-values.
> - `fit_func(p_lsq_regularization[0], x_points)` returns the corresponding y-values based on the fitted polynomial parameters `p_lsq_regularization[0]`.
>
> So, when you plot `x_points` against `fit_func(p_lsq_regularization[0], x_points)`, each x-value in `x_points` has a corresponding y-value in the output of `fit_func`.



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250117221513978.png" alt="image-20250117221513978" style="zoom: 67%;" />

