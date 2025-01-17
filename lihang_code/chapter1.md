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

在线学习是指每次接受一个样本，进行预测。







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