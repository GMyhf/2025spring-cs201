# 对于XOR问题（输入为[0,0], [0,1], [1,0], [1,1]），期望输出为[0,1,1,0]
# 手动实现反向传播，没有使用深度学习框架，这有助于理解底层原理
# https://www.geeksforgeeks.org/backpropagation-in-neural-network/
import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size  # 输入特征维度
        self.hidden_size = hidden_size  # 隐藏层神经元数量
        self.output_size = output_size  # 输出层神经元数量

        # 输入层到隐藏层的权重，形状为 (输入维度, 隐藏层维度)
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        # 隐藏层到输出层的权重，形状为 (隐藏层维度, 输出层维度)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        # 隐藏层的偏置，形状为 (1, 隐藏层维度)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        # 输出层的偏置，形状为 (1, 输出层维度)
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):  # 激活函数，将输入压缩到(0,1)区间
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)  # Sigmoid的导数，用于反向传播中的梯度计算

    def feedforward(self, X):
        # 隐藏层计算
        self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden  # 线性变换
        self.hidden_output = self.sigmoid(self.hidden_activation)  # 激活函数

        # 输出层计算
        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_activation)

        return self.predicted_output

    def backward(self, X, y, learning_rate):
        # 计算输出层误差
        output_error = y - self.predicted_output  # 误差 = 真实值 - 预测值
        # 计算输出层的delta（不是直接的梯度，而是用于计算梯度的一部分）
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)  # Delta = 误差 × 激活函数导数

        # 计算隐藏层接收到的误差
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)  # 将误差从输出层反向传播到隐藏层
        # 计算隐藏层的delta（不是直接的梯度，而是用于计算梯度的一部分）
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)  # Delta = 误差 × 激活函数导数

        # 更新权重和偏置（使用梯度下降法）
        # 计算并更新从隐藏层到输出层的权重的梯度
        self.weights_hidden_output += np.dot(self.hidden_output.T,
                                             output_delta) * learning_rate  # 权重更新量 = 学习率 × (隐藏层输出转置 × 输出层delta)
        # 更新输出层偏置，基于所有样本的输出层delta沿列求和
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate  # 偏置更新量 = 学习率 × (沿列求和输出层delta)
        # 计算并更新从输入层到隐藏层的权重的梯度
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate  # 权重更新量 = 学习率 × (输入数据转置 × 隐藏层delta)

        # 更新隐藏层偏置，基于所有样本的隐藏层delta沿列求和
        # axis=0：沿列求和，聚合所有样本的梯度
        # keepdims=True：保持原矩阵的行数维度，确保偏置更新的形状兼容性
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate  # 偏置更新量 = 学习率 × (沿列求和隐藏层delta)
        
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.feedforward(X)  # 前向传播
            self.backward(X, y, learning_rate)  # 反向传播与参数更新
            if epoch % 4000 == 0:
                loss = np.mean(np.square(y - output))  # 计算均方误差
                print(f"Epoch {epoch}, Loss:{loss}")


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 输入维度 2（二维二进制特征），隐藏层4个神经元，输出层1个神经元（二分类问题）
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
# 训练总轮次, 学习率
nn.train(X, y, epochs=10000, learning_rate=0.1)

output = nn.feedforward(X)
print("Predictions after training:")
print(output)
"""
Epoch 0, Loss:0.2653166263520884
Epoch 4000, Loss:0.007000926683956338
Epoch 8000, Loss:0.001973630232951721
Predictions after training:
[[0.03613239]
 [0.96431351]
 [0.96058291]
 [0.03919372]]
"""
