# 神经网络中的反向传播

Backpropagation in Neural Network

Last Updated : 26 Nov, 2025

https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/



反向传播（Back Propagation），又称为“误差的反向传播”，是一种用于训练神经网络的方法。其目标是通过调整网络中的权重（weights）和偏置（biases），来减小模型预测输出与实际输出之间的差异。

它通过迭代方式更新权重和偏置，以最小化损失函数（cost function）。在每一个训练周期（epoch）中，模型会根据误差梯度（error gradient）更新参数，常用的优化算法包括梯度下降（Gradient Descent）或随机梯度下降（SGD）。该算法使用微积分中的链式法则来计算梯度，从而能够有效地穿越复杂的神经网络结构，优化损失函数。

> Back Propagation is also known as "Backward Propagation of Errors" is a method used to train neural network . Its goal is to reduce the difference between the model’s predicted output and the actual output by adjusting the weights and biases in the network.
>
> It works iteratively to adjust weights and bias to minimize the cost function. In each epoch the model adapts these parameters by reducing loss by following the error gradient. It often uses optimization algorithms like **gradient descent** or **stochastic gradient descent**. The algorithm computes the gradient using the chain rule from calculus allowing it to effectively navigate complex layers in the neural network to minimize the cost function.

![Backpropagation-in-Neural-Network-1](https://media.geeksforgeeks.org/wp-content/uploads/20250701163824448467/Backpropagation-in-Neural-Network-1.webp)

<center>Fig(a) A simple illustration of how the backpropagation works by adjustments of weights</center>

<center>图(a)：通过权重调整，简单展示反向传播的工作方式</center>



**反向传播的重要性：**

- **高效的权重更新**：利用链式法则计算损失函数对每个权重的梯度，从而高效地更新参数。
- **良好的扩展性**：适用于多层结构和复杂架构，是深度学习可行的核心算法。
- **自动学习能力**：训练过程自动进行，模型会不断调整自身来优化性能。

> **Back Propagation** plays a critical role in how neural networks improve over time. Here's why:
>
> 1. **Efficient Weight Update**: It computes the gradient of the loss function with respect to each weight using the chain rule making it possible to update weights efficiently.
> 2. **Scalability**: The Back Propagation algorithm scales well to networks with multiple layers and complex architectures making deep learning feasible.
> 3. **Automated Learning**: With Back Propagation the learning process becomes automated and the model can adjust itself to optimize its performance.



## 反向传播算法的工作流程

反向传播算法包括两个主要步骤：**前向传播（Forward Pass）** 和 **反向传播（Backward Pass）**

### 1. Forward Pass Work前向传播

输入数据从输入层开始，经过带权重的连接传递到隐藏层。例如，一个有两个隐藏层 h1 和 h2 的网络中，h1 的输出作为 h2 的输入。在应用激活函数前，还会加上偏置项。

每一层都会计算输入的加权和（记作 `a`），再通过如 ReLU 等激活函数得到输出 `o`。最终，输出层通常会使用 softmax 激活函数将结果转换为分类概率。

> ### Working of Back Propagation Algorithm
>
> The Back Propagation algorithm involves two main steps: the **Forward Pass** and the **Backward Pass**.
>
> ### 1. Forward Pass Work
>
> In **forward pass** the input data is fed into the input layer. These inputs combined with their respective weights are passed to hidden layers. For example in a network with two hidden layers (h1 and h2) the output from h1 serves as the input to h2. Before applying an activation function, a bias is added to the weighted inputs.
>
> Each hidden layer computes the weighted sum (`a`) of the inputs then applies an activation function like [**ReLU (Rectified Linear Unit)**](https://www.geeksforgeeks.org/deep-learning/relu-activation-function-in-deep-learning/) to obtain the output (`o`). The output is passed to the next layer where an activation function such as [**softmax**](https://www.geeksforgeeks.org/deep-learning/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/) converts the weighted outputs into probabilities for classification.

![Backpropagation-in-Neural-Network-2](https://media.geeksforgeeks.org/wp-content/uploads/20250701163954688803/Backpropagation-in-Neural-Network-2.webp)

<center>The forward pass using weights and biases</center>

> h1,h2，表示隐藏层的两个神经元



### 2. Backward Pass反向传播

反向传播阶段会将预测输出与实际输出的误差向后传递，并调整每一层的权重和偏置。常见的误差计算方法是**均方误差（MSE）**：

$MSE = (\text{Predicted Output} − \text{Actual Output})^2$

在误差计算之后，通过链式法则计算梯度，这些梯度用于指导权重和偏置的更新方向和幅度。反向传播过程是逐层执行的，<mark>激活函数的导数在梯度计算中起着关键作用</mark>。



**反向传播的示例：机器学习中的案例**

假设我们使用 sigmoid 激活函数，目标输出为 0.5，学习率为 1。

> ### 2. Backward Pass
>
> In the backward pass the error (the difference between the predicted and actual output) is propagated back through the network to adjust the weights and biases. One common method for error calculation is the [**Mean Squared Error (MSE)**](https://www.geeksforgeeks.org/maths/mean-squared-error/) given by:
>
> $MSE = (\text{Predicted Output} − \text{Actual Output})^2$
>
> Once the error is calculated the network adjusts weights using **gradients** which are computed with the chain rule. These gradients indicate how much each weight and bias should be adjusted to minimize the error in the next iteration. The backward pass continues layer by layer ensuring that the network learns and improves its performance. The activation function through its derivative plays a crucial role in computing these gradients during Back Propagation.
>
> 
>
> ## Example of Back Propagation in Machine Learning
>
> Let’s walk through an example of Back Propagation in machine learning. Assume the neurons use the sigmoid activation function for the forward and backward pass. The target output is 0.5 and the learning rate is 1.

![Backpropagation-in-Neural-Network-3](https://media.geeksforgeeks.org/wp-content/uploads/20250701164029130520/Backpropagation-in-Neural-Network-3.webp)

<center>Example (1) of backpropagation sum</center>



## 前向传播Forward Propagation

### 1. Initial Calculation初始计算

The weighted sum at each node is calculated using:

> $a_j=\sum(w_{i,j}∗x_i)$

Where,

- $a_j$ is the weighted sum of all the inputs and weights at each node
- $w_{i,j}$ represents the weights between the $i^{th}$ input and the $j^{th}$ neuron
- $x_i$ represents the value of the $i^{th}$ input

`O (output):`After applying the activation function to `a`, we get the output of the neuron:

> $o_j = \text{activation function}(a_j)$

### 2. Sigmoid Function

The sigmoid function returns a value between 0 and 1, introducing non-linearity into the model.

> $y_j = \frac{1}{1+e^{−a_j}}$ 

![Backpropagation-in-Neural-Network-4](https://media.geeksforgeeks.org/wp-content/uploads/20250701164114106895/Backpropagation-in-Neural-Network-4.webp)

<center>To find the outputs of y3, y4 and y5</center>



### 3. Computing Outputs输出计算

h1 节点：
$$
a_1 = (w_{1,1} \times x_1) + (w_{2,1} \times x_2)
$$
$$
a_1 = (0.2 \times 0.35) + (0.2 \times 0.7) = 0.21
$$

计算完 $a_1$ 后，我们可以继续计算 $y_3$ 的值：

$$
y_j = F(a_j) = \frac{1}{1 + e^{-a_1}}
$$
$$
y_3 = F(0.21) = \frac{1}{1 + e^{-0.21}} = 0.56
$$



h2 节点：
$$
a_2 = (w_{1,2} \times x_1) + (w_{2,2} \times x_2) = (0.3 \times 0.35) + (0.3 \times 0.7) = 0.315
$$
$$
y_4 = F(0.315) = \frac{1}{1 + e^{-0.315}} = 0.578
$$



输出节点 O3：
$$
a_3 = (w_{1,3} \times y_3) + (w_{2,3} \times y_4) = (0.3 \times 0.56) + (0.9 \times 0.58) = 0.702
$$
$$
y_5 = F(0.702) = \frac{1}{1 + e^{-0.702}} = 0.67
$$



> At h1 node
>
> Once we calculated the a1 value, we can now proceed to find the y3 value:
>
> Similarly find the values of y4 at h2 and y5 at O3



![Backpropagation-in-Neural-Network-5](https://media.geeksforgeeks.org/wp-content/uploads/20250701164956768059/Backpropagation-in-Neural-Network-5.webp)

<center>Values of y3, y4 and y5</center>



### 4. Error Calculation误差计算

Our actual output is 0.5 but we obtained 0.67**.** To calculate the error we can use the below formula:

> $Error_j=y_{target}−y_5$ 

=> 0.5−0.67=−0.17

Using this error value we will be backpropagating.



## 反向传播Back Propagation

### 1. Calculating Gradients计算梯度

The change in each weight is calculated as:

> $Δw_{ij}=η×δ_j×O_j$

Where:

- $δ_j$ is the error term for each unit,
- $η$ is the learning rate.

### 2. Output Unit Error输出层误差

For O3:

> $δ_5=y_5(1−y_5)(y_{target}−y_5)$

=0.67(1−0.67)(−0.17)=−0.0376

### 3. Hidden Unit Error隐藏层误差

For h1:

> $δ_3=y_3(1−y_3)(w_{1,3}×δ_5)$

=0.56(1−0.56)(0.3×−0.0376)=−0.0027



For h2:

> $δ_4=y_4(1−y_4)(w_{2,3}×δ_5)$

=0.59(1−0.59)(0.9×−0.0376)=−0.0819



### 4. Weight Updates权重更新

For the weights from hidden to output layer:

> $Δw_{2,3}=1×(−0.0376)×0.59=−0.022184$

New weight:

> $w_{2,3}(new)=−0.022184+0.9=0.877816$

For weights from input to hidden layer:

> $Δw_{1,1}=1×(−0.0027)×0.35=0.000945$

New weight:

> $w_{1,1}(new)=0.000945+0.2=0.200945$

Similarly other weights are updated:

- $w_{1,2}(new)=0.273225$
- $w_{1,3}(new)=0.086615$
- $w_{2,1}(new)=0.269445$
- $w_{2,2}(new)=0.18534$

The updated weights are illustrated below

![Backpropagation-in-Neural-Network-5](https://media.geeksforgeeks.org/wp-content/uploads/20250701165041328619/Backpropagation-in-Neural-Network-5.webp)

<center>Through backward pass the weights are updated</center>

> 上图权重没有更新，例如：$w_{2,2}$应该更新为0.18534



After updating the weights the forward pass is repeated yielding:

- y3=0.57
- y4=0.56
- y5=0.61

仍未达到目标值 0.5，因此继续进行反向传播，直到收敛。

> Since y5=0.61 is still not the target output the process of calculating the error and backpropagating continues until the desired output is reached.



This process demonstrates how Back Propagation iteratively updates weights by minimizing errors until the network accurately predicts the output.

> $Error=y_{target}−y_5$

=0.5−0.61=−0.11=0.5−0.61=−0.11

This process is said to be continued until the actual output is gained by the neural network.



## Back Propagation Implementation in Python for XOR Problem

This code demonstrates how Back Propagation is used in a neural network to solve the XOR problem. The neural network consists of:

### 1. Defining Neural Network定义神经网络结构

输入层：2个节点，隐藏层：4个神经元，输出层：1个神经元，激活函数：Sigmoid

> We define a neural network as Input layer with 2 inputs, Hidden layer with 4 neurons, Output layer with 1 output neuron and use **Sigmoid** function as activation function.

- **self.input_size = input_size**: stores the size of the input layer
- **self.hidden_size = hidden_size:** stores the size of the hidden layer
- **self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)**: initializes weights for input to hidden layer
- **self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)**: initializes weights for hidden to output layer
- **self.bias_hidden = np.zeros((1, self.hidden_size)):** initializes bias for hidden layer
- **self.bias_output = np.zeros((1, self.output_size)):** initializes bias for output layer



```python3

import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights_input_hidden = np.random.randn(
            self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(
            self.hidden_size, self.output_size)

        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
```



### 2. Defining Feed Forward Network定义前向传播

In Forward pass inputs are passed through the network activating the hidden and output layers using the sigmoid function.

- **self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden**: calculates activation for hidden layer
- **self.hidden_output= self.sigmoid(self.hidden_activation)**: applies activation function to hidden layer
- **self.output_activation= np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output:** calculates activation for output layer
- **self.predicted_output = self.sigmoid(self.output_activation):** applies activation function to output layer





```python3

def feedforward(self, X):
    self.hidden_activation = np.dot(
        X, self.weights_input_hidden) + self.bias_hidden
    self.hidden_output = self.sigmoid(self.hidden_activation)

    self.output_activation = np.dot(
        self.hidden_output, self.weights_hidden_output) + self.bias_output
    self.predicted_output = self.sigmoid(self.output_activation)

    return self.predicted_output
```



### 3. Defining Backward Network定义反向传播

In Backward pass or Back Propagation the errors between the predicted and actual outputs are computed. The gradients are calculated using the derivative of the sigmoid function and weights and biases are updated accordingly.

- **output_error = y - self.predicted_output:** calculates the error at the output layer
- **output_delta = output_error * self.sigmoid_derivative(self.predicted_output):** calculates the delta for the output layer
- **hidden_error = np.dot(output_delta, self.weights_hidden_output.T):** calculates the error at the hidden layer
- **hidden_delta = hidden_error \* self.sigmoid_derivative(self.hidden_output):** calculates the delta for the hidden layer
- **self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate:** updates weights between hidden and output layers
- **self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate:** updates weights between input and hidden layers



```python3

def backward(self, X, y, learning_rate):
    output_error = y - self.predicted_output
    output_delta = output_error * \
        self.sigmoid_derivative(self.predicted_output)

    hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
    hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

    self.weights_hidden_output += np.dot(self.hidden_output.T,
                                         output_delta) * learning_rate
    self.bias_output += np.sum(output_delta, axis=0,
                               keepdims=True) * learning_rate
    self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
    self.bias_hidden += np.sum(hidden_delta, axis=0,
                               keepdims=True) * learning_rate
```



### 4. Training Network训练网络

The network is trained over 10,000 epochs using the Back Propagation algorithm with a learning rate of 0.1 progressively reducing the error.

- **output = self.feedforward(X):** computes the output for the current inputs
- **self.backward(X, y, learning_rate):** updates weights and biases using Back Propagation
- **loss = np.mean(np.square(y - output)):** calculates the mean squared error (MSE) loss



```python3

def train(self, X, y, epochs, learning_rate):
    for epoch in range(epochs):
        output = self.feedforward(X)
        self.backward(X, y, learning_rate)
        if epoch % 4000 == 0:
            loss = np.mean(np.square(y - output))
            print(f"Epoch {epoch}, Loss:{loss}")
```

### 5. Testing Neural Network测试神经网络

- **X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]):** defines the input data
- **y = np.array([[0], [1], [1], [0]]):** defines the target values
- **nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1):** initializes the neural network
- **nn.train(X, y, epochs=10000, learning_rate=0.1):** trains the network
- **output = nn.feedforward(X):** gets the final predictions after training





```python3

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.train(X, y, epochs=10000, learning_rate=0.1)

output = nn.feedforward(X)
print("Predictions after training:")
print(output)
```

**Output:**

![Screenshot-2025-03-07-130223](https://media.geeksforgeeks.org/wp-content/uploads/20250307130234877256/Screenshot-2025-03-07-130223.png)

<center>Trained Model</center>



训练初期损失为 0.2713，逐步下降到 0.0066（第8000轮）。最终模型可以很好地逼近 XOR 函数的输出，即：

- 对于输入 [0,0] 和 [1,1]，输出接近 0

- 对于输入 [0,1] 和 [1,0]，输出接近 1

  

> - The output shows the training progress of a neural network over 10,000 epochs. Initially the loss was high (0.2713) but it gradually decreased as the network learned reaching a low value of 0.0066 by epoch 8000.
> - The final predictions are close to the expected XOR outputs: approximately 0 for [0, 0] and [1, 1] and approximately 1 for [0, 1] and [1, 0] indicating that the network successfully learned to approximate the XOR function.



## 反向传播的优点

**易于实现**：适合初学者，无需太多神经网络背景

**结构简单，灵活应用**：从简单前馈到复杂卷积/循环网络都可使用

**高效**：直接根据误差更新权重，学习速度快

**良好的泛化能力**：有助于模型在新数据上表现更好

**可扩展性好**：适用于大型数据集和深层模型

> **Advantages of Back Propagation for Neural Network Training**
>
> The key benefits of using the Back Propagation algorithm are:
>
> 1. **Ease of Implementation:** Back Propagation is beginner-friendly requiring no prior neural network knowledge and simplifies programming by adjusting weights with error derivatives.
> 2. **Simplicity and Flexibility:** Its straightforward design suits a range of tasks from basic feedforward to complex convolutional or recurrent networks.
> 3. **Efficiency**: Back Propagation accelerates learning by directly updating weights based on error especially in deep networks.
> 4. **Generalization:** It helps models generalize well to new data improving prediction accuracy on unseen examples.
> 5. **Scalability:** The algorithm scales efficiently with larger datasets and more complex networks making it ideal for large-scale tasks.



## 反向传播面临的挑战

**梯度消失**：在深层网络中梯度可能过小，导致学习困难（特别是在使用 sigmoid/tanh 时）

**梯度爆炸**：梯度可能变得过大，使训练不稳定

**过拟合**：模型结构过于复杂时，可能记住训练集而非学习一般性规律

> **Challenges with Back Propagation**
>
> While Back Propagation is useful it does face some challenges:
>
> 1. **Vanishing Gradient Problem**: In deep networks the gradients can become very small during Back Propagation making it difficult for the network to learn. This is common when using activation functions like sigmoid or tanh.
> 2. **Exploding Gradients**: The gradients can also become excessively large causing the network to diverge during training.
> 3. **Overfitting:** If the network is too complex it might memorize the training data instead of learning general patterns.



## 完整xor_nn代码

```python
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
        # 计算输出层的delta（梯度的一部分，损失对激活输入的梯度）
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)  # Delta = 误差 × 激活函数导数
        # output_delta = (y - ŷ) * σ'(z_output)

        # 计算隐藏层误差（反向传播）
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)  # 将误差从输出层反向传播到隐藏层
        # hidden_error = output_delta @ W_hidden_output^T
        # 计算隐藏层的delta（损失对隐藏层激活输入的梯度）
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)  # Delta = 误差 × 激活函数导数
        # hidden_delta = (hidden_error) * σ'(z_hidden)

        # 更新权重和偏置（使用梯度下降法）
        # 计算并更新隐藏层到输出层的权重
        self.weights_hidden_output += np.dot(self.hidden_output.T,
                                             output_delta) * learning_rate  # 权重更新量 = 学习率 × (隐藏层输出转置 × 输出层delta)
        # W_hidden_output += learning_rate * (hidden_output^T @ output_delta)

        # 更新输出层偏置，基于所有样本的输出层delta沿列求和
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate  # 偏置更新量 = 学习率 × (沿列求和输出层delta)
        # b_output += learning_rate * sum(output_delta)

        # 计算并更新从输入层到隐藏层的权重的梯度
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate  # 权重更新量 = 学习率 × (输入数据转置 × 隐藏层delta)
        # W_input_hidden += learning_rate * (X^T @ hidden_delta)

        # 更新隐藏层偏置，基于所有样本的隐藏层delta沿列求和
        # axis=0：沿列求和，聚合所有样本的梯度
        # keepdims=True：保持原矩阵的行数维度，确保偏置更新的形状兼容性
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate  # 偏置更新量 = 学习率 × (沿列求和隐藏层delta)
        # b_hidden += learning_rate * sum(hidden_delta)

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
```

