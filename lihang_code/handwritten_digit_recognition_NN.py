import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

np.random.seed(2023)


def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        f.read(16)  # 跳过文件头
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(-1, 28 * 28).astype('float32') / 255.0


def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        f.read(8)  # 跳过文件头
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


# 定义神经网络类（不变）
class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.layers = layers
        self.alpha = alpha
        self.weights = []
        self.biases = []
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i - 1], layers[i]))
            self.biases.append(np.random.randn(layers[i]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, inputs):
        self.activations = [inputs]
        self.weighted_inputs = []
        for i in range(len(self.weights)):
            weighted_input = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.weighted_inputs.append(weighted_input)
            activation = self.sigmoid(weighted_input)
            self.activations.append(activation)
        return self.activations[-1]

    def backpropagate(self, expected):
        errors = [expected - self.activations[-1]]
        deltas = [errors[-1] * self.sigmoid_derivative(self.activations[-1])]

        for i in range(len(self.weights) - 1, 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            errors.append(error)
            delta = errors[-1] * self.sigmoid_derivative(self.activations[i])
            deltas.append(delta)
        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] += self.alpha * np.array([self.activations[i]]).T.dot(np.array([deltas[i]]))
            self.biases[i] += self.alpha * np.sum(deltas[i], axis=0)

    def train(self, inputs, expected_outputs, epochs):
        for i in tqdm(range(epochs)):
            for j in range(len(inputs)):
                self.feedforward(inputs[j])
                self.backpropagate(expected_outputs[j])


# 加载本地数据
train_images = load_mnist_images("./MNIST/train-images-idx3-ubyte")
train_labels = load_mnist_labels("./MNIST/train-labels-idx1-ubyte")
test_images = load_mnist_images("./MNIST/t10k-images-idx3-ubyte")
test_labels = load_mnist_labels("./MNIST/t10k-labels-idx1-ubyte")

# 处理标签为独热编码
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.transform(test_labels)

# 划分训练集和测试集（可选）
X_train, X_test, y_train, y_test = train_images, test_images, train_labels, test_labels

# 初始化并训练神经网络
nn = NeuralNetwork([784, 100, 50, 10], alpha=0.1)
nn.train(X_train, y_train, epochs=10)

# 使用测试集评估模型
correct = 0
for i in range(len(X_test)):
    output = nn.feedforward(X_test[i])
    prediction = np.argmax(output)
    actual = np.argmax(y_test[i])
    if prediction == actual:
        correct += 1

accuracy = correct / len(X_test) * 100
print("Accuracy: {:.2f} %".format(accuracy))

# 100%|██████████| 10/10 [01:22<00:00,  8.24s/it]
# Accuracy: 94.47 %