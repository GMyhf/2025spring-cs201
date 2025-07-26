# Neural Network实现鸢尾花卉数据分类

Updated 2013 GMT+8 Jul 26 2025

2025 spring, Complied by Hongfei Yan



鸢（yuān）尾花数据集包含三个类别，每个类别有50个样本，每个样本有四个特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度。目标是根据这四个特征预测花的种类，属于多分类问题。

## 1.准备数据

鸢尾花数据集通常是通过scikit-learn的datasets模块获取的，所以可能需要结合scikit-learn来加载数据，然后转换成PyTorch的Tensor。

数据预处理方面，需要将特征数据和标签分开。特征数据需要标准化或归一化，因为不同特征的量纲可能不同，这对神经网络的训练有帮助。标签需要转换成数值形式，比如0、1、2，然后可能还需要转换为长整型张量，因为交叉熵损失函数在PyTorch中通常要求这样的格式。

## 2.训练模型和验证

然后，构建神经网络模型。考虑到鸢尾花数据集相对简单，可能不需要很深的网络。一个简单的全连接网络可能就足够了。比如，输入层4个节点，隐藏层可以选择10个节点，输出层3个节点对应三个类别。激活函数可以用ReLU，输出层用Softmax，不过更常见的做法是使用CrossEntropyLoss，它内部已经结合了Softmax，所以输出层不需要显式应用Softmax。

接下来是数据集的划分，通常分为训练集和测试集。这里需要注意的是，鸢尾花数据集样本较少，可能需要进行分层抽样，确保每个类别的样本在训练集和测试集中的比例一致。或者使用交叉验证，但由于用户可能希望一个简单的示例，可能直接采用80-20的划分。

将数据转换为PyTorch的DataLoader，方便批量训练。数据集进行训练可以分小批量处理。

训练过程中，定义损失函数和优化器。交叉熵损失函数适用于多分类问题，优化器可以选择Adam或SGD。学习率需要适当设置，比如0.01或0.001。

训练循环部分，需要迭代多个epoch，在每个epoch中进行前向传播、计算损失、反向传播和参数更新。同时，可以监控训练过程中的损失和准确率的变化。

验证阶段，使用测试集评估模型的准确率，确保模型没有过拟合，并且泛化能力良好。

可能需要注意的问题包括数据标准化应该在训练集上计算均值和标准差，然后应用到测试集，避免数据泄漏。另外，标签是否需要转换为one-hot编码？在PyTorch中，CrossEntropyLoss不需要，因为标签是类别的索引。

现在，将这些步骤整合成代码：

1. 导入必要的库：torch, sklearn.datasets中的load_iris，sklearn.model_selection的train_test_split，以及相关的PyTorch模块。
2. 加载数据，划分训练集和测试集，并进行标准化处理。使用StandardScaler对特征进行标准化，处理时注意仅拟合训练数据。
3. 转换数据为PyTorch的Tensor，并创建DataLoader。可能由于数据量小，直接使用整个数据集作为批量。
4. 定义神经网络模型的结构，使用全连接层和ReLU激活函数。
5. 初始化模型、损失函数和优化器。
6. 编写训练循环，迭代epoch，计算损失，反向传播，记录准确率。
7. 在测试集上评估模型性能。

另外，由于鸢尾花数据集样本较少，模型可能会很快过拟合，所以可能需要早停或者调整网络结构，比如减少隐藏层的大小，或者添加正则化，比如Dropout层。不过，作为示例，可能保持简单即可。

这样应该能完成一个基本的分类任务。不过可能需要调整超参数，比如学习率、epoch次数，或者网络结构，以达到更好的效果。例如，如果准确率不够高，可以尝试增加隐藏层的大小，或者增加更多的层。但鸢尾花数据集相对简单，可能简单的模型就足够。

另外，可能需要注意数据划分时的随机种子，确保结果可复现。此外，标准化处理是否正确应用，避免数据泄漏。在示例中，已经正确使用fit_transform在训练集，transform在测试集。



以下是使用PyTorch实现Fisher鸢尾花数据集分类的Python程序`iris_neural_network.py`，强调使用了神经网络进行分类。

如果缺少module，安装即可，例如：pip install torch，pip install sklearn等。

> 使用 pip install sklearn 可能不会得到预期的结果，因为PyPI（Python Package Index）上的官方包名称是 scikit-learn。但我确实用的 pip install sklearn。这是因为 sklearn 是 scikit-learn 的一个别名，某些情况下通过 pip 安装时能够识别这个别名并正确安装软件包。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 定义模型结构
class IrisNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=10, num_classes=3):
        super(IrisNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

    return running_loss / len(dataloader.dataset)


# 测试函数
def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).float().mean().item()
    return accuracy, predicted


# 主程序
def main():
    # 设置设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    """
    random_state=42
    设定随机数种子，从而确保每次运行代码时数据划分的结果都是相同的。这样做可以使实验具有可重复性，
    有利于调试和结果对比。

    stratify=y
    这个参数表示按照 y 中的标签进行分层抽样，也就是说，训练集和测试集中各类别的
    比例会与原始数据中的类别比例保持一致。这对于类别不平衡的数据集尤为重要，可以
    避免某一类别在划分时被严重低估或过采样。
    """

    # 标准化：只在训练集上计算均值和标准差，再将相同的变换应用到测试集上
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 转换为 Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 构造 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 模型、损失函数、优化器
    model = IrisNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练
    num_epochs = 100
    for epoch in range(1, num_epochs + 1):
        loss = train(model, train_loader, criterion, optimizer, device)
        if epoch % 10 == 0:
            print(f"Epoch [{epoch:3d}/{num_epochs}], Loss: {loss:.4f}")

    # 评估
    test_acc, test_pred = evaluate(model, X_test, y_test, device)
    print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

    # 示例预测
    sample = X_test[0].unsqueeze(0)
    sample_pred = model(sample.to(device))
    pred_class = torch.argmax(sample_pred, dim=1).item()
    print(f"🔍 Sample Prediction: True = {y_test[0].item()}, Predicted = {pred_class}")


if __name__ == "__main__":
    main()

```

> 云虚拟机运行结果：
>
> <img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250223151816482.png" alt="image-20250223151816482" style="zoom:50%;" />



**代码说明：**

1. **数据准备**：
   - 使用scikit-learn加载鸢尾花数据集
   - 将数据划分为训练集（80%）和测试集（20%）
   - 使用标准化处理（StandardScaler）对特征进行归一化

2. **神经网络结构**：
   - 输入层：4个神经元（对应4个特征）
   - 隐藏层：10个神经元（使用ReLU激活函数）
   - 输出层：3个神经元（对应3个类别）

3. **训练配置**：
   - 使用交叉熵损失函数（CrossEntropyLoss）
   - 使用Adam优化器（学习率0.01）
   - 训练100个epoch

4. **训练过程**：
   - 每个epoch记录损失和准确率
   - 每10个epoch打印训练进度

5. **评估与预测**：
   - 最终在测试集上评估模型准确率
   - 包含一个预测示例展示

**输出示例：**

```
$ python iris_neural_network.py 
Epoch [ 10/100], Loss: 0.2363
Epoch [ 20/100], Loss: 0.0899
Epoch [ 30/100], Loss: 0.0614
Epoch [ 40/100], Loss: 0.0634
Epoch [ 50/100], Loss: 0.0498
Epoch [ 60/100], Loss: 0.0492
Epoch [ 70/100], Loss: 0.0492
Epoch [ 80/100], Loss: 0.0451
Epoch [ 90/100], Loss: 0.0479
Epoch [100/100], Loss: 0.0436

✅ Test Accuracy: 100.00%
🔍 Sample Prediction: True = 0, Predicted = 0
```

**注意事项：**

1. 由于数据集较小，模型可能很快达到100%训练准确率
2. 可以调整以下参数优化性能：
   - 隐藏层大小（10）
   - 学习率（0.01）
   - epoch数量（100）
   - 优化器（尝试SGD等）
3. 添加正则化（如Dropout层）可以防止过拟合
4. 可以使用GPU加速（将数据和模型移动到`cuda`设备）
