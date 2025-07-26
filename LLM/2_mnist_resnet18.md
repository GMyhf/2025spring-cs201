# PyTorch 实例 - MNIST 图像分类项目

Updated 1934 GMT+8 Jul 26 2025

2025 summer, Complied by Hongfei Yan



通过手写数字识别的实例，我们可以看到神经网络的强大，也可以更好地理解它是如何运行的。

MNIST是一个著名的手写体数字识别数据集，训练集包含60,000个样本和标签，测试集包含10,000 个样本和标签。其中样本为代表0~9中的一个数字的灰度图片，对应一个所代表数字的标签，图片大小28*28，且数字出现在图片正中间。

**数据集**: 数据集 MNIST。

**数据预处理**: 因为 MNIST 图像是单通道（灰度图），而 ResNet18 模型是为 RGB 图像设计的，期望输入的通道数为 3，所以调整数据预处理的部分。

**输出层**: 将模型输出层为 10 类，因为 MNIST 有 10 个类别（0 到 9）。

**模型调整**: 使用的模型仍然是 `ResNet18`，但需要考虑到 MNIST 是单通道图像，调整输入层。

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import time

def main():
    # 1. 数据增强 + 预处理
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # MNIST 是单通道，使用 (0.5,) 来规范化
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载 MNIST 数据集
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

    classes = [str(i) for i in range(10)]  # MNIST 类别是 0 到 9

    # 2. 设置设备和模型
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # 加载预定义的 ResNet18 并修改输入层和输出层
    net = models.resnet18(weights=None)
    # 修改输入层的第一个卷积层，使其接受单通道（1通道灰度图像）
    net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = nn.Linear(net.fc.in_features, 10)  # MNIST 10 类
    net.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 3. 训练过程
    best_loss = float('inf')
    patience = 10  # 提高耐心
    patience_counter = 0

    start_time = time.time()
    print("Starting training with early stopping...")
    for epoch in range(800):  # 可适当增大 epoch
        net.train()
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i % 100 == 99:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}")

        avg_loss = epoch_loss / len(trainloader)
        print(f"[{epoch+1}] Avg Loss: {avg_loss:.3f}")

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"✅ Training completed in {execution_time_minutes:.2f} minutes.")

    # 保存模型
    torch.save(net.state_dict(), './resnet18_mnist.pth')

    # 4. 测试准确率
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test images: {100 * correct / total:.2f}%")

    # 每类准确率
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                class_correct[labels[i]] += c[i].item()
                class_total[labels[i]] += 1

    for i in range(10):
        print(f'Accuracy of {classes[i]:5s}: {100 * class_correct[i] / class_total[i]:.2f}%')

    # --- 可视化预测 ---

    def imshow_grid(images, labels, preds=None, classes=None, rows=8, cols=8):
        images = images.cpu() / 2 + 0.5  # unnormalize
        npimg = images.numpy()
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        for i in range(rows * cols):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            img = np.transpose(npimg[i], (1, 2, 0))
            ax.imshow(img.squeeze(), cmap="gray")
            title = f'{classes[labels[i]]}'
            if preds is not None:
                title += f'\n→ {classes[preds[i]]}'
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    # 获取一批图像用于显示
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    while images.size(0) < 64:
        more_images, more_labels = next(dataiter)
        images = torch.cat([images, more_images], dim=0)
        labels = torch.cat([labels, more_labels], dim=0)
    images = images[:64]
    labels = labels[:64]

    # 预测
    net.eval()
    with torch.no_grad():
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs, 1)

    # 显示图像网格
    imshow_grid(images, labels, predicted.cpu(), classes=classes, rows=8, cols=8)

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()


```



运行机器

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202507261935350.png" alt="b452b39cfb47eb8bf5b640c828b6b71b" style="zoom:50%;" />



运行结果

```
/Users/hfyan/miniconda3/bin/python /Users/hfyan/git/2025spring-cs201/LLM/mnist_resnet18.py 
100%|██████████| 9.91M/9.91M [02:52<00:00, 57.6kB/s]
100%|██████████| 28.9k/28.9k [00:00<00:00, 97.2kB/s]
100%|██████████| 1.65M/1.65M [00:04<00:00, 374kB/s]
100%|██████████| 4.54k/4.54k [00:00<00:00, 6.74kB/s]
Using device: mps
Starting training with early stopping...
/Users/hfyan/miniconda3/lib/python3.10/site-packages/torch/utils/data/dataloader.py:683: UserWarning: 'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.
  warnings.warn(warn_msg)
[1,   100] loss: 0.136
[1,   200] loss: 0.132
[1,   300] loss: 0.035
[1,   400] loss: 0.098
[1] Avg Loss: 0.150
[2,   100] loss: 0.137
[2,   200] loss: 0.030
[2,   300] loss: 0.030
[2,   400] loss: 0.015
[2] Avg Loss: 0.052
[3,   100] loss: 0.018
[3,   200] loss: 0.105
[3,   300] loss: 0.078
[3,   400] loss: 0.026
[3] Avg Loss: 0.039
[4,   100] loss: 0.032
[4,   200] loss: 0.056
[4,   300] loss: 0.008
[4,   400] loss: 0.013
[4] Avg Loss: 0.031
[5,   100] loss: 0.003
[5,   200] loss: 0.025
[5,   300] loss: 0.029
[5,   400] loss: 0.022
[5] Avg Loss: 0.027
[6,   100] loss: 0.041
[6,   200] loss: 0.022
[6,   300] loss: 0.047
[6,   400] loss: 0.005
[6] Avg Loss: 0.023
[7,   100] loss: 0.039
[7,   200] loss: 0.000
[7,   300] loss: 0.022
[7,   400] loss: 0.014
[7] Avg Loss: 0.018
[8,   100] loss: 0.001
[8,   200] loss: 0.044
[8,   300] loss: 0.021
[8,   400] loss: 0.002
[8] Avg Loss: 0.019
No improvement. Patience: 1/10
[9,   100] loss: 0.002
[9,   200] loss: 0.020
[9,   300] loss: 0.002
[9,   400] loss: 0.007
[9] Avg Loss: 0.017
[10,   100] loss: 0.027
[10,   200] loss: 0.034
[10,   300] loss: 0.031
[10,   400] loss: 0.004
[10] Avg Loss: 0.016
[11,   100] loss: 0.003
[11,   200] loss: 0.004
[11,   300] loss: 0.005
[11,   400] loss: 0.003
[11] Avg Loss: 0.015
[12,   100] loss: 0.011
[12,   200] loss: 0.000
[12,   300] loss: 0.031
[12,   400] loss: 0.003
[12] Avg Loss: 0.015
No improvement. Patience: 1/10
[13,   100] loss: 0.002
[13,   200] loss: 0.002
[13,   300] loss: 0.002
[13,   400] loss: 0.019
[13] Avg Loss: 0.013
[14,   100] loss: 0.019
[14,   200] loss: 0.004
[14,   300] loss: 0.025
[14,   400] loss: 0.003
[14] Avg Loss: 0.013
No improvement. Patience: 1/10
[15,   100] loss: 0.003
[15,   200] loss: 0.001
[15,   300] loss: 0.011
[15,   400] loss: 0.056
[15] Avg Loss: 0.013
[16,   100] loss: 0.034
[16,   200] loss: 0.008
[16,   300] loss: 0.001
[16,   400] loss: 0.003
[16] Avg Loss: 0.011
[17,   100] loss: 0.008
[17,   200] loss: 0.001
[17,   300] loss: 0.001
[17,   400] loss: 0.001
[17] Avg Loss: 0.011
[18,   100] loss: 0.009
[18,   200] loss: 0.015
[18,   300] loss: 0.002
[18,   400] loss: 0.036
[18] Avg Loss: 0.013
No improvement. Patience: 1/10
[19,   100] loss: 0.019
[19,   200] loss: 0.001
[19,   300] loss: 0.023
[19,   400] loss: 0.005
[19] Avg Loss: 0.011
[20,   100] loss: 0.002
[20,   200] loss: 0.007
[20,   300] loss: 0.007
[20,   400] loss: 0.005
[20] Avg Loss: 0.011
No improvement. Patience: 1/10
[21,   100] loss: 0.001
[21,   200] loss: 0.008
[21,   300] loss: 0.012
[21,   400] loss: 0.005
[21] Avg Loss: 0.011
[22,   100] loss: 0.007
[22,   200] loss: 0.001
[22,   300] loss: 0.001
[22,   400] loss: 0.002
[22] Avg Loss: 0.011
No improvement. Patience: 1/10
[23,   100] loss: 0.003
[23,   200] loss: 0.002
[23,   300] loss: 0.001
[23,   400] loss: 0.014
[23] Avg Loss: 0.011
No improvement. Patience: 2/10
[24,   100] loss: 0.003
[24,   200] loss: 0.001
[24,   300] loss: 0.003
[24,   400] loss: 0.002
[24] Avg Loss: 0.010
[25,   100] loss: 0.016
[25,   200] loss: 0.002
[25,   300] loss: 0.010
[25,   400] loss: 0.000
[25] Avg Loss: 0.009
[26,   100] loss: 0.001
[26,   200] loss: 0.002
[26,   300] loss: 0.006
[26,   400] loss: 0.021
[26] Avg Loss: 0.008
[27,   100] loss: 0.002
[27,   200] loss: 0.002
[27,   300] loss: 0.017
[27,   400] loss: 0.000
[27] Avg Loss: 0.010
No improvement. Patience: 1/10
[28,   100] loss: 0.001
[28,   200] loss: 0.012
[28,   300] loss: 0.009
[28,   400] loss: 0.000
[28] Avg Loss: 0.008
No improvement. Patience: 2/10
[29,   100] loss: 0.001
[29,   200] loss: 0.008
[29,   300] loss: 0.009
[29,   400] loss: 0.031
[29] Avg Loss: 0.010
No improvement. Patience: 3/10
[30,   100] loss: 0.038
[30,   200] loss: 0.001
[30,   300] loss: 0.031
[30,   400] loss: 0.001
[30] Avg Loss: 0.011
No improvement. Patience: 4/10
[31,   100] loss: 0.017
[31,   200] loss: 0.013
[31,   300] loss: 0.029
[31,   400] loss: 0.032
[31] Avg Loss: 0.010
No improvement. Patience: 5/10
[32,   100] loss: 0.002
[32,   200] loss: 0.000
[32,   300] loss: 0.003
[32,   400] loss: 0.001
[32] Avg Loss: 0.009
No improvement. Patience: 6/10
[33,   100] loss: 0.009
[33,   200] loss: 0.018
[33,   300] loss: 0.001
[33,   400] loss: 0.007
[33] Avg Loss: 0.010
No improvement. Patience: 7/10
[34,   100] loss: 0.001
[34,   200] loss: 0.001
[34,   300] loss: 0.001
[34,   400] loss: 0.011
[34] Avg Loss: 0.010
No improvement. Patience: 8/10
[35,   100] loss: 0.004
[35,   200] loss: 0.005
[35,   300] loss: 0.009
[35,   400] loss: 0.010
[35] Avg Loss: 0.011
No improvement. Patience: 9/10
[36,   100] loss: 0.001
[36,   200] loss: 0.004
[36,   300] loss: 0.013
[36,   400] loss: 0.007
[36] Avg Loss: 0.008
[37,   100] loss: 0.008
[37,   200] loss: 0.003
[37,   300] loss: 0.007
[37,   400] loss: 0.002
[37] Avg Loss: 0.010
No improvement. Patience: 1/10
[38,   100] loss: 0.002
[38,   200] loss: 0.002
[38,   300] loss: 0.011
[38,   400] loss: 0.004
[38] Avg Loss: 0.009
No improvement. Patience: 2/10
[39,   100] loss: 0.006
[39,   200] loss: 0.003
[39,   300] loss: 0.002
[39,   400] loss: 0.001
[39] Avg Loss: 0.008
No improvement. Patience: 3/10
[40,   100] loss: 0.000
[40,   200] loss: 0.012
[40,   300] loss: 0.011
[40,   400] loss: 0.001
[40] Avg Loss: 0.009
No improvement. Patience: 4/10
[41,   100] loss: 0.010
[41,   200] loss: 0.008
[41,   300] loss: 0.006
[41,   400] loss: 0.002
[41] Avg Loss: 0.008
No improvement. Patience: 5/10
[42,   100] loss: 0.005
[42,   200] loss: 0.003
[42,   300] loss: 0.014
[42,   400] loss: 0.005
[42] Avg Loss: 0.010
No improvement. Patience: 6/10
[43,   100] loss: 0.010
[43,   200] loss: 0.000
[43,   300] loss: 0.012
[43,   400] loss: 0.002
[43] Avg Loss: 0.008
No improvement. Patience: 7/10
[44,   100] loss: 0.001
[44,   200] loss: 0.004
[44,   300] loss: 0.035
[44,   400] loss: 0.000
[44] Avg Loss: 0.011
No improvement. Patience: 8/10
[45,   100] loss: 0.002
[45,   200] loss: 0.014
[45,   300] loss: 0.010
[45,   400] loss: 0.014
[45] Avg Loss: 0.010
No improvement. Patience: 9/10
[46,   100] loss: 0.001
[46,   200] loss: 0.076
[46,   300] loss: 0.001
[46,   400] loss: 0.004
[46] Avg Loss: 0.009
No improvement. Patience: 10/10
Early stopping triggered.
✅ Training completed in 27.72 minutes.
Accuracy on test images: 99.57%
Accuracy of 0    : 99.59%
Accuracy of 1    : 99.91%
Accuracy of 2    : 99.71%
Accuracy of 3    : 99.80%
Accuracy of 4    : 99.49%
Accuracy of 5    : 99.33%
Accuracy of 6    : 99.37%
Accuracy of 7    : 99.22%
Accuracy of 8    : 99.90%
Accuracy of 9    : 99.31%

Process finished with exit code 0

```



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202507261936331.png" alt="22485e1e277b7dfea954fe0cd8a1af4f" style="zoom:50%;" />



**Q: net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) ?**

> 是的，`net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)`这一行代码是用于修改 ResNet18 的输入层，使其适应 MNIST 数据集。
>
> **解释：**
>
> - **MNIST 图像** 是灰度图（单通道），所以输入图像的通道数是 1，而原始的 ResNet18 模型是为 RGB 图像设计的，期望输入的通道数为 3。
> - **`nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)`** 这一行就是修改原来 `ResNet18` 中第一个卷积层（`conv1`）的定义，以使其能接收单通道的灰度图像（`1` 通道）。
>   - **`1`**: 输入图像的通道数（即 MNIST 图像的灰度通道数）。
>   - **`64`**: 输出的卷积通道数（ResNet18 中通常是 64）。
>   - **`kernel_size=(7, 7)`**: 卷积核的大小为 7x7。这个值与原始 ResNet18 中的设置一致。
>   - **`stride=(2, 2)`**: 卷积的步长为 2，这意味着每次卷积后，图像尺寸会减少一半。
>   - **`padding=(3, 3)`**: 填充为 3，保持输入图像的尺寸在卷积后不至于变化太大（确保卷积后输出的空间维度适当）。
>   - **`bias=False`**: 通常在深度网络中，如果使用了批量归一化（BatchNorm）等层，卷积层可以去掉偏置项。
>
> **为什么要做这一步：**
>
> - **输入层适配**: MNIST 图像是单通道的（灰度图），所以需要将 `ResNet18` 的输入层 `conv1` 的输入通道数由 3 调整为 1。否则，如果保持原样，网络无法正确处理单通道输入图像。
> - **卷积核和步长选择**: `kernel_size`, `stride`, `padding` 都是保持与原始 ResNet18 模型一致的超参数，目的是尽量保持模型结构不变，从而确保预期的效果。
>
> **总结：**
>
> 这一修改是必须的，因为 MNIST 是灰度图像，而 ResNet18 是为 RGB 图像设计的，必须修改输入通道数才能正确处理 MNIST 数据集。如果你用其他数据集（比如 CIFAR-10 或更大尺寸的图像），这部分的修改就不需要。



