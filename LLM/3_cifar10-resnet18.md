# PyTorch 实例 - CIFAR-10 图像分类项目

Updated 1123 GMT+8 Nov 28 2025

2025 summer, Complied by Hongfei Yan



用到的数据集是 The CIFAR-10 dataset，https://www.cs.toronto.edu/~kriz/cifar.html

> The CIFAR-10 and CIFAR-100 datasets are labeled subsets of the [80 million tiny images](http://people.csail.mit.edu/torralba/tinyimages/) dataset. CIFAR-10 and CIFAR-100 were created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
>
> ## The CIFAR-10 dataset
>
> The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
>
> The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.
>
> Here are the classes in the dataset, as well as 10 random images from each:
>
> | airplane   | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/airplane10.png) |
> | ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
> | automobile | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/automobile10.png) |
> | bird       | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/bird10.png) |
> | cat        | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/cat10.png) |
> | deer       | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/deer10.png) |
> | dog        | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog10.png) |
> | frog       | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/frog10.png) |
> | horse      | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/horse10.png) |
> | ship       | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/ship10.png) |
> | truck      | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck1.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck2.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck3.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck4.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck5.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck6.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck7.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck8.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck9.png) | ![img](https://www.cs.toronto.edu/~kriz/cifar-10-sample/truck10.png) |
>
> The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.
>
> ### Download
>
> If you're going to use this dataset, please cite the tech report at the bottom of this page. 
>
> | Version                                                      | Size   | md5sum                           |
> | ------------------------------------------------------------ | ------ | -------------------------------- |
> | [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) | 163 MB | c58f30108f718f92721af3b95e74349a |
> | [CIFAR-10 Matlab version](https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz) | 175 MB | 70270af85842c9e89bb428ec9976c926 |
> | [CIFAR-10 binary version (suitable for C programs)](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) | 162 MB | c32a1d4ab5d03f1284b67883e8d87530 |
>
> ### Baseline results
>
> You can find some baseline replicable results on this dataset [on the project page for cuda-convnet](http://code.google.com/p/cuda-convnet/). These results were obtained with a convolutional neural network. Briefly, they are 18% test error without data augmentation and 11% with. Additionally, [Jasper Snoek](http://www.cs.toronto.edu/~jasper/) has a [new paper](http://hips.seas.harvard.edu/content/practical-bayesian-optimization-machine-learning-algorithms) in which he used Bayesian hyperparameter optimization to find nice settings of the weight decay and other hyperparameters, which allowed him to obtain a test error rate of 15% (without data augmentation) using the architecture of the net that got 18%.



本意是运行 https://www.runoob.com/pytorch/pytorch-image-classification.html

但是准确率没有达到baseline的82%，所以做了改进。

## 用ResNet18结构，重新训练，做了数据增强

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
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 色彩调整
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 2. 设置设备和模型
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # 加载预定义的 ResNet18 并修改输出层
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 10)  # CIFAR10 10 类
    net.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # 3. 训练过程
    best_loss = float('inf')
    patience = 10 # 提高耐心
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
    torch.save(net.state_dict(), './resnet18_cifar10_data_augument.pth')

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
            ax.imshow(img)
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



### 在Mac Studio（Apple M1 Ultra, 64GB）运行

结果如下：

```
/Users/hfyan/miniconda3/bin/python /Users/hfyan/Desktop/LLMs-from-scratch-main/runoob/pytorch-image-classification/image_classification-ResNet18-RandomCropFlipLR_Cosine.py 
Using device: mps
Starting training with early stopping...
[1,   100] loss: 1.752
[1,   200] loss: 1.675
[1,   300] loss: 1.654
[1] Avg Loss: 1.806
[2,   100] loss: 1.497
[2,   200] loss: 1.459
[2,   300] loss: 1.453
[2] Avg Loss: 1.520
[3,   100] loss: 1.534
[3,   200] loss: 1.383
[3,   300] loss: 1.167
[3] Avg Loss: 1.372
[4,   100] loss: 1.390
[4,   200] loss: 1.221
[4,   300] loss: 1.238
[4] Avg Loss: 1.244
[5,   100] loss: 1.089
[5,   200] loss: 1.020
[5,   300] loss: 1.133
[5] Avg Loss: 1.159
......
No improvement. Patience: 1/10
[192,   100] loss: 0.187
[192,   200] loss: 0.293
[192,   300] loss: 0.356
[192] Avg Loss: 0.302
[193,   100] loss: 0.223
[193,   200] loss: 0.348
[193,   300] loss: 0.309
[193] Avg Loss: 0.301
[194,   100] loss: 0.303
[194,   200] loss: 0.219
[194,   300] loss: 0.280
[194] Avg Loss: 0.304
No improvement. Patience: 1/10
[195,   100] loss: 0.279
[195,   200] loss: 0.296
[195,   300] loss: 0.313
[195] Avg Loss: 0.296
[196,   100] loss: 0.254
[196,   200] loss: 0.385
[196,   300] loss: 0.280
[196] Avg Loss: 0.300
No improvement. Patience: 1/10
[197,   100] loss: 0.216
[197,   200] loss: 0.298
[197,   300] loss: 0.290
[197] Avg Loss: 0.298
No improvement. Patience: 2/10
[198,   100] loss: 0.267
[198,   200] loss: 0.218
[198,   300] loss: 0.367
[198] Avg Loss: 0.290
[199,   100] loss: 0.270
[199,   200] loss: 0.240
[199,   300] loss: 0.351
[199] Avg Loss: 0.301
No improvement. Patience: 1/10
[200,   100] loss: 0.251
[200,   200] loss: 0.227
[200,   300] loss: 0.302
[200] Avg Loss: 0.299
No improvement. Patience: 2/10
[201,   100] loss: 0.348
[201,   200] loss: 0.301
[201,   300] loss: 0.193
[201] Avg Loss: 0.299
No improvement. Patience: 3/10
[202,   100] loss: 0.313
[202,   200] loss: 0.329
[202,   300] loss: 0.305
[202] Avg Loss: 0.295
No improvement. Patience: 4/10
[203,   100] loss: 0.266
[203,   200] loss: 0.254
[203,   300] loss: 0.307
[203] Avg Loss: 0.294
No improvement. Patience: 5/10
[204,   100] loss: 0.372
[204,   200] loss: 0.295
[204,   300] loss: 0.348
[204] Avg Loss: 0.300
No improvement. Patience: 6/10
[205,   100] loss: 0.392
[205,   200] loss: 0.353
[205,   300] loss: 0.306
[205] Avg Loss: 0.296
No improvement. Patience: 7/10
[206,   100] loss: 0.262
[206,   200] loss: 0.213
[206,   300] loss: 0.396
[206] Avg Loss: 0.293
No improvement. Patience: 8/10
[207,   100] loss: 0.293
[207,   200] loss: 0.204
[207,   300] loss: 0.337
[207] Avg Loss: 0.291
No improvement. Patience: 9/10
[208,   100] loss: 0.413
[208,   200] loss: 0.294
[208,   300] loss: 0.315
[208] Avg Loss: 0.295
No improvement. Patience: 10/10
Early stopping triggered.
✅ Training completed in 79.91 minutes.
Accuracy on test images: 83.57%
Accuracy of plane: 83.70%
Accuracy of car  : 92.20%
Accuracy of bird : 78.70%
Accuracy of cat  : 60.40%
Accuracy of deer : 79.30%
Accuracy of dog  : 77.40%
Accuracy of frog : 90.30%
Accuracy of horse: 92.50%
Accuracy of ship : 88.50%
Accuracy of truck: 92.70%

Process finished with exit code 0
```



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202507280020181.jpg" alt="d561be986280572516ac1023e9ff715c" style="zoom:50%;" />



### 在Clab云虚拟机（内存32GB，虚拟CPU32）运行

结果如下：

```
/home/rocky/AI_literacy/.venv/bin/python /home/rocky/AI_literacy/CIFAR-10_nn.py 
100%|██████████| 170M/170M [00:15<00:00, 11.1MB/s]
/usr/lib64/python3.9/tarfile.py:2288: RuntimeWarning: The default behavior of tarfile extraction has been changed to disallow common exploits (including CVE-2007-4559). By default, absolute/parent paths are disallowed and some mode bits are cleared. See https://access.redhat.com/articles/7004769 for more details.
  warnings.warn(
Using device: cpu
Starting training with early stopping...
/home/rocky/AI_literacy/.venv/lib64/python3.9/site-packages/torch/utils/data/dataloader.py:666: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  warnings.warn(warn_msg)
[1,   100] loss: 1.893
[1,   200] loss: 1.658
[1,   300] loss: 1.698
[1] Avg Loss: 1.804
[2,   100] loss: 1.399
[2,   200] loss: 1.580
[2,   300] loss: 1.407
[2] Avg Loss: 1.517
[3,   100] loss: 1.346
[3,   200] loss: 1.492
[3,   300] loss: 1.097
[3] Avg Loss: 1.373
[4,   100] loss: 1.309
[4,   200] loss: 1.182
[4,   300] loss: 1.187
[4] Avg Loss: 1.244
[5,   100] loss: 1.104
[5,   200] loss: 1.084
[5,   300] loss: 1.081
[5] Avg Loss: 1.151
......
[267,   100] loss: 0.278
[267,   200] loss: 0.226
[267,   300] loss: 0.188
[267] Avg Loss: 0.249
[268,   100] loss: 0.218
[268,   200] loss: 0.220
[268,   300] loss: 0.206
[268] Avg Loss: 0.256
No improvement. Patience: 1/10
[269,   100] loss: 0.200
[269,   200] loss: 0.175
[269,   300] loss: 0.298
[269] Avg Loss: 0.256
No improvement. Patience: 2/10
[270,   100] loss: 0.325
[270,   200] loss: 0.265
[270,   300] loss: 0.319
[270] Avg Loss: 0.258
No improvement. Patience: 3/10
[271,   100] loss: 0.316
[271,   200] loss: 0.139
[271,   300] loss: 0.263
[271] Avg Loss: 0.251
No improvement. Patience: 4/10
[272,   100] loss: 0.260
[272,   200] loss: 0.184
[272,   300] loss: 0.141
[272] Avg Loss: 0.256
No improvement. Patience: 5/10
[273,   100] loss: 0.263
[273,   200] loss: 0.327
[273,   300] loss: 0.246
[273] Avg Loss: 0.250
No improvement. Patience: 6/10
[274,   100] loss: 0.193
[274,   200] loss: 0.221
[274,   300] loss: 0.227
[274] Avg Loss: 0.250
No improvement. Patience: 7/10
[275,   100] loss: 0.185
[275,   200] loss: 0.354
[275,   300] loss: 0.254
[275] Avg Loss: 0.258
No improvement. Patience: 8/10
[276,   100] loss: 0.354
[276,   200] loss: 0.326
[276,   300] loss: 0.246
[276] Avg Loss: 0.250
No improvement. Patience: 9/10
[277,   100] loss: 0.183
[277,   200] loss: 0.383
[277,   300] loss: 0.295
[277] Avg Loss: 0.256
No improvement. Patience: 10/10
Early stopping triggered.
✅ Training completed in 636.54 minutes.
Accuracy on test images: 83.67%
Accuracy of plane: 84.70%
Accuracy of car  : 91.90%
Accuracy of bird : 84.00%
Accuracy of cat  : 65.90%
Accuracy of deer : 81.10%
Accuracy of dog  : 77.60%
Accuracy of frog : 89.40%
Accuracy of horse: 85.00%
Accuracy of ship : 91.70%
Accuracy of truck: 85.40%

Process finished with exit code 0

```



![f44849630ac92c19ff8f0b2e922801ab](https://raw.githubusercontent.com/GMyhf/img/main/img/f44849630ac92c19ff8f0b2e922801ab.png)



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/b353056ebdc1085718fc43ab6d2a1316.png" alt="b353056ebdc1085718fc43ab6d2a1316" style="zoom: 50%;" />



### Q: 之前版本 `image_classification-ResNet18.py` 没有baseline好？

> 你现在的结果是：
>
> - **Test Accuracy:** 76.77%
> - **Test Error Rate:** 100% - 76.77% = **23.23%**
>
> 而你引用的 **baseline（来自 cuda-convnet 项目）**提供的是：
>
> | 方法                       | Test Error (%) | Test Accuracy (%) |
> | -------------------------- | -------------- | ----------------- |
> | Baseline CNN（无数据增强） | **18.0%**      | **82.0%**         |
> | Bayesian 优化后（无增强）  | **15.0%**      | **85.0%**         |
> | CNN（带数据增强）          | **11.0%**      | **89.0%**         |
>
> ------
>
> 🆚 对比分析：
>
> | 指标              | 你的模型（ResNet18） | baseline 最好  | 差距                |
> | ----------------- | -------------------- | -------------- | ------------------- |
> | Test Accuracy (%) | **76.77%**           | **89.0%**      | ↓ ~12.2%            |
> | 网络结构          | ResNet18             | 自定义 CNN     | ResNet18 是更强模型 |
> | 数据增强          | 否                   | 是             | ✅ 可提升            |
> | 超参调优          | 默认                 | 有（Bayesian） | ✅ 可提升            |
>
> ------
>
> ## 🧠 结论：
>
> 你当前的结果是 **很不错的起点**，考虑到：
>
> - 使用的是标准 `ResNet18`，且没有做任何 **数据增强**；
> - 没有特别调 `lr`、`weight_decay`、`batch_size` 等超参数；
> - 没有用额外 trick，如 CosineLR、Mixup、Label Smoothing 等；
> - 用的是 **MPS 后端**（速度快，但可能稳定性和 cuda 还差点）；
>
> **因此，你的结果基本和“未调优的 ResNet18 基线”持平。**
>
> ------
>
> ✅ 如何追平甚至超越 baseline？
>
> 你的 ResNet18 很强，但你可以继续提升准确率：
>
> | 方法                    | 说明                                        |
> | ----------------------- | ------------------------------------------- |
> | ✅ 加数据增强            | `transforms.RandomCrop + HorizontalFlip`    |
> | ✅ 调优学习率/调度器     | 使用 `CosineAnnealingLR`、warmup 等         |
> | ✅ 引入正则化技巧        | 加 weight_decay，dropout 或 label smoothing |
> | ✅ 换更高效模型（可选）  | MobileNetV2 或 EfficientNet                 |
> | ✅ 加入验证集 early stop | 防止过拟合，提升泛化                        |
> | ✅ 多轮训练+冻结微调     | 微调 ImageNet 权重部分                      |
>
> ------
>
> 是否需要我帮你**加数据增强版本**？几行代码就可以在 `transform` 中添加，让 ResNet18 发挥更大潜力。
>
> ```python
> transform_train = transforms.Compose([
>     transforms.RandomCrop(32, padding=4),
>     transforms.RandomHorizontalFlip(),
>     transforms.ToTensor(),
>     transforms.Normalize(mean, std),
> ])
> ```
>
> 是否需要我直接给出对比完整训练脚本？可以一键替换你当前版本。



### Q: transforms.Resize(224), # ResNet 输入要求至少224x224，现在不需要了？

> 你问得非常好，这涉及到 **ResNet18 与 CIFAR-10 输入图像尺寸的兼容性问题**：
>
> ------
>
> 🔍 ResNet18 的原始设计输入尺寸
>
> - 原始 ImageNet 预训练的 ResNet18 输入尺寸是 **224×224**。
>
> - 因此常见用法中通常包含：
>
>   ```python
>   transforms.Resize(256),
>   transforms.CenterCrop(224),
>   ```
>
> ------
>
> ✅ 在 CIFAR-10 上使用 ResNet18 不需要 `Resize(224)` 的情况
>
> 因为你在训练中使用的是：
>
> ```python
> models.resnet18(weights=None)
> ```
>
> 这表示：
>
> - **没有加载预训练权重**
> - 网络从头开始训练
> - 所以你可以**自由地替换第一层卷积**或直接用默认的，**不需要放大到 224×224**
>
> 而且你的数据（CIFAR-10）是 `32x32`，所以 torchvision 中常见的处理方式是：
>
> - **保留原始尺寸**
> - 只使用：`RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` 就足够了
>
> ------
>
> 🧠 如果你想使用预训练权重
>
> 如果你打算使用：
>
> ```python
> models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
> ```
>
> 则你必须：
>
> 1. **调整输入图像大小到 ≥224x224**：
>
>    ```python
>    transforms.Resize(256),
>    transforms.CenterCrop(224),
>    ```
>
> 2. 或者修改模型第一层：
>
>    将：
>
>    ```python
>    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
>    model.maxpool = nn.Identity()
>    ```
>
> ------
>
> ✅ 结论
>
> | 用法                                           | 是否需要 Resize(224) | 说明                |
> | ---------------------------------------------- | -------------------- | ------------------- |
> | `weights=None`（从头训练）                     | ❌ 不需要             | 可直接用 32×32 图像 |
> | `weights=ResNet18_Weights.DEFAULT`（用预训练） | ✅ 需要               | 需Resize到224x224   |
>
> ------
>
> 是否需要我帮你写一份使用 **预训练ResNet18** + 自动调整尺寸的完整代码版本？



### Q:  **精度 (Accuracy)** 仍然比 **baseline (82%)** 稍低，可能的原因和改进建议：

> 你的训练结果已经相当不错，但 **精度 (Accuracy)** 仍然比 **baseline (82%)** 稍低，以下是一些可能的原因和改进建议：
>
> ------
>
> ✅ 可能的原因：
>
> 1. **没有数据增强**：
>    - 你的当前数据增强设置中仅包含了 `RandomCrop` 和 `RandomHorizontalFlip`。这对 ResNet18 来说可能不够，尤其是在 CIFAR-10 这种小规模数据集上。Baseline 的 82% 准确率是使用了 **更多的数据增强**。
> 2. **学习率（Learning Rate）设置问题**：
>    - 你的学习率设置是 `lr=0.1`，这可能稍微有些高，导致网络在训练初期就发生了震荡，无法有效收敛。
>    - 你可以尝试 **逐步降低学习率**，比如使用学习率调度器（`lr_scheduler`）或者 **更低的初始学习率**（如 0.01 或 0.001）。
> 3. **训练轮数（Epochs）不足**：
>    - 虽然你设置了 **100 个 epoch**，但根据训练过程来看，模型在 15 分钟内只训练了大约 **24 个 epoch**。这意味着可能还需要更多的训练时间来进一步降低损失。
> 4. **早期停止（Early Stopping）**：
>    - 早期停止可能在损失尚未完全降低时就停止了训练。你可能会希望 **放宽早期停止的条件**，让训练继续更多 epoch，直到精度达到更高水平。
>
> ------
>
> ✅ 改进建议：
>
> 1. **增加数据增强**：
>
> - 为了提高模型的泛化能力，建议添加更多的数据增强操作。可以加入：
>   - `RandomRotation(10)`：随机旋转图像
>   - `ColorJitter`：调整图像的亮度、对比度等
>   - `RandomAffine`：随机仿射变换
>
> 修改后的数据增强代码：
>
> ```python
> transform_train = transforms.Compose([
>     transforms.RandomCrop(32, padding=4),
>     transforms.RandomHorizontalFlip(),
>     transforms.RandomRotation(10),  # 随机旋转
>     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # 色彩调整
>     transforms.ToTensor(),
>     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
> ])
> ```
>
> 2. **调整学习率（Learning Rate）**：
>
> - **逐步降低学习率**：建议在训练时 **使用动态学习率**，可以尝试更低的初始学习率（如 0.01），并设置一个学习率调度器（`lr_scheduler`）来逐步降低学习率。
>
> ```python
> optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
> scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 每20个epoch将学习率降低50%
> ```
>
> 3. **放宽 Early Stopping 的条件**：
>
> - 提高 `patience`（例如从 5 改为 10），或者将停止条件设为更宽松的标准，比如 **增加验证集准确率的持续提升**。
>
> ```python
> patience = 10  # 提高耐心
> ```
>
> 4. **增加训练轮数（Epochs）**：
>
> - 尝试增加训练轮数（例如 200-300），即使使用早期停止，增加训练时间仍然有助于模型的收敛。
>
> 5. **检查数据集的划分**：
>
> - 确保你在训练和验证集上没有数据泄漏，并且确保 **数据加载时没有随机性** 影响模型评估（比如使用固定的随机种子）。
>
> 6. **使用数据增强（如 MixUp 或 Cutout）**：
>
> - 你可以进一步尝试 **MixUp**（通过线性插值进行数据增强）或 **Cutout**（随机遮挡一部分图像）等方法，已知在 CIFAR-10 上这些方法有助于提高准确率。
>
> ------
>
> ✅ 小结：
>
> 你的模型训练已经非常接近 baseline 的结果，经过数据增强、适当的学习率调节、更多训练轮次等优化后，应该能进一步提升准确率。你还可以尝试 **使用预训练模型**，或者 **迁移学习**，这些方法能够让模型更快收敛，特别是在较小数据集上。
>
> ------
>
> 是否需要我进一步帮助你集成 **学习率调度器**、**更多数据增强**、**验证集评估**等功能，或者直接给出一个全新的版本进行改进？



> 🔧 新改进点：
>
> 1. **数据增强**：
>    - 增加了 `RandomRotation` 和 `ColorJitter`，可以进一步增强模型的泛化能力。
> 2. **学习率调度器**：
>    - 使用 `CosineAnnealingLR` 来动态调整学习率，帮助模型在训练后期更平滑地收敛。
> 3. **早期停止**：
>    - 调整了 `patience` 为 10 轮，允许模型有更多时间来收敛，防止过早停止。
> 4. **训练时间统计**：
>    - 输出了总训练时间，方便了解训练消耗。
> 5. **模型保存**：
>    - 保存了训练好的模型，方便以后加载和推理。
>
> ------
>
> 下一步：
>
> 你可以运行这个脚本，看是否能 **超越 baseline 的 82%** 准确率。如果结果不理想，可以进一步调整超参数、添加更多的技巧（如 MixUp、Cutout）。



### **Q: 解读代码**

```
解读：    # 每类准确率
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
```

> 这段代码用于计算 **每一类的准确率**，具体步骤如下：
>
> ------
>
> 🧩 代码解读
>
> ```python
> # 创建两个列表，分别用于存储每一类的正确预测数和总样本数
> class_correct = list(0. for _ in range(10))  # 存储每个类别正确预测的次数
> class_total = list(0. for _ in range(10))    # 存储每个类别的样本数
> ```
>
> - `class_correct[i]` 用于记录类别 `i` 的正确预测次数。
> - `class_total[i]` 用于记录类别 `i` 的总样本数。
>
> `10` 是 CIFAR-10 数据集中的类别数量（10 类）。
>
> ------
>
> ```python
> # 不计算梯度，加速推理
> with torch.no_grad():
>     for data in testloader:
>         images, labels = data
>         images, labels = images.to(device), labels.to(device)
>         outputs = net(images)
>         _, predicted = torch.max(outputs.data, 1)
> ```
>
> - 使用 `torch.no_grad()` 来 **关闭梯度计算**，这样在推理时节省内存和计算。
> - `testloader` 是加载测试集的数据迭代器，`images` 是输入图像，`labels` 是对应的标签。
> - `outputs = net(images)`：用训练好的模型 `net` 对图像进行推理，得到每个类别的预测概率。
> - `_, predicted = torch.max(outputs.data, 1)`：通过 `torch.max` 获取 **每张图像的最大概率类别**（即模型的预测标签）。
>
> ------
>
> ```python
> # 计算每个类别的正确预测数量和总样本数量
> c = (predicted == labels).squeeze()
> for i in range(len(labels)):
>     class_correct[labels[i]] += c[i].item()  # 记录正确预测的数量
>     class_total[labels[i]] += 1              # 记录每个类别的样本数量
> ```
>
> - `(predicted == labels)`：检查预测标签与实际标签是否一致，返回布尔值（True 或 False）。`True` 代表预测正确，`False` 代表预测错误。
> - `c[i].item()`：转换布尔值为 1 或 0，表示该样本是否被正确分类。
> - `class_correct[labels[i]] += c[i].item()`：将每个正确预测的样本数量累加到对应类别的 `class_correct` 中。
> - `class_total[labels[i]] += 1`：每遇到一个样本，就累加到该类别的总样本数 `class_total`。
>
> ------
>
> 🧠 总结：
>
> 这段代码的目的是通过遍历测试集，计算每个类别的**正确预测数**和**总样本数**。最后，你可以通过以下方式计算 **每个类别的准确率**：
>
> ```python
> for i in range(10):
>     accuracy = 100 * class_correct[i] / class_total[i]
>     print(f'Accuracy of {classes[i]:5s}: {accuracy:.2f}%')
> ```
>
> `classes[i]` 是 CIFAR-10 中的类别名称（如 `plane`, `car` 等）。
>
> ------
>
> 🔄 关键点：
>
> 1. **`torch.max(outputs.data, 1)`**: 获取每个样本的预测类别。
> 2. **`class_correct[labels[i]] += c[i].item()`**: 累加正确预测的数量。
> 3. **`class_total[labels[i]] += 1`**: 累加每个类别的样本数量。
>
> 通过这个过程，你能得到每个类别的准确率，有助于了解模型在哪些类别上表现较好或较差。



### **Q: 具体示例呢？是一个一个图像检测吧**

> 是的，**这段代码确实是在一个个图像上检测预测是否正确**，并按类别统计每类的预测准确率。下面我们结合一个**具体示例**来演示这段代码在实际运行中做了什么。
>
> ------
>
> 🎯 假设场景
>
> 我们假设测试集中某个 batch（小批量）里有 5 张图像，真实标签为：
>
> ```python
> labels = [1, 3, 3, 5, 1]  # 实际类别编号
> ```
>
> 模型预测结果为：
>
> ```python
> predicted = [1, 2, 3, 5, 0]  # 模型预测的类别编号
> ```
>
> 那么对比：
>
> | 图片编号 | 实际标签（labels[i]） | 模型预测（predicted[i]） | 是否正确 (==) |
> | -------- | --------------------- | ------------------------ | ------------- |
> | 0        | 1                     | 1                        | ✅ True        |
> | 1        | 3                     | 2                        | ❌ False       |
> | 2        | 3                     | 3                        | ✅ True        |
> | 3        | 5                     | 5                        | ✅ True        |
> | 4        | 1                     | 0                        | ❌ False       |
>
> 生成布尔比较：
>
> ```python
> c = [True, False, True, True, False]
> ```
>
> ------
>
> 📊 累加到 `class_correct` 和 `class_total`
>
> 初始化：
>
> ```python
> class_correct = [0.0] * 10
> class_total = [0.0] * 10
> ```
>
> 遍历每个样本：
>
> ```python
> for i in range(5):
>     class_correct[labels[i]] += c[i].item()
>     class_total[labels[i]] += 1
> ```
>
> 每一步的变化如下：
>
> | i    | labels[i] | c[i]  | class_correct         | class_total           |
> | ---- | --------- | ----- | --------------------- | --------------------- |
> | 0    | 1         | True  | [0, 1, 0, 0, 0…]      | [0, 1, 0, 0, 0…]      |
> | 1    | 3         | False | [0, 1, 0, 0, 0…]      | [0, 1, 0, 1, 0…]      |
> | 2    | 3         | True  | [0, 1, 0, 1, 0…]      | [0, 1, 0, 2, 0…]      |
> | 3    | 5         | True  | [0, 1, 0, 1, 0, 1, …] | [0, 1, 0, 2, 0, 1, …] |
> | 4    | 1         | False | [0, 1, 0, 1, 0, 1, …] | [0, 2, 0, 2, 0, 1, …] |
>
> ------
>
> ✅ 准确率结果（最终）
>
> ```python
> Accuracy of class 1: 1 / 2 = 50.00%
> Accuracy of class 3: 1 / 2 = 50.00%
> Accuracy of class 5: 1 / 1 = 100.00%
> ```
>
> 其他类别没有出现，准确率为 0%。
>
> ------
>
> 🔁 在真实代码中：
>
> 这段逻辑在 **整个 testloader（测试集）** 上运行，每次处理一个 batch（如 100 张图像），最后统计出完整的 per-class accuracy。
>
> ------
>
> ✅ 总结：
>
> - 是一个图像一个图像在比对 `预测是否正确`。
> - 累加每个类别的正确预测次数和样本总数。
> - 最后计算 `每类准确率 = 正确预测数 / 样本总数`。
>
> 你可以在训练完后非常直观地看到：**模型在哪些类别上表现好（准确率高），哪些类别不好（比如猫通常较难分类）**。这对分析模型瓶颈很有帮助。



### **Q: _, predicted = torch.max(outputs.data, 1) ？**

> 这行代码：
>
> ```python
> _, predicted = torch.max(outputs.data, 1)
> ```
>
> 是 PyTorch 中的常见写法，用于从模型的输出中**获取每张图片预测的类别编号**（也就是最大得分对应的下标）。下面是详细解读。
>
> ------
>
> 📘 背景：模型输出是什么？
>
> 在分类任务中，比如 CIFAR-10，模型对每张图像的输出是一个 **长度为10的向量**，表示每个类别的得分（logits）。例如对一张图片，输出可能是：
>
> ```python
> [2.1, 1.5, -0.3, 4.2, 0.8, 3.3, 1.1, 0.0, -1.2, 2.0]
> ```
>
> 这表示对类别 0～9 的“置信度”或“匹配程度”，越大越可能是正确类别。
>
> ------
>
> 🧠 `torch.max(outputs.data, 1)` 解释
>
> ```python
> torch.max(tensor, dim)
> ```
>
> 这个函数会在指定的维度 `dim` 上寻找最大值，并返回两个结果：
>
> 1. **最大值的数值**
> 2. **最大值的下标（索引）**
>
> 例如：
>
> ```python
> outputs.data = tensor([
>   [2.1, 1.5, -0.3, 4.2, 0.8, 3.3, 1.1, 0.0, -1.2, 2.0],
>   [0.5, 2.8, 1.0, 0.7, 0.1, 0.0, 3.9, 0.2, 0.3, 0.5]
> ])
> ```
>
> 表示 2 张图片的分类输出。
>
> 执行：
>
> ```python
> _, predicted = torch.max(outputs.data, 1)
> ```
>
> 就会得到：
>
> ```python
> predicted = tensor([3, 6])
> ```
>
> 表示第 1 张图像预测为类别 3，第 2 张为类别 6。
>
> ------
>
> 📌 为什么用 `_` 接收第一个返回值？
>
> 因为你只关心“哪个类别是最大”，不关心“最大得分是多少”，所以第一个返回值（得分）就用 `_` 忽略掉了。
>
> ------
>
> ✅ 总结
>
> ```python
> _, predicted = torch.max(outputs.data, 1)
> ```
>
> 含义是：
>
> > 在每张图像的输出向量中，找出最大值所在的下标 `predicted`，作为该图像的预测类别编号。
>
> 这个 `predicted` 就可以和 `labels` 做比较来判断是否预测正确。





## 附录

### ImageNet 图像分类任务中的 Top-5 错误率

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250726164756658.png" alt="image-20250726164756658" style="zoom: 67%;" />



**x轴：模型和年份**

从左到右列出了不同年份的代表性模型：

| 年份 | 模型         | 所属机构/作者            |
| ---- | ------------ | ------------------------ |
| 2011 | XRCE         | 非深度学习方法           |
| 2012 | AlexNet      | Alex Krizhevsky 等人     |
| 2013 | ZFNet        | Zeiler & Fergus          |
| 2014 | VGG          | Visual Geometry Group    |
| 2014 | GoogLeNet    | Google（Inception）      |
| 2015 | ResNet       | Microsoft Research       |
| 2016 | GoogLeNet-v4 | 更深更复杂的 GoogLeNet   |
| ---  | **Human**    | 人类平均表现（对比基线） |

y轴：**Top-5 错误率（%）**

表示模型在 ImageNet 上错误预测的概率（Top-5）。错误率越低，模型性能越好。

从图中可以看到：

- **2011 XRCE：Top-5错误率为 26.0%**
- **2012 AlexNet** 将错误率降低到了 **16.4%**，开启深度学习时代
- **2015 ResNet** 将错误率降至 **3.6%**，首次超过了 **人类**（人类误差为 5.0%）
- **2016 GoogLeNet-v4** 降到仅 **3.1%**

------

🧠 什么是 Top-5 错误率？

在 ImageNet 图像分类中，**每张图像的标签属于 1000 个类别之一**。

- **Top-1 错误率**：预测第一名是否正确。
- **Top-5 错误率**：只要模型预测出的前 5 个类别中包含正确答案，就算 **预测成功**。

👉 例如：
如果模型对某张图片的输出是：

```text
1. dog
2. cat
3. horse
4. frog
5. deer
```

而真实类别是 **frog**，则：

- **Top-1** 是错的（第一名是 dog）
- **Top-5** 是对的（frog 在前五名中）

------

✅ 总结

这张图展示了深度学习模型在 ImageNet 上的飞跃式进展，尤其是：

- AlexNet（2012）首次使用深度 CNN，大幅改进结果；
- ResNet（2015）提出残差网络，首次超过人类；
- GoogLeNet 和 VGG 也都是代表性的经典 CNN 架构。

**Top-5 错误率低，意味着模型更稳定、鲁棒性更强**。在工业界、比赛中经常作为重要评价指标。

如果你想，我可以进一步解释每个模型架构的改进要点。
