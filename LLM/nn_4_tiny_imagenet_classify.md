# PyTorch 实例 -Tiny ImageNet图像分类项目

Updated 1947 GMT+8 Jul 26 2025

2025 spring, Complied by Hongfei Yan



现在用的chatgpt架构都是transformer模型, Transformer包括编码器和解码器两大部分，其中编码器负责将输入序列映射为隐藏表示，而解码器则利用这些隐藏表示来生成输出序列。<mark>transformer看起来是否与ASCII编码相似</mark>。

![image-20250224172229492](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250224172229492.png)



基于 PyTorch 和 torchvision 库的示例代码，我们演示如何加载Tiny ImageNet 数据集、构建基于预训练 ResNet50 模型的神经网络，并进行微调训练实现图像分类。ImageNet 数据集体积较大，我们使用Tiny ImageNet，即是这样，训练时还需要充足的存储和计算资源，实际使用时需要正确设置数据路径、batch size 和其他超参数。

起因是书中提到

![image-20250224172024784](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250224172024784.png)



![image-20250224172055496](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250224172055496.png)

**Q1. ResNet50?**

> ResNet-50是一种深度卷积神经网络，属于ResNet（Residual Network）系列模型之一。它由微软研究院的Kaiming He等人在2015年提出，旨在解决训练非常深的神经网络时遇到的梯度消失或梯度爆炸问题。ResNet通过引入“残差学习单元”（residual learning units 或者 shortcut connections/跳跃连接），使得网络可以更深且更容易训练。
>
> **ResNet-50结构特点**
>
> - **深度**: ResNet-50是一个包含50层的深层网络（包括卷积层、激活函数层、池化层等），因此得名ResNet-50。
> - **残差块(Residual Block)**: 核心概念是残差学习，通过添加直连边（shortcut connections），让输入直接跳过一层或多层传到后面层，这有助于解决深层网络中的退化问题（即随着网络深度增加，训练错误率反而上升的问题）。公式表达为：\[y = F(x, \{W_i\}) + x\]，其中\(x\)和\(y\)分别是层的输入和输出，\(F(x, \{W_i\})\)表示学习到的残差映射。
> - **具体架构**:
>   - 输入图像尺寸通常是224x224。
>   - 网络开始于一个7x7的卷积层，接着是最大池化层。
>   - 接下来是4个主要的残差阶段，每个阶段分别包含了3、4、6和3个残差块（对于ResNet-50），每个残差块内部有两个3x3的卷积层。
>   - 在最后一个残差阶段之后，使用全局平均池化层和一个全连接层进行分类任务，默认设置下用于ImageNet数据集的分类，有1000个输出节点。
>
> **应用场景**
>
> 由于其优秀的性能和相对合理的计算需求，ResNet-50广泛应用于各种计算机视觉任务中，如图像分类、目标检测、人脸识别等。此外，它也常被用作迁移学习的基础模型，在特定领域的小型数据集上进行微调以达到较好的效果。
>
> ResNet的成功推动了深度学习特别是计算机视觉领域的快速发展，后续还出现了更深的版本如ResNet-101和ResNet-152，以及针对效率优化的变体如ResNeXt。



## 1.准备Tiny ImageNet数据集

Tiny ImageNet。它包含 200 个类别，每个类别 500 张训练图片，总数据量大约 500MB，非常适合实验和调试。

下载 `wget http://cs231n.stanford.edu/tiny-imagenet-200.zip`，记237MB。

验证集通常解压后所有图片会在同一个文件夹中，而 ImageFolder 要求每个类别有独立子文件夹。你需要根据官方提供的 验证集标签文件，如 val_annotations.txt，对图片进行分类整理。常见的做法是编写一个脚本，根据文件中的类别信息将图片移动到对应的子文件夹中。

`tinyimagenet.sh`

```sh
#!/bin/bash

# download and unzip dataset
#wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

current="$(pwd)/tiny-imagenet-200"

# training data
cd $current/train
for DIR in $(ls); do
   cd $DIR
   rm *.txt
   mv images/* .
   rm -r images
   cd ..
done

# validation data
cd $current/val
annotate_file="val_annotations.txt"
length=$(cat $annotate_file | wc -l)
for i in $(seq 1 $length); do
    # fetch i th line
    line=$(sed -n ${i}p $annotate_file)
    # get file name and directory name
    file=$(echo $line | cut -f1 -d" " )
    directory=$(echo $line | cut -f2 -d" ")
    mkdir -p $directory
    mv images/$file $directory
done
rm -r images
echo "done"

```



运行`sh tinyimagenet.sh`，数据解压并分类准备好，记472MB。

```
% ls -l
total 5200
drwxrwxr-x    3 hfyan  staff       96 Dec 12  2014 test
drwxrwxr-x  202 hfyan  staff     6464 Dec 12  2014 train
drwxrwxr-x  203 hfyan  staff     6496 Feb 24 11:09 val
-rw-rw-r--    1 hfyan  staff     2000 Feb  9  2015 wnids.txt
-rw-------    1 hfyan  staff  2655750 Feb  9  2015 words.txt
(base) hfyan@HongfeideMac-Studio tiny-imagenet-200 % pwd
/Users/hfyan/data/tiny-imagenet-200

```



## 2.查看本地机器Mac Studio的配置

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202502160845440.png" alt="image-20250216084507879" style="zoom:50%;" />

Apple M1 Ultra 是 Apple 芯片系列中的一员，专为高性能需求设计，特别是在 Mac Studio 等设备中使用。M1 Ultra 的配置包括了中央处理器（CPU）、图形处理器（GPU）以及统一内存架构（Unified Memory Architecture, UMA），其中统一内存可供 CPU、GPU 以及其他组件共享。

**关于 GPU 内存**

在 M1 Ultra 中， 64GB 内存实际上是整个系统共享的统一内存容量，这意味着这64GB内存是由CPU、GPU及其他组件共同使用的，而不是专门分配给GPU的独立内存。 这种设计极大地提高了灵活性和性能表现，尤其是在处理复杂图形任务或多任务处理场景下。 

- **统一内存架构**：Apple 的设计理念是通过统一内存架构来提升性能和效率。这种架构允许 GPU 和 CPU 访问相同的内存池，减少了数据复制的需求，并且可以更灵活地根据需要分配内存资源。

- **M1 Ultra 的 GPU 资源**：M1 Ultra 配备了一个强大的 48 核心 GPU。尽管没有“专用”的 GPU 显存，但其可以从整个 64GB 统一内存中获取所需的工作内存。这对于许多图形密集型应用来说是非常有利的，因为它避免了传统显存与主存之间可能存在的瓶颈。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202502160845741.png" alt="image-20250216084526735" style="zoom:50%;" />



## 3. 训练模型

基于 PyTorch 和 torchvision 库的示例代码，该代码演示了如何加载 ImageNet 数据集、构建基于预训练 ResNet 模型的神经网络，并进行微调训练实现图像分类。

在clab.pku云端虚拟机，内存只有4GB，无GPU，跑不起来。以下是在我本地mac机器运行的。

> “Killed” 通常是操作系统（Linux 内核）出于内存不足（OOM）的原因终止了进程。这可能是由于以下原因引起的：
>
> - **内存不足**：程序在训练过程中占用了过多内存，超出了系统可用内存或交换空间（swap）的限制。
> - **GPU 内存不足**：如果使用 GPU 训练，可能也会出现 GPU 内存溢出的问题。

`tiny_imagenet_resnet50_epoch25.py`

```python
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

# 训练和验证函数
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, device='cpu'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # 每个 epoch 分为训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置为训练模式
            else:
                model.eval()   # 设置为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # 梯度清零

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 仅在训练阶段反向传播与参数更新
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            # MPS 后端不支持 float64 运算。解决方法是使用 float32，即调用 .float()。
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 保存最佳模型参数
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    print('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

def main():
    # 1. 数据预处理与加载
    # 注意：此处假定ImageNet数据集按照train/val文件夹分别存放各类别图片，
    # 且每个类别作为一个子文件夹存在
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),           # 随机裁剪为224×224
            transforms.RandomHorizontalFlip(),           # 随机水平翻转
            transforms.ToTensor(),                         # 转为Tensor
            transforms.Normalize([0.485, 0.456, 0.406],    # ImageNet均值
                                 [0.229, 0.224, 0.225])      # ImageNet标准差
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),                           # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # Tiny ImageNet 数据路径
    data_dir = '/Users/hfyan/data/tiny-imagenet-200'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                data_transforms[x])
                      for x in ['train', 'val']}

    # 设置 num_workers 为 4 以利用多进程数据加载
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                    batch_size=128,    # 可根据实际情况调整
                                                    shuffle=True,
                                                    num_workers=8)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 使用 MPS 作为 GPU 后端（适用于 Apple Silicon）
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("MPS device not available, using CPU")

    #2. 构建模型（使用预训练 ResNet50）
    # 这里我们加载预训练的 ResNet50 模型，并修改最后的全连接层以适应Tiny ImageNet的类别数（200类）
    model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # 学习率调整策略，每7个epoch降低一次学习率
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #3. 训练模型
    num_epochs = 25  # 可根据需要调整epoch数量
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           dataloaders, dataset_sizes, num_epochs=num_epochs, device=device)

    #4. 保存模型，文件名建议为 tiny_imagenet_resnet50_epoch25.pth
    torch.save(model_ft.state_dict(), 'tiny_imagenet_resnet50_epoch25.pth')
    print("Model saved as tiny_imagenet_resnet50_epoch25.pth")

if __name__ == '__main__':
    main()

```

> **说明**
>
> - **数据预处理**
>   使用了 `transforms` 对数据进行了数据增强（如随机裁剪、水平翻转）以及归一化（ImageNet常用的均值和标准差）。数据文件夹需要符合 `ImageFolder` 的要求，每个类别存放在独立的子文件夹中。
> - **模型构建**
>   本示例中采用预训练的 ResNet50 模型，并修改了最后一层全连接层以输出与类别数匹配的概率分布。
> - **训练过程**
>   代码中定义了 `train_model` 函数，对模型进行训练和验证，并在验证集上选取准确率最高的模型参数。学习率调度器用于逐步降低学习率以便更好地收敛。
> - **注意事项**
>   - ImageNet 数据集较大，建议在使用时注意数据加载、内存管理和训练时长。
>   - 如需更深入的模型调优或使用分布式训练，请参考 PyTorch 官方文档和相关资料。
>
> 该示例代码为入门级示例，实际项目中可能需要更多的优化和配置。
>
> 
>
> **主入口保护**：所有涉及多进程或多线程的代码都封装在 `if __name__ == '__main__':` 下，避免 macOS 下的启动问题。



**Q. 为什么每次运行都重新下载resnet50?**

为什么每次运行都重新下载 Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to /Users/hfyan/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth ，下载过了一次了。

> PyTorch 会将预训练模型的权重下载到默认缓存目录（通常是 `~/.cache/torch/hub/checkpoints`），如果每次运行都重新下载，可能有以下几个原因：
>
> 1. **缓存目录权限或环境变量问题**  
>    请检查 `/Users/hfyan/.cache/torch/hub/checkpoints/` 目录是否存在、是否有写入权限以及文件是否完整。如果该目录不可写或文件被删除，PyTorch 就无法使用缓存文件。你可以尝试设置环境变量 `TORCH_HOME`，将缓存目录指定到一个你确定有权限并且持久的目录：
>    ```bash
>    export TORCH_HOME=/Users/hfyan/torch_cache
>    ```
>    然后再运行程序。
>
> 2. **运行环境重置**  
>    如果你每次运行程序时使用的是临时环境（例如某些 IDE 的沙盒模式、容器或虚拟环境在退出后清除缓存），那么每次启动时缓存文件可能都会被清空，从而导致重新下载。
>
> 3. **下载文件校验问题**  
>    PyTorch 在加载权重时会检查文件的完整性。如果文件损坏或校验失败，也会重新下载。
>
> 建议先确认缓存目录中是否已经存在 `resnet50-11ad3fa6.pth` 文件，且该文件大小和校验值是否正确。如果有权限问题或缓存目录被重置，可以通过设置 `TORCH_HOME` 环境变量来指定一个稳定的缓存目录，从而避免每次运行时重新下载。



### 2025/2/24 11:30开始运行，16:00结束

>  File "/Users/hfyan/data/tiny_imagenet_resnet50_epoch25.py", line 47, in train_model
>
>   epoch_acc = running_corrects.double() / dataset_sizes[phase]
>
> TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
>
> (base) hfyan@HongfeideMac-Studio data % python tiny_imagenet_resnet50_epoch25.py 

```
(base) hfyan@HongfeideMac-Studio data % python tiny_imagenet_resnet50_epoch25.py 
Using MPS device for GPU acceleration
Epoch 1/25
----------
train Loss: 5.0366 Acc: 0.0720
val Loss: 4.1348 Acc: 0.2819

Epoch 2/25
----------
train Loss: 3.2563 Acc: 0.3406
val Loss: 1.7006 Acc: 0.6197

Epoch 3/25
----------
train Loss: 2.1834 Acc: 0.5065
val Loss: 1.2068 Acc: 0.7062

Epoch 4/25
----------
train Loss: 1.8635 Acc: 0.5663
val Loss: 1.0010 Acc: 0.7498

Epoch 5/25
----------
train Loss: 1.6788 Acc: 0.6029
val Loss: 0.8927 Acc: 0.7702

Epoch 6/25
----------
train Loss: 1.5723 Acc: 0.6268
val Loss: 0.8407 Acc: 0.7808

Epoch 7/25
----------
train Loss: 1.5044 Acc: 0.6390
val Loss: 0.7990 Acc: 0.7907

Epoch 8/25
----------
train Loss: 1.4324 Acc: 0.6567
val Loss: 0.7788 Acc: 0.7939

Epoch 9/25
----------
train Loss: 1.4212 Acc: 0.6571
val Loss: 0.7701 Acc: 0.7981

Epoch 10/25
----------
train Loss: 1.4054 Acc: 0.6614
val Loss: 0.7669 Acc: 0.7966

Epoch 11/25
----------
train Loss: 1.4035 Acc: 0.6615
val Loss: 0.7634 Acc: 0.7980

Epoch 12/25
----------
train Loss: 1.3995 Acc: 0.6626
val Loss: 0.7595 Acc: 0.7990

Epoch 13/25
----------
train Loss: 1.3882 Acc: 0.6647
val Loss: 0.7558 Acc: 0.7988

Epoch 14/25
----------
train Loss: 1.3747 Acc: 0.6680
val Loss: 0.7517 Acc: 0.7997

Epoch 15/25
----------
train Loss: 1.3754 Acc: 0.6683
val Loss: 0.7490 Acc: 0.8006

Epoch 16/25
----------
train Loss: 1.3685 Acc: 0.6689
val Loss: 0.7592 Acc: 0.7970

Epoch 17/25
----------
train Loss: 1.3771 Acc: 0.6681
val Loss: 0.7567 Acc: 0.8009

Epoch 18/25
----------
train Loss: 1.3690 Acc: 0.6688
val Loss: 0.7508 Acc: 0.8011

Epoch 19/25
----------
train Loss: 1.3716 Acc: 0.6694
val Loss: 0.7521 Acc: 0.8008

Epoch 20/25
----------
train Loss: 1.3729 Acc: 0.6687
val Loss: 0.7527 Acc: 0.8002

Epoch 21/25
----------
train Loss: 1.3709 Acc: 0.6689
val Loss: 0.7501 Acc: 0.8014

Epoch 22/25
----------
train Loss: 1.3706 Acc: 0.6708
val Loss: 0.7516 Acc: 0.8008

Epoch 23/25
----------
train Loss: 1.3681 Acc: 0.6696
val Loss: 0.7502 Acc: 0.8002

Epoch 24/25
----------
train Loss: 1.3725 Acc: 0.6698
val Loss: 0.7508 Acc: 0.8003

Epoch 25/25
----------
train Loss: 1.3708 Acc: 0.6696
val Loss: 0.7480 Acc: 0.8004

Best val Acc: 0.8014
Model saved as tiny_imagenet_resnet50_epoch25.pth

```

跑了4小时30分钟。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250224171542842.png" alt="image-20250224171542842" style="zoom:50%;" />



```
% ls -lh *.pth
-rw-r--r--  1 hfyan  staff    92M Feb 24 16:02 tiny_imagenet_resnet50_epoch25.pth
```





**Q.这个程序保存为什么文件名字好？**

通常建议文件名能够反映出数据集、模型结构以及训练轮数等关键信息，这样便于后续管理和调试。比如你可以保存为：

```python
tiny_imagenet_resnet50_epoch25.pth
```

这里 "tiny_imagenet" 表示数据集，"resnet50" 表示模型结构，"epoch25" 则说明训练了 25 个轮次。



## 4.加载训练好的的模型并进行验证

前面已经保存了模型权重，可以通过如下步骤加载模型并在验证集上进行评估：

1. **加载模型结构和权重**  
   请确保你定义的模型结构与训练时保持一致。使用 `torch.load` 加载权重，并用 `model.load_state_dict` 导入。

2. **切换到评估模式**  
   调用 `model.eval()` 确保模型关闭 BatchNorm、Dropout 等训练时特有的行为。

3. **遍历验证数据并计算准确率**  
   使用 `torch.no_grad()` 关闭梯度计算，加快验证速度，并防止内存浪费。

下面是一个完整的示例代码，`eval_tiny_imagenet_resnet50_epoch25_pth.py `：

```python
import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

# 数据预处理
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# 确保子进程安全启动
if __name__ == '__main__':
    data_dir = '/Users/hfyan/data/tiny-imagenet-200'
    val_dir = os.path.join(data_dir, 'val')
    val_dataset = datasets.ImageFolder(val_dir, data_transforms['val'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64,
                                             shuffle=False, num_workers=4)

    # 选择设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("MPS device not available, using CPU")

    # 加载模型
    model_ft = models.resnet50(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(val_dataset.classes))
    model_ft = model_ft.to(device)

    # 加载模型权重
    model_path = 'tiny_imagenet_resnet50_epoch25.pth'
    model_ft.load_state_dict(torch.load(model_path, map_location=device))

    # 评估模式
    model_ft.eval()

    # 模型评估
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)

    val_acc = running_corrects.float() / total_samples
    print('Validation Accuracy: {:.4f}'.format(val_acc))

```

---

**总结**

- **模型保存文件名建议**：  
  使用描述性文件名，例如 `tiny_imagenet_resnet50_epoch25.pth`，这样便于识别数据集、模型及训练轮次。

- **验证步骤**：  
  - 加载与你训练时一致的模型结构。  
  - 使用 `model.load_state_dict()` 加载权重。  
  - 调用 `model.eval()` 进入验证模式。  
  - 遍历验证数据集，计算准确率或其他指标。

这样，你就可以加载已保存的模型并对验证集数据进行测试。

![image-20250224171857564](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250224171857564.png)



```python
(base) hfyan@HongfeideMac-Studio data % python eval_tiny_imagenet_resnet50_epoch25_pth.py 
Using MPS device for GPU acceleration
/Users/hfyan/miniconda3/lib/python3.10/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/Users/hfyan/miniconda3/lib/python3.10/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
  warnings.warn(msg)
Validation Accuracy: 0.8014
(base) hfyan@HongfeideMac-Studio data % 
```



## 附录



### 多进程 vs. 多线程

**多进程**：`multiprocessing` 模块允许你创建多个独立的进程。每个进程都有自己的Python解释器实例和内存空间，这意味着它们不会受到GIL（Global Interpreter Lock，全局解释器锁）的影响，并且可以充分利用多核处理器的能力进行真正的并行计算。然而，进程间的通信和数据交换（如共享状态或传递消息）相比线程会更加复杂和消耗资源。

**多线程**：另一方面，Python 的 `threading` 模块提供了多线程的支持。在同一个程序内，你可以启动多个线程，这些线程共享相同的内存空间。尽管这使得线程间通信更加简单直接，但由于GIL的存在，对于CPU密集型任务，多线程并不能带来真正的并行执行，它更适合于I/O密集型的任务（例如网络请求、文件读写等），在这种情况下，线程可以在等待I/O操作完成的同时让出执行权给其他线程。

因此，如果你正在处理需要大量CPU计算的任务，并希望利用多核处理器提高性能，那么使用 `multiprocessing` 模块是更合适的选择。而对于涉及大量等待外部资源（如数据库访问、网络请求等）的应用场景，`threading` 可能更为适用。





#### 示例18161: 矩阵运算

matrices, http://cs101.openjudge.cn/practice/18161



##### 多进程multiprocessing

使用 multiprocessing 模块以及 dot_product 函数来实现矩阵运算 A·B + C。将利用多进程并行计算矩阵乘法部分，然后将结果与矩阵 C 相加。

内存: 29964，时间: 520ms

```python
import multiprocessing

def dot_product(row, col):
    """
    计算两个向量的点积。
    :param row: 第一个矩阵的一行
    :param col: 第二个矩阵的一列
    :return: 点积结果
    """
    return sum(a * b for a, b in zip(row, col))

def matrix_multiply_parallel(A, B, num_processes=None):
    """
    使用多进程并行计算两个矩阵的乘积。
    :param A: 第一个矩阵，作为列表的列表
    :param B: 第二个矩阵，作为列表的列表
    :param num_processes: 要使用的进程数，默认为 None (自动决定)
    :return: 矩阵乘积的结果
    """
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions do not match for multiplication")

    result = [[None for _ in range(len(B[0]))] for _ in range(len(A))]
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        tasks = []
        for i in range(len(A)):
            for j in range(len(B[0])):
                row = A[i]
                col = [B_element[j] for B_element in B]
                # 异步调用dot_product函数
                tasks.append(pool.apply_async(dot_product, (row, col)))
        
        index = 0
        for i in range(len(A)):
            for j in range(len(B[0])):
                #任务收集逻辑：通过直接使用 tasks[index].get() 来获取每个异步任务的结果
                result[i][j] = tasks[index].get()
                index += 1
    
    return result

def matrix_add(X, Y):
    if len(X) != len(Y) or len(X[0]) != len(Y[0]):
        raise ValueError("Matrices must have the same dimensions for addition")
    return [[X[i][j] + Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]

def read_matrix():
    import sys
    input = sys.stdin.read
    data = input().strip().split('\n')
    matrices = []
    idx = 0
    while idx < len(data):
        row, col = map(int, data[idx].split())
        matrix = []
        for r in range(row):
            matrix.append(list(map(int, data[idx + 1 + r].split())))
        matrices.append(matrix)
        idx += row + 1
    return matrices

def main():
    matrices = read_matrix()
    A, B, C = matrices
    
    try:
        AB = matrix_multiply_parallel(A, B, 4)
        result = matrix_add(AB, C)
        for row in result:
            print(' '.join(map(str, row)))
    except ValueError:
        print("Error!")

if __name__ == "__main__":
    main()
```



##### 单进程

内存: 4392kB，时间: 71ms

```python
def read_matrix():
    """读取矩阵输入"""
    row, col = map(int, input().split())
    matrix = [list(map(int, input().split())) for _ in range(row)]
    return matrix

def matrix_multiply(A, B):
    """计算两个矩阵的乘积"""
    # 确保A的列数等于B的行数
    if len(A[0]) != len(B):
        raise ValueError("Matrix dimensions do not match for multiplication")
    
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            sum = 0
            for k in range(len(A[0])):
                sum += A[i][k] * B[k][j]
            row.append(sum)
        result.append(row)
    return result

def matrix_add(X, Y):
    """计算两个矩阵的和"""
    # 确保X和Y的维度相同
    if len(X) != len(Y) or len(X[0]) != len(Y[0]):
        raise ValueError("Matrices must have the same dimensions for addition")
    
    result = [[X[i][j] + Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]
    return result

def main():
    # 读取矩阵A、B、C
    A = read_matrix()
    B = read_matrix()
    C = read_matrix()

    try:
        # 计算矩阵乘法A·B
        AB = matrix_multiply(A, B)

        # 检查矩阵AB和C的维度是否适合相加
        if len(AB) != len(C) or len(AB[0]) != len(C[0]):
            raise ValueError("Resultant matrix AB and C must have the same dimensions for addition")

        # 计算矩阵加法AB+C
        result = matrix_add(AB, C)

        # 输出结果
        for row in result:
            print(' '.join(map(str, row)))
    except ValueError as e:
        print("Error!")

if __name__ == "__main__":
    main()
```



> Q. 开 multiprocessing，4线程，运行时间比普通方法计算反倒慢了一个数量级。是测试数据太少？
>
> 使用 `multiprocessing` 模块进行并行计算时，确实有可能在某些情况下（特别是处理小规模数据或简单任务）导致性能下降。这主要是因为：
>
> 1. **进程启动和通信开销**：创建新进程以及跨进程边界传递数据（如将任务分配给不同的工作进程）需要时间。如果任务本身非常轻量级（比如简单的矩阵乘法元素计算），这些额外的开销可能会超过并行化带来的性能增益。
>
> 2. **数据规模不足**：对于较小的数据集或简单的计算任务，CPU可能很快就能完成计算，此时多线程或多进程的优势无法体现出来。实际上，由于上述提到的进程管理和通信成本，反而可能导致总体执行时间增加。
>
> 3. **Python的全局解释器锁（GIL）**：尽管使用了 `multiprocessing` 可以绕过 GIL，但对于I/O密集型而非计算密集型任务，其优势并不明显。此外，在计算密集型任务中，如果任务粒度太细，频繁的上下文切换也可能影响性能。
>
> **测试数据大小的影响**
>
> 如果你使用的测试数据很小（例如样例输入中的3x1、1x2等小矩阵），那么并行化的开销会显得尤为突出。这是因为：
>
> - 并行化带来的加速比依赖于任务可以被有效分割的程度以及每个子任务的计算量。
> - 对于非常小的矩阵，单个点积操作的计算量不足以抵消进程间通信和管理的开销。
>
> **改进建议**
>
> 如果你想验证并行化是否能在更大规模的数据上带来性能提升，建议尝试以下方法：
>
> 1. **增加数据规模**：使用更大的矩阵进行测试，这样每个子任务（即两个向量的点积）会有更多的计算量，从而更有可能从并行化中受益。
>
> 2. **调整线程/进程数**：根据你的CPU核心数合理设置进程池的大小。过多的进程会导致不必要的上下文切换，而过少则不能充分利用多核处理器的能力。
>
> 3. **优化任务粒度**：确保你划分的任务足够大，以便能够掩盖进程间的通信和调度开销。例如，可以考虑将整个行的计算作为一个任务，而不是单独计算每个元素。
>
> 4. **测量性能**：使用计时工具（如 `time` 模块）来精确测量不同实现方式下的运行时间，并对比分析结果。
>
> 总之，对于小规模数据或简单计算任务，直接使用单线程实现通常更为高效。当处理大规模数据或复杂计算时，适当利用并行化技术才能显著提高性能。





### Torch 的多进程计算不受 GIL 限制

https://github.com/GMyhf/2025spring-cs201/blob/main/LLM/nn_4_tiny_imagenet_classify.md

```python
...
    # 设置 num_workers 为 4 以利用多线程数据加载
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                    batch_size=128,    
                                                    shuffle=True,
                                                    num_workers=8)
...
```

GIL 的全称是 **Global Interpreter Lock**（全局解释器锁）。这是 Python 解释器（CPython）中的一个机制，用于确保同一时间只有一个线程执行 Python 字节码，即便在多核 CPU 上，也无法实现真正的并行计算。

不过，**Torch 的多近程计算** 通常 **不受 GIL 限制**，原因如下：

1. **计算密集型任务在 C/C++ 后端执行**  
   PyTorch 的底层计算是基于 C++ 和 CUDA 实现的。一旦计算任务进入底层库（如调用 `torch.matmul()`、`torch.nn` 等操作），它就脱离了 Python 的解释层，因此不会被 GIL 锁住。计算主要在 C++ 层面并行执行，或者交给 GPU 加速。

2. **DataLoader 使用多进程并行加载数据**  
   你设置的 `num_workers=8` 实际上会启动 8 个 **子进程** 来加载数据，因为 PyTorch 的 `DataLoader` 默认使用 **multiprocessing** 库，而不是线程池。这种方式是跨进程的，不会触发 GIL。

3. **异步计算（CUDA）**  
   如果你在使用 GPU 计算（如 `.to('cuda')`），大部分操作是异步的。Python 线程触发计算任务后，CUDA 内核在 GPU 上执行，线程会立刻返回，继续处理其他任务。因此不会卡在 GIL 上。

**总结：**

- **CPU-bound任务**（纯 Python 代码计算）会受到 GIL 限制。
- **I/O-bound任务**（如数据加载、文件读取）在 PyTorch 中主要靠多进程，不受 GIL 影响。
- **GPU-bound任务** 完全在 CUDA 上执行，与 GIL 无关。

所以 `num_workers=8` 是在开 8 个进程来并行加载数据，计算部分走的底层 C++/CUDA，不会因为 GIL 产生性能瓶颈！



### 并行计算

我有两个同样的Mac Studio机器，如何协同工作，能让这个程序跑的快点。现在每个epoch在单机跑的慢。



要让你的两台 Mac Studio 协同加速训练，可以使用 **分布式数据并行（DDP, Distributed Data Parallel）**。PyTorch 支持基于多个设备甚至多个节点（机器）进行分布式训练。以下是步骤：

**步骤1：设置主机和端口**

选择一台 Mac 作为主节点（主控机），假设它的局域网 IP 是 `192.168.1.100`。

在两台机器的终端中分别运行：

```bash
# 替换<node_rank>:
# 主节点：node_rank=0
# 从节点：node_rank=1

python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=<node_rank> \
    --master_addr=192.168.1.100 \
    --master_port=12345 \
    tiny_imagenet_resnet50_epoch25.py
```

- `--nproc_per_node=1` 表示每台机器用一个 GPU（M1 Ultra GPU 作为一个逻辑设备）。
- `--nnodes=2` 表示两台机器参与训练。
- `--node_rank` 主节点是 0，从节点是 1。
- `--master_addr` 设置为主节点的 IP 地址。
- `--master_port` 可以是任意空闲端口号。

---

**步骤2：修改代码以支持 DDP**

在你的训练代码里，添加以下内容：

**初始化分布式训练：**

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
```

**模型包装为 DDP：**

在模型定义后加：

```python
device = torch.device("mps")
model = model.to(device)
model = DDP(model, device_ids=[0])
```

**分布式数据加载：**

使用 `DistributedSampler` 来确保各节点的数据不会重复：

```python
from torch.utils.data.distributed import DistributedSampler

train_sampler = DistributedSampler(image_datasets['train'])
val_sampler = DistributedSampler(image_datasets['val'])

dataloaders = {
    'train': torch.utils.data.DataLoader(
        image_datasets['train'], batch_size=128, sampler=train_sampler, num_workers=8
    ),
    'val': torch.utils.data.DataLoader(
        image_datasets['val'], batch_size=128, sampler=val_sampler, num_workers=8
    )
}
```

**训练循环同步：**

在每个 epoch 结束后，确保梯度同步：

```python
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    train_model(...)  # 你的训练逻辑
```

---

**步骤3：网络配置**

确保两台 Mac：

- 在同一个局域网内，互相可以通过 IP 地址访问。
- 检查防火墙设置是否允许端口通信：

```bash
sudo ufw allow 12345
```

或者临时关闭防火墙以测试：

```bash
sudo systemctl stop firewall
```

---

**额外建议**

1. **监控 GPU 和内存使用情况：**  
   macOS 上可以使用 `Activity Monitor` 来观察 GPU 和内存占用。

2. **调试分布式通信：**  
   如果出现超时或连接失败错误，检查节点 IP 是否正确，以及端口是否被占用。

---

这样设置后，你的两台 Mac Studio 就可以并行训练模型，大大加快训练速度！



### Q.resnet50 图像分类，精度达到多少？

ResNet50 的精度根据不同的训练设置和数据集有所变化。以下是几个不同情况下的准确率：

1. 原始的 ResNet-50 模型在 ImageNet 数据集上的 Top-1 准确率大约为 76.5%（根据原始论文中的结果）。

2. 在一些优化和改进后，比如使用一系列技巧（tricks），可以将 ResNet-50 的 Top-1 准确率提升到大约 79.1%。

3. 使用 SSLD（Simple Semi-supervised Label Distillation）知识蒸馏策略，并结合 ImageNet22k 数据集进行训练，可以进一步提高模型性能，使得 ResNet-50_vd 的 Top-1 准确率达到 82.4%。

4. 更进一步的改进，如采用特定的数据增强、分辨率调整以及基于 Fix 策略的微调等方法，可以在 ImageNet 验证集上达到更高的精度。例如，PaddleClas 文档中提到，通过这些优化措施，ResNet50_vd 的精度达到了 83%，而在 320x320 尺度下测试时精度可达 83.7%。最终，在采用了特定的 Fix 策略之后，精度甚至可以达到 84.0%。

5. 另外，有文献报道了通过知识蒸馏等方法，可以让 ResNet-50 达到 82.8% 的 Top-1 准确率。

请注意，上述数值反映了在标准 ImageNet 数据集上的表现，实际应用中针对特定任务或数据集的精度可能会有所不同。如果你要针对自己的数据集进行训练，可能需要根据具体情况调整网络结构、训练参数或使用迁移学习等技术来提高分类精度。



### Q.**ImageNet**和**Tiny ImageNet**的数据规模

**ImageNet**  

- **类别数**：ImageNet的常规版本（如ILSVRC挑战使用的数据）包含 **1000个类别**。  
- **每类图片数**：  
  - **训练集**：大约 **1200张图片/类**。  
  - **验证集**：每类 **50张图片**。  
  - **测试集**：每类 **100张图片**。  
- **总图片数**：  
  - 训练集约 **128万张图片**。  
  - 验证集 **50,000张图片**。  
  - 测试集 **100,000张图片**。  
- **硬盘空间**：原始ImageNet的完整数据集大约 **150GB**。解压后可能更大。  

---

**Tiny ImageNet**  

- **类别数**：**200个类别**。  
- **每类图片数**：  
  - **训练集**：每类 **500张图片**。  
  - **验证集**：每类 **50张图片**。  
  - **测试集**：每类 **50张图片**。  
- **总图片数**：  
  - 训练集 **100,000张图片**。  
  - 验证集 **10,000张图片**。  
  - 测试集 **10,000张图片**。  
- **图片尺寸**：所有图片均为 **64x64像素**。  
- **硬盘空间**：大约 **250MB**。  

---

**总结**：

- **ImageNet** 是大规模数据集（150GB+），适合训练深层模型。  
- **Tiny ImageNet** 是简化版，250MB，非常适合快速测试模型原型或进行学术实验。  
