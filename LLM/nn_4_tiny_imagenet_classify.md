# Neural Network实现Tiny ImageNet图像分类

Updated 1723 GMT+8 Feb 24 2025

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

    # 设置 num_workers 为 4 以利用多线程数据加载
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                    batch_size=128,    # 可根据实际情况调整
                                                    shuffle=True,
                                                    num_workers=8)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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



## Todo. 并行计算

我有两个同样的Mac Studio机器，如何协同工作，能让这个程序跑的快点。现在每个epoch在单机跑的慢。



要让你的两台 Mac Studio 协同加速训练，可以使用 **分布式数据并行（DDP, Distributed Data Parallel）**。PyTorch 支持基于多个设备甚至多个节点（机器）进行分布式训练。以下是步骤：

### 步骤1：设置主机和端口
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

### 步骤2：修改代码以支持 DDP

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

### 步骤3：网络配置

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

### 额外建议

1. **监控 GPU 和内存使用情况：**  
   macOS 上可以使用 `Activity Monitor` 来观察 GPU 和内存占用。

2. **调试分布式通信：**  
   如果出现超时或连接失败错误，检查节点 IP 是否正确，以及端口是否被占用。

---

这样设置后，你的两台 Mac Studio 就可以并行训练模型，大大加快训练速度！