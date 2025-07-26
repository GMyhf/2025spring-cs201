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

