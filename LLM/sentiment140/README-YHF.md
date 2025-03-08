# sentiment140文本情感分类

Updated 2337 GMT+8 Mar 8 2025

2025 spring, Complied by Hongfei Yan



## 原来程序

`test.py`内容

```python
import pandas as pd
from sklearn.metrics import accuracy_score

# 将当前工作目录切换到 submission 文件夹下，用来导入模块
import os
import sys
os.chdir("./submission")
sys.path.append('.')
from sentiment_analysis import PREDICT
import random
dataset = pd.read_csv("../Sentiment_Analysis_Dataset_test.csv", on_bad_lines='skip')
label = dataset['Sentiment']
data = dataset['SentimentText']
prediction = PREDICT(data)
print("ok!")
# print(prediction)
# print(type(prediction))
# print(prediction.shape)
for i in range(len(prediction)):
    if prediction[i] >= 0.2:
        prediction[i] = 1
    else:
        prediction[i] = 0
print("local test accuracy:", accuracy_score(label, prediction))

```



`submission`结构

```plain
submission
├── sentiment_analysis.py：包含 PREDICT 方法
├── train.py
└── Sentiment_Analysis_Dataset.csv：训练时的数据集，157MB
```



`train.py` 内容如下

```python
def TRAIN():
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer  

    dataset=pd.read_csv("Sentiment_Analysis_Dataset.csv", on_bad_lines='skip')
    label=dataset['Sentiment']
    data=dataset['SentimentText']
    # 词袋模型
    bow_vectorizer = CountVectorizer()
    data = bow_vectorizer.fit_transform(data)
    # 交叉验证，随机分隔数据集
    xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(data, label, random_state=42, test_size=0.3)
    # 线性模型训练
    model = LinearRegression()
    model.fit(xtrain_bow, ytrain)

    return model, bow_vectorizer
```

`sentiment_analysis.py` 内容如下
```python
def PREDICT(data):  # 输入是 np array
    # 训练模型
    from train import TRAIN
    model, bow_vectorizer = TRAIN()
    # 预测结果
    numericdata = bow_vectorizer.transform(data)
    # 请返回np array，验证方法，print(numericdata.shape)，输出(xxxx,)即为正确
    return model.predict(numericdata)
```



`Sentiment_Analysis_Dataset.csv`数据

```plain
ItemID,Sentiment,SentimentSource,SentimentText
1,0,Sentiment140,                     is so sad for my APL friend.............
2,0,Sentiment140,                   I missed the New Moon trailer...
3,1,Sentiment140,              omg its already 7:30 :O
4,0,Sentiment140,          .. Omgaga. Im sooo  im gunna CRy. I've been at this dentist since 11.. I was suposed 2 just get a crown put on (30mins)...
5,0,Sentiment140,         i think mi bf is cheating on me!!!       T_T
6,0,Sentiment140,         or i just worry too much?        
7,1,Sentiment140,       Juuuuuuuuuuuuuuuuussssst Chillin!!
8,0,Sentiment140,       Sunny Again        Work Tomorrow  :-|       TV Tonight
9,1,Sentiment140,      handed in my uniform today . i miss you already
10,1,Sentiment140,      hmmmm.... i wonder how she my number @-)
11,0,Sentiment140,      I must think about positive..
12,1,Sentiment140,      thanks to all the haters up in my face all day! 112-102
13,0,Sentiment140,      this weekend has sucked so far
14,0,Sentiment140,     jb isnt showing in australia any more!
15,0,Sentiment140,     ok thats it you win.
16,0,Sentiment140,    &lt;-------- This is the way i feel right now...
17,0,Sentiment140,"    awhhe man.... I'm completely useless rt now. Funny, all I can do is twitter. http://myloc.me/27HX"
18,1,Sentiment140,    Feeling strangely fine. Now I'm gonna go listen to some Semisonic to celebrate
19,0,Sentiment140,    HUGE roll of thunder just now...SO scary!!!!
20,0,Sentiment140,    I just cut my beard off. It's only been growing for well over a year. I'm gonna start it over. @shaunamanu is happy in the meantime.
21,0,Sentiment140,    Very sad about Iran.
22,0,Sentiment140,    wompppp wompp
23,1,Sentiment140,    You're the only one who can see this cause no one else is following me this is for you because you're pretty awesome
24,0,Sentiment140,   &lt;---Sad level is 3. I was writing a massive blog tweet on Myspace and my comp shut down. Now it's all lost *lays in fetal position*
25,0,Sentiment140,   ...  Headed to Hospitol : Had to pull out of the Golf Tourny in 3rd place!!!!!!!!!!! I Think I Re-Ripped something !!! Yeah THAT !!
26,0,Sentiment140,   BoRinG   ): whats wrong with him??     Please tell me........   :-/
27,0,Sentiment140,   can't be bothered. i wish i could spend the rest of my life just sat here and going to gigs. seriously.
28,0,Sentiment140,"   Feeeling like shit right now. I really want to sleep, but nooo I have 3 hours of dancing and an art assignment to finish. "
29,1,Sentiment140,"   goodbye exams, HELLO ALCOHOL TONIGHT "
30,0,Sentiment140,   I didn't realize it was THAT deep. Geez give a girl a warning atleast!
31,0,Sentiment140,   I hate it when any athlete appears to tear an ACL on live television.
32,0,Sentiment140,   i miss you guys too     i think i'm wearing skinny jeans a cute sweater and heels   not really sure   what are you doing today
33,0,Sentiment140,  -- Meet your Meat http://bit.ly/15SSCI
34,0,Sentiment140,   My horsie is moving on Saturday morning.
35,0,Sentiment140,   No Sat off...Need to work 6 days a week 
36,0,Sentiment140,   Really Dont Like Doing my Room Its So Boring  Sick Of Doing My Wardrobe Out Cant Waiit Till I Have My Walk In One  Yay
37,0,Sentiment140,"   SOX!     Floyd was great, but relievers need a scolding!"
38,0,Sentiment140,   times by like a million
39,1,Sentiment140,   uploading pictures on friendster 
40,0,Sentiment140,   what type of a spaz downloads a virus? my brother that's who :\ MSN is now fucked forever    :'(
41,0,Sentiment140,  &amp;&amp;Fightiin Wiit The Babes...
42,1,Sentiment140,  (: !!!!!! - so i wrote something last week. and i got a call from someone in the new york office... http://tumblr.com/xcn21w6o7
43,0,Sentiment140,  *enough said*
44,1,Sentiment140,"  ... Do I need to even say it?  Do I?  Well, here I go anyways:  CHRIS CORNELL IN CHICAGO!  ... TONIGHT!    "
45,1,Sentiment140,  ... health class (what a joke!)
46,1,Sentiment140,  @ginaaa &lt;3 GO TO THE SHOW TONIGHT
47,0,Sentiment140,  @Spiral_galaxy @YMPtweet  it really makes me sad when i look at Muslims reality now
48,0,Sentiment140, - All Time Low shall be my motivation for the rest of the week.
49,0,Sentiment140,"  and the entertainment is over, someone complained properly..   @rupturerapture experimental you say? he should experiment with a melody"
50,0,Sentiment140,  another year of Lakers .. That's neither magic nor fun ...
51,0,Sentiment140,  baddest day eveer. 
52,1,Sentiment140,  bathroom is clean..... now on to more enjoyable tasks......
53,1,Sentiment140,  boom boom pow
54,0,Sentiment140,  but i'm proud.
55,0,Sentiment140,  congrats to helio though
56,0,Sentiment140,  David must be hospitalized for five days end of July (palatine tonsils). I will probably never see Katie in concert. 
57,0,Sentiment140,  friends are leaving me 'cause of this stupid love  http://bit.ly/ZoxZC
58,1,Sentiment140,  go give ur mom a hug right now. http://bit.ly/azFwv
59,1,Sentiment140,  Going To See Harry Sunday Happiness 
60,0,Sentiment140,  Hand quilting it is then...
61,0,Sentiment140,  hate u ...  leysh t9ar5 ... =((((((( ..
...
...
1578616,1,Sentiment140,zzz twitter. good day today. got a lot accomplished. imstorm. got into it w yet another girl. dress shopping tmrw 
1578617,1,Sentiment140,"zzz's time, goodnight.  http://plurk.com/p/ri9qn"
1578618,0,Sentiment140,Zzzz lying in bed watching the countryside through my window. Last day of freedom  gonna make an incred bacon sandwich...
1578619,1,Sentiment140,Zzzz... Fuck Ã¼ : Zzzz... Fuck Ã¼  http://bit.ly/1a8JxZ
1578620,1,Sentiment140,Zzzz...no work tomorrow..yayyy!! 
1578621,1,Sentiment140,ZZZZZ time.. Tomorrow will be a busy day for serving &amp; loving people! Luv y'all! 
1578622,0,Sentiment140,Zzzzz want to sleep but at sister's in-laws's house. 
1578623,1,Sentiment140,Zzzzzz.... Finally! Night tweeters! 
1578624,1,Sentiment140,"Zzzzzzz, sleep well people "
1578625,0,Sentiment140,ZzzZzZzzzZ... wait no I have homework. 
1578626,0,Sentiment140,"ZzZzzzZZZZzzz meh, what am I doing up again? "
1578627,0,Sentiment140,"Zzzzzzzzzzzzzzzzzzz, I wish "
```

目前的 local test accuracy: 0.745



## 改进模型

### 1问AI：Embedding+LSTM模型

我之前用PyTorch库，用Neural Network实现Tiny ImageNet图像分类，精度到80%左右。

链接是 https://github.com/GMyhf/2025spring-cs201/blob/main/LLM/nn_4_tiny_imagenet_classify.md

`tiny_imagenet_resnet50_epoch25.py`代码是：

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



请用类似方法改造上面的Sentiment_Analysis，使之精度提高。



#### AI反馈

Reasoned about evaluating sentiment model for 35 seconds >

下面给出一种改造方案，思路与 Tiny ImageNet 分类的改造类似——采用 PyTorch 构建一个神经网络模型来处理文本情感分类任务。基本流程包括：

1. **数据预处理与自定义数据集**
   – 定义简单的分词函数，对文本进行预处理（转小写、去除标点等）
   – 根据训练数据构建词典（添加填充和未知词标记），并将文本转为固定长度的词索引序列
2. **模型设计**
   – 构建一个基于 Embedding+LSTM（或 GRU）的文本分类网络
   – 最后用全连接层将 LSTM 的隐藏状态映射到二分类（输出维度为 1），用 BCEWithLogitsLoss 作为损失函数
3. **训练流程**
   – 将数据分为训练集和验证集，使用 DataLoader 加载数据
   – 采用类似 Tiny ImageNet 中的训练循环（包括学习率调度等）进行训练，并保存效果最好的模型权重及词典
4. **预测函数改造**
   – 在 `sentiment_analysis.py` 中加载保存好的模型和词典
   – 对输入文本进行相同的预处理，转换为模型输入，然后输出概率（由 test.py 进一步阈值化）

下面给出改造后的代码示例。

`train_YHF.py`

```python
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import pickle
from sklearn.model_selection import train_test_split


# 1. 数据预处理工具
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    return tokens


def build_vocab(texts, min_freq=2):
    word_freq = {}
    for text in texts:
        tokens = tokenize(text)
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def text_to_sequence(text, vocab, max_len=100):
    tokens = tokenize(text)
    sequence = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(sequence) < max_len:
        sequence += [vocab["<PAD>"]] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return sequence


# 2. 自定义 Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx] if isinstance(self.texts, pd.Series) else self.texts[idx]
        label = self.labels.iloc[idx] if isinstance(self.labels, pd.Series) else self.labels[idx]
        sequence = text_to_sequence(text, self.vocab, self.max_len)
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.float)


# 3. 模型定义：Embedding + LSTM 分类器
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        _, (hn, _) = self.lstm(embedded)  # hn: [num_layers, batch_size, hidden_dim]
        hn = hn[-1]  # 取最后一层隐藏状态 [batch_size, hidden_dim]
        out = self.fc(hn)  # [batch_size, 1]
        return out.squeeze(1)  # [batch_size]


# 4. 训练流程
def TRAIN():
    # 读取数据集
    dataset = pd.read_csv("Sentiment_Analysis_Dataset.csv", on_bad_lines='skip')
    texts = dataset['SentimentText']
    labels = dataset['Sentiment']

    # 构建词典
    vocab = build_vocab(texts)
    print("词典大小:", len(vocab))

    # 划分训练集和验证集
    texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
    max_len = 100

    train_dataset = SentimentDataset(texts_train, labels_train, vocab, max_len)
    val_dataset = SentimentDataset(texts_val, labels_val, vocab, max_len)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("MPS device not available, using CPU")
    model = SentimentClassifier(len(vocab), embed_dim=128, hidden_dim=128, num_layers=1)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    best_model_state = None
    num_epochs = 10  # 可根据需求调节
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        for sequences, labels_batch in train_loader:
            sequences = sequences.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * sequences.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds.float() == labels_batch).sum().item()
            total_train += sequences.size(0)
        train_loss /= total_train
        train_acc = train_correct / total_train

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for sequences, labels_batch in val_loader:
                sequences = sequences.to(device)
                labels_batch = labels_batch.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item() * sequences.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds.float() == labels_batch).sum().item()
                total_val += sequences.size(0)
        val_loss /= total_val
        val_acc = val_correct / total_val

        print(f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()

        scheduler.step()
        print()

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 保存模型和词典
    torch.save(model.state_dict(), "sentiment_model.pth")
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("模型和词典已保存。")

    return model, vocab


if __name__ == '__main__':
    TRAIN()

```



**新版 `sentiment_analysis_YHF.py`**

```
python


CopyEdit
import torch
import torch.nn as nn
import pickle
import re
import numpy as np

# 与训练阶段一致的预处理
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    return tokens

def text_to_sequence(text, vocab, max_len=100):
    tokens = tokenize(text)
    sequence = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(sequence) < max_len:
        sequence += [vocab["<PAD>"]] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return sequence

# 定义与训练时相同的模型结构
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hn, _) = self.lstm(embedded)
        hn = hn[-1]
        out = self.fc(hn)
        return out.squeeze(1)

# 加载保存好的模型与词典
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)
model = SentimentClassifier(vocab_size, embed_dim=128, hidden_dim=128, num_layers=1)
model.load_state_dict(torch.load("sentiment_model.pth", map_location=device))
model.to(device)
model.eval()

def PREDICT(data):
    """
    输入 data 为 numpy array 或列表，包含若干文本。
    返回值为 numpy array，每个元素为模型预测的概率（[0, 1]区间）。
    注意：test.py 中会基于阈值对结果进行二值化。
    """
    if isinstance(data, np.ndarray):
        texts = data.tolist()
    elif isinstance(data, list):
        texts = data
    else:
        texts = [data]
    
    max_len = 100
    sequences = [text_to_sequence(text, vocab, max_len) for text in texts]
    sequences = torch.tensor(sequences, dtype=torch.long).to(device)
    
    with torch.no_grad():
        outputs = model(sequences)
        probs = torch.sigmoid(outputs)  # 将 logits 转为概率
    return probs.cpu().numpy()
```

------

说明

- **预处理一致性**
  训练和预测过程中都使用了同样的分词、词典构建及序列化方法，确保输入一致。
- **模型架构**
  采用了 Embedding 层和 LSTM 来捕捉文本的序列信息，相较于传统词袋+线性回归模型，能够更好地学习上下文信息，从而有望提高情感分类的精度。
- **训练流程与保存**
  训练过程中采用交叉验证（简单的训练/验证划分）以及学习率调度，并保存了效果最好的模型和词典，供预测时加载。
- **预测函数**
  PREDICT 接收文本数据，进行同样的预处理后，输出经过 sigmoid 转换的概率。test.py 中对概率进行阈值化操作（原代码使用 0.2 阈值，可根据实际情况调整）。

通过这样的改造，如果使用足够的训练数据和适当的超参数调优，神经网络模型通常能在文本情感分类任务上取得比 0.745 更高的精度。你可以根据实际效果进一步调整模型结构、增加数据预处理（例如引入停用词过滤或更高级的分词工具）、或采用更复杂的网络（如 Transformer 模型）来进一步提升性能。



#### 训练模型

在Mac Studio 上运行3个小时

```
/Users/hfyan/miniconda3/bin/python /Users/hfyan/Code/sentiment/sentiment_analysis-v2.0/submission/train_YHF.py 
词典大小: 251179
Using MPS device for GPU acceleration
Epoch 1/10
Train Loss: 0.6932  Acc: 0.5000
Val Loss: 0.6931  Acc: 0.5006

Epoch 2/10
Train Loss: 0.6932  Acc: 0.5003
Val Loss: 0.6931  Acc: 0.4994

Epoch 3/10
Train Loss: 0.6932  Acc: 0.5001
Val Loss: 0.6932  Acc: 0.4994

Epoch 4/10
Train Loss: 0.6932  Acc: 0.5003
Val Loss: 0.6932  Acc: 0.4994

Epoch 5/10
Train Loss: 0.6932  Acc: 0.5003
Val Loss: 0.6932  Acc: 0.4994

Epoch 6/10
Train Loss: 0.6932  Acc: 0.5002
Val Loss: 0.6932  Acc: 0.5006

Epoch 7/10
Train Loss: 0.6932  Acc: 0.5002
Val Loss: 0.6932  Acc: 0.4994

Epoch 8/10
Train Loss: 0.6932  Acc: 0.5000
Val Loss: 0.6931  Acc: 0.5006

Epoch 9/10
Train Loss: 0.6932  Acc: 0.4997
Val Loss: 0.6931  Acc: 0.5006

Epoch 10/10
Train Loss: 0.6932  Acc: 0.4999
Val Loss: 0.6932  Acc: 0.4994

模型和词典已保存。
```



保存的模型和字典是：

<mark>sentiment_model.pth, 129MB，vocab.pkl, 4MB</mark>



### 2问AI：评估Embedding+LSTM模型

请利用保存的模型和字典，对`Sentiment_Analysis_Dataset_test.csv`进行评测，得到精度等。给出python代码。

```
ItemID,Sentiment,SentimentText
1,1,HAPPY GRADUATION DAY TO MY BESTIE!!!! LOVE YA AND CONGRATS!!!! I'M SO SO PROUD!! 
2,0,@kilianvalkhof I read sass differently and was epically disappointed when the site had no sarcasm at all but just geek coding stuff 
3,1,@gauravkanoongo I dont think Ab had a choise but to not accept it! 
4,1,@tamarlevine You are the best carrier of the best links to the best sides ever â€“ I think I`m your fan 
5,1,"@OsoupieO ugh NO, I did not, dumb dream...little kids did, though. I shared it in the Bono random thread "
6,0,My iTunes is being stupid. I HATE when it creates multiple albums out of only one album. My cover flow is all messed up. 
7,0,Can't enter a Lion King contest cuz I'm not a D23 member and it costs money to become a member 
8,1,@LFTA Well I could as Fabulous would say &quot;...put this piece to the left side of ur head n push ur mind right&quot;  Not sure what that does tho
9,1,"@asten77 yes, but I drew a blank.  "
10,1,@debbie2k9 Its sooooo good huh  one of my faveourite movies 
11,1,@AndreMJonesJr  I'll take two please! 
12,0,"@gerbyl LOL you know me so well! its black but there are little pink bits on it and only ONE shop does the pink one, the others are blue! "
13,0,soooo much to do.... wondering how im going to finish all my work  
14,0,@mitchelmusso omg! r songs r awesome! i love them! i cant buy the CD tho...  but i m gonna get them on my ipod! yay! !!!!
15,1,@Zellgado: I agree! Thanks for taking us out there and making it enjoyable 
16,0,@jjblackshear Holy crap!!!!!!!!!!!!!!!!!!!!   
17,1,@davistbell you should move here brother.. I'm almost always in a good mood   http://twitpic.com/3j5b4
18,0,"@DavidArchie ahh! omg, its 4am and i just got back from the hospital cause i got a chemical in my eye!  im glad you made it safely to CT!"
19,1,"@CustomMadness Hehe, thanks "
20,1,"@wendywings ROFLMAO! but but but! I have the greatest ideas  Do as I say, not as I do??"
21,0,"not working tonight, so i ended up eating a bit of dinner with my family, it's in the potty now, but i still feel guilty "
22,0,@nathangielis they aren't scum coz they are a bit sucky atm. poor Raiders 
23,0,"@SteedfastES I still haven't been able to get any  Little things just keep coming up. I plan to get those T-boots, though..."
24,1,@brkaminski Ahh how do you add pictures on here? Teach me Teach me Lol 
25,0,@edwinting *cough* Qais... but yea they really need to work out the insing.com site. The navigation is broken on my browser 
26,0,I want to go home... 
27,0,"@ryan_mitchell Reh... COMEBACK.... Not... Felt good huh?? You loved it, yea??I'm missing it  ... Looking fwd to tomorrow night!!"
28,1,@theclockworkbox: thx for the ff on twitter  â™« http://blip.fm/~8blqi
29,0,Hmm ... my home computer isn't talking to the internet anymore 
30,1,@cassieventura you will either find away or make one ! 
31,1,@JDWRIGHT25 Okays!  Just let me know
32,1,Order of Omega Induction Ceremont at 1130a 
33,0,Up now man I'm still sleepy  gotta do this court thing so I'll tweet wit ya'll tweeps soon
34,1,@toritruslow tea tea tea! i'm sitting at work eating cheesy biscuits  enjoy your thai lesson  xxx
35,0,@ePandu srry yar.. but u see.. papa k saamne mera chalti nhi hai.. 
36,0,@unknownkj ut 
37,0,"@DeniseVlogs I would, but would u just talk to ur YouTube friends? "
38,1,@sueveeyall Lol I never had 8 AM's 
39,1,@dougiemcfly want me to walk with you?  x
40,0,@firestarlight thats pretty cool! i have so long without going to a show i havent gone to a concert in literally a year since lastyear WT 
...
956,0,Boo headache. 
957,1,The lake is a mirror that moves your reflection. Feeding ducks 
958,1,@LynseyNoel: Agreed! 
959,0,"@Harsayis oh, it died several weeks agio and I didn't get around to replacing it until now "
960,0," forgot XBL was off today, was about to check to see if a game was on XBLA that I wanted to buy  oh well, maybe tomorrow"
961,0,I'm getting pwned by Rhythmic reading exercises in compound meters. 
962,0,Computer assembled. Win7 installed. Works well so far. Can't get wireless network going with airport card in mac. No internets on new pc 
963,0,@trniii ok im not sad anymore...that leads me to think about what else i say to you that you dont pay attention too...sad again 
964,0,@SweetEther gotta meet my girl @ the Westin. Her dude has a concert tonight.  Will let u kno but U absolutely HATE Scotts
965,0,sitting at home by myself 
966,0,"@jackieollie Whats wrong ,bud? "
967,0,@Feltan  I'm so sorry for your loss...
968,1,"@Paula_Deen Wow, Hey Paula!  Glad to find you hear on Twitter.  My  5 year son and I are big fans.  I don't have cable and I miss you! "
969,0,@lharp AIR (FlahVM) is _badly_ needed for extra platforms. As real interoperability is very limited now 
970,1,Twitter is so much more fun with new tweeters! It's gonna be a great weekend! 
971,0,@sickestkid awww I haven't watched that in probably 2 years  I think I should because it's amazing.
972,1,@rickyboylamb LoL!!  I still would have got a look at his face just in case!!   
973,1,got my coffee. everythings good. now im waiting for 10 o'clock to roll around so i can buy some denver blink tickets!!  
974,0,I need to detox myself from watching 30 Days of Night. Scared shitless of the dark now. 
975,1,@Zebgora I have so much content it is oozing off my notebook. 
976,1,"had a great night with T-poo, Greg, and Kate.  I LOVE bonfires!"
977,0,@mialuna1 i wanna go to florida! It must be so amazing... I've never been to america 
978,1,awesome night and morning 
979,1,"@taswell l.a.-pic can't be real. sure thing, photoshoped with iPhone. "
980,1,@debbie1573 thats a niiice view!! 
981,0,reading for my exams... 
982,1,@djsirvere ask @petrina_xposed about my executive decision.It's in relation to when you get to melb  but you cant tell anyone lol GANGZ
983,0,"I cuurently have the flu, struggling to get well. "
984,1,@yelyahwilliams Hayley. Hayley. Hayley Orange Hair 
985,1,@vino_delectable  there are 10 towns named Hollywood in the u.s. !!! (looks like geography has no secrets for us 
986,0,Don't leave me alone with L&amp;O!  He's spamming me with DMs about some stupid story that happened last night.
987,0,@n_ng Thanks! I typed upsome of the story on facebook on a comment on my status. Twitter wont let me fit it here  Haha.
988,1,@ninirific awww! thanks 
989,0,is rather bored today and wants to go shopping with no moneyy   &lt; 3 Shanee
990,0,woke up from a nightmare and now I cant sleep 
991,0,The only girls that love me are on ecstasy 
992,1,@rutty_uk Happy Birthday old man! You are SO catching me up 
993,0,just got home from work &amp; steven's.... started a migraine halfway through my shift....  thank goodness for excedrin for migraines!!
994,0,i went to pizza hut with a friend and had pizza... i failed. 
995,1,@oOdolfinaOo I just really hope I have a dry or wet nappy to change as opposed to a stinky muddy one ROFL... Fingers crossed 
996,1,"Yes I'm a twilight fan, suck it up "
997,0,@jilliancyork that there link didn't werk on me iphoner 
998,1,"@DavidArchie ooohh.. yeah..  it's music video is awesome, check it out archie."
999,1,@kodomut: I'm singing it whenever i'm bored. 
1000,0,@auiefab i miss you 
```



#### AI反馈

Reasoned about evaluating sentiment model for 17 seconds >

下面是一份完整的 Python 代码示例，用于加载保存的模型和词典，并对 `Sentiment_Analysis_Dataset_test.csv` 中的测试数据进行评估，计算精度。请将该代码保存为例如 `evaluate.py` 后运行。

保存为`test_YHF.py`，可以替代`sentiment_analysis_YHF.py`

```python
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import re
import numpy as np
from sklearn.metrics import accuracy_score

# 文本预处理：分词和文本转序列
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def text_to_sequence(text, vocab, max_len=100):
    tokens = tokenize(text)
    sequence = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(sequence) < max_len:
        sequence += [vocab["<PAD>"]] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return sequence

# 自定义Dataset，用于加载测试数据
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 兼容 Series 或列表
        text = self.texts.iloc[idx] if isinstance(self.texts, pd.Series) else self.texts[idx]
        label = self.labels.iloc[idx] if isinstance(self.labels, pd.Series) else self.labels[idx]
        sequence = text_to_sequence(text, self.vocab, self.max_len)
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# 定义与训练时一致的模型结构
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)               # [batch_size, seq_len, embed_dim]
        _, (hn, _) = self.lstm(embedded)             # hn: [num_layers, batch_size, hidden_dim]
        hn = hn[-1]                                # 取最后一层隐藏状态 [batch_size, hidden_dim]
        out = self.fc(hn)                          # [batch_size, 1]
        return out.squeeze(1)                      # [batch_size]

def evaluate():
    # 1. 加载测试数据
    test_df = pd.read_csv("Sentiment_Analysis_Dataset_test.csv", on_bad_lines='skip')
    texts = test_df["SentimentText"]
    labels = test_df["Sentiment"]

    # 2. 加载保存的词典
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    # 3. 加载保存的模型
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SentimentClassifier(vocab_size, embed_dim=128, hidden_dim=128, num_layers=1)
    model.load_state_dict(torch.load("sentiment_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # 4. 构建测试数据集和 DataLoader
    test_dataset = SentimentDataset(texts, labels, vocab, max_len=100)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, batch_labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            probs = torch.sigmoid(outputs)             # 将输出转为概率
            preds = (probs > 0.5).float()                # 0.5 阈值进行二值化
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # 计算并打印准确率
    acc = accuracy_score(all_labels, all_preds)
    print("Test Accuracy:", acc)

if __name__ == '__main__':
    evaluate()
```

代码说明

1. **预处理与数据加载**  
   使用 `tokenize` 和 `text_to_sequence` 函数将文本转为固定长度的序列，并通过 `SentimentDataset` 类封装测试数据。

2. **模型加载**  
   定义了与训练时相同的 `SentimentClassifier` 网络结构，然后加载保存的模型参数（`sentiment_model.pth`）和词典（`vocab.pkl`）。

3. **评估过程**  
   利用 DataLoader 遍历测试数据，预测结果经过 sigmoid 激活后采用 0.5 阈值进行二值化，最终使用 scikit-learn 的 `accuracy_score` 计算测试集准确率。

请根据实际情况（如文件路径、Batch Size 等）对代码做相应调整。



#### 评估模型

<mark>得到的精度只有50%</mark>



### 3问AI：双向 LSTM并增加 dropout

你提供的训练和预测过程，跑了3个小时，精度只有50%。能否改进，提升性能，至少超过基线模型的74.5%精度。



#### AI反馈

Reasoned about evaluating sentiment model for 13 seconds >

下面提供一个改进后的版本，主要修改包括：

1. **限制词典规模**  
   通过统计词频后，只保留出现次数最多的 50K 个词，从而减少模型参数和过拟合风险。

2. **改进模型结构**  
   使用双向 LSTM（bidirectional LSTM）并增加 dropout（在 LSTM 层和全连接层后）以增强泛化能力。双向 LSTM 能够同时捕捉前后文信息，比单向 LSTM 表现更好。

3. **训练策略调整**  
   延长训练周期（例如 15 个 epoch），并使用学习率调度器调整学习率。

以下是完整的改进版代码，你可以将其保存为 `train_improved.py` 后运行。训练结束后生成的模型和词典用于评估时有望超过 74.5% 的准确率。

---

```python
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter

# 1. 数据预处理工具
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    return tokens

def build_vocab(texts, min_freq=2, max_vocab_size=50000):
    word_freq = Counter()
    for text in texts:
        tokens = tokenize(text)
        word_freq.update(tokens)
    # 保留出现次数>=min_freq的词
    valid_words = [word for word, freq in word_freq.items() if freq >= min_freq]
    # 按频率排序
    valid_words = sorted(valid_words, key=lambda x: word_freq[x], reverse=True)
    # 截断词典规模
    if len(valid_words) > max_vocab_size - 2:  # 留出 <PAD>, <UNK>
        valid_words = valid_words[:max_vocab_size - 2]
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word in valid_words:
        vocab[word] = len(vocab)
    return vocab

def text_to_sequence(text, vocab, max_len=100):
    tokens = tokenize(text)
    sequence = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(sequence) < max_len:
        sequence += [vocab["<PAD>"]] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return sequence

# 2. 自定义 Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx] if isinstance(self.texts, pd.Series) else self.texts[idx]
        label = self.labels.iloc[idx] if isinstance(self.labels, pd.Series) else self.labels[idx]
        sequence = text_to_sequence(text, self.vocab, self.max_len)
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# 3. 改进后的模型定义：Embedding + 双向 LSTM + Dropout
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.5):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        # 双向输出：hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        # LSTM输出：(output, (hn, cn))
        _, (hn, _) = self.lstm(embedded)  # hn: [num_layers*2, batch_size, hidden_dim]
        # 将正反向最后一层的隐藏状态拼接
        # 若num_layers==1，则hn形状为 [2, batch_size, hidden_dim]
        hn = torch.cat((hn[-2], hn[-1]), dim=1)  # [batch_size, hidden_dim*2]
        out = self.dropout(hn)
        out = self.fc(out)  # [batch_size, 1]
        return out.squeeze(1)

# 4. 训练流程
def TRAIN():
    # 读取数据集
    dataset = pd.read_csv("Sentiment_Analysis_Dataset.csv", on_bad_lines='skip')
    texts = dataset['SentimentText']
    labels = dataset['Sentiment']

    # 构建词典（限制规模）
    vocab = build_vocab(texts, min_freq=2, max_vocab_size=50000)
    print("词典大小:", len(vocab))

    # 划分训练集和验证集
    texts_train, texts_val, labels_train, labels_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
    max_len = 100

    train_dataset = SentimentDataset(texts_train, labels_train, vocab, max_len)
    val_dataset = SentimentDataset(texts_val, labels_val, vocab, max_len)

    batch_size = 128  # 尝试增大 batch_size 以更好利用硬件
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 选择设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for GPU acceleration")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

    # 使用改进后的模型结构
    model = SentimentClassifier(len(vocab), embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.5)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 每3个epoch降低一次学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_acc = 0.0
    best_model_state = None
    num_epochs = 15  # 增加训练周期
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        for sequences, labels_batch in train_loader:
            sequences = sequences.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * sequences.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            train_correct += (preds.float() == labels_batch).sum().item()
            total_train += sequences.size(0)
        train_loss /= total_train
        train_acc = train_correct / total_train

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for sequences, labels_batch in val_loader:
                sequences = sequences.to(device)
                labels_batch = labels_batch.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item() * sequences.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds.float() == labels_batch).sum().item()
                total_val += sequences.size(0)
        val_loss /= total_val
        val_acc = val_correct / total_val

        print(f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()

        scheduler.step()
        print()

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 保存模型和词典
    torch.save(model.state_dict(), "sentiment_model.pth")
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("模型和词典已保存。")
    print("最佳验证准确率:", best_acc)

    return model, vocab

if __name__ == '__main__':
    TRAIN()
```

---

改进说明

- **词典限制**  
  限制词典规模可以减少模型参数，提升训练效率，同时避免罕见词对模型训练造成干扰。

- **双向 LSTM 与 Dropout**  
  双向 LSTM 能够利用前后文信息，通常能提高文本分类性能；同时增加 dropout（在 LSTM 输出和全连接层前）能够有效防止过拟合。

- **训练参数调整**  
  增加训练周期和适当调整学习率调度，使模型有更多机会收敛。

你可以尝试该版本，如果超参数（如 hidden_dim、dropout、num_epochs 等）仍需调整，可以根据验证集效果进一步调优。这样一来，模型有望超过之前 74.5% 的基线精度。



#### 训练改进模型

模型稍微改了下，炼丹参数稍微改了下，第一轮就有0.8了。

训练时间长，我把上面代码的 num_epochs 改为4轮。

```
/Users/hfyan/miniconda3/bin/python /Users/hfyan/Code/sentiment/sentiment_analysis-v2.0/submission/train_YHF_improved.py 
词典大小: 50000
Using MPS device for GPU acceleration
Epoch 1/4
Train Loss: 0.4385  Acc: 0.7951
Val Loss: 0.3947  Acc: 0.8214

Epoch 2/4
Train Loss: 0.3795  Acc: 0.8301
Val Loss: 0.3817  Acc: 0.8287

Epoch 3/4
Train Loss: 0.3546  Acc: 0.8436
Val Loss: 0.3812  Acc: 0.8297

Epoch 4/4
Train Loss: 0.3229  Acc: 0.8600
Val Loss: 0.3839  Acc: 0.8313

模型和词典已保存。
最佳验证准确率: 0.8313078236302075
```



![image-20250308231833314](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250308231833314.png)

训练中，GPU使用情况

![image-20250308231905161](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250308231905161.png)



### 4问AI：评估双向 LSTM并增加 dropout

请利用保存的模型和字典，精度已经达到了83.1%，放在 `./训练出的模型和字典/`目录中 ，对`Sentiment_Analysis_Dataset_test.csv`进行评测，得到精度等。给出python代码。



#### AI反馈

下面是一份完整的 Python 代码示例，它将从目录 `./训练出的模型和字典/` 加载已训练好的模型（精度83.1%）和词典，然后读取 `Sentiment_Analysis_Dataset_test.csv` 进行评测，并输出准确率等指标。

请将下面代码保存为例如 `evaluate_trained.py` 后运行。

<mark>改名为`test_YHF_improved.py`</mark>

```python
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import re
from sklearn.metrics import accuracy_score

# --------------------
# 文本预处理函数
# --------------------
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()

def text_to_sequence(text, vocab, max_len=100):
    tokens = tokenize(text)
    sequence = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(sequence) < max_len:
        sequence += [vocab["<PAD>"]] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return sequence

# --------------------
# 自定义 Dataset
# --------------------
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 兼容 Series 或列表
        text = self.texts.iloc[idx] if isinstance(self.texts, pd.Series) else self.texts[idx]
        label = self.labels.iloc[idx] if isinstance(self.labels, pd.Series) else self.labels[idx]
        sequence = text_to_sequence(text, self.vocab, self.max_len)
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.float)

# --------------------
# 模型定义：改进版双向 LSTM 模型（与训练时一致）
# --------------------
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.5):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        # 双向 LSTM 输出拼接后，hidden_dim*2
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        _, (hn, _) = self.lstm(embedded)  # hn shape: [num_layers*2, batch_size, hidden_dim]
        # 拼接最后一层正向和反向的隐藏状态
        hn = torch.cat((hn[-2], hn[-1]), dim=1)  # [batch_size, hidden_dim*2]
        out = self.dropout(hn)
        out = self.fc(out)  # [batch_size, 1]
        return out.squeeze(1)

# --------------------
# 评测函数
# --------------------
def evaluate():
    # 1. 读取测试数据集
    test_df = pd.read_csv("Sentiment_Analysis_Dataset_test.csv", on_bad_lines='skip')
    texts = test_df["SentimentText"]
    labels = test_df["Sentiment"]

    # 2. 加载保存的词典和模型（位于 ./训练出的模型和字典/ 目录中）
    model_dir = "./训练出的模型和字典/"
    with open(model_dir + "vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    # 选择设备：优先使用 MPS 或 CUDA
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for GPU acceleration")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

    # 加载模型
    model = SentimentClassifier(vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.5)
    model.load_state_dict(torch.load(model_dir + "sentiment_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # 3. 构建 DataLoader
    test_dataset = SentimentDataset(texts, labels, vocab, max_len=100)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, batch_labels in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            probs = torch.sigmoid(outputs)  # 转换为概率
            preds = (probs > 0.5).float()     # 二值化，阈值 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    # 4. 计算准确率
    acc = accuracy_score(all_labels, all_preds)
    print("Test Accuracy:", acc)

if __name__ == '__main__':
    evaluate()
```

---

代码说明

1. **预处理和 Dataset 定义**  
   使用 `tokenize` 和 `text_to_sequence` 将测试文本转换为固定长度的数值序列，然后通过自定义的 `SentimentDataset` 进行加载。

2. **模型加载**  
   模型结构与训练时保持一致（改进版双向 LSTM），从 `./训练出的模型和字典/` 目录加载保存的 `vocab.pkl` 和 `sentiment_model.pth`。

3. **评测过程**  
   利用 DataLoader 遍历测试数据，经过模型预测后，采用 sigmoid 和 0.5 阈值进行二值化，最终使用 scikit-learn 的 `accuracy_score` 计算测试集的准确率。

运行后，终端会输出测试集的准确率（例如 83.1% 或更高）。根据需要，你也可以添加更多评测指标。



#### 评估改进模型测试集

改进模型 <mark>sentiment_model.pth, 27MB，vocab.pkl, 662KB</mark>。

```
/Users/hfyan/miniconda3/bin/python /Users/hfyan/Code/sentiment/sentiment_analysis-v2.0/test_YHF_improved.py 
Using MPS device for GPU acceleration
Test Accuracy: 0.862
```



删除之前不好的模型

`训练出的模型和字典`结构

```plain
训练出的模型和字典
├── sentiment_model_0.pth：Embedding+LSTM, 129.1MB
├── vocab_0.pkl：4MB
├── sentiment_model.pth：双向LSTM, 26.7MB
└── vocab.pkl：662KB
```



<mark>0.862</mark>精度，可以下班了。









