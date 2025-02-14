# 20250218-Week1-大模型启动

Updated 1313 GMT+8 Feb 14 2025

2025 spring, Complied by Hongfei Yan



logs：

> Get up and running with large language models
>
> 计概课程看图灵自传改编的电影《模拟游戏》，数算课程看英伟达的AI科普书《黄仁勋：英伟达之芯》，期末考试双百加油！



# 0 热身题目

## 27300:模型整理

http://cs101.openjudge.cn/practice/27300/

深度学习模型（尤其是大模型）是近两年计算机学术和业界热门的研究方向。每个模型可以用 “模型名称-参数量” 命名，其中参数量的单位会使用两种：M，即百万；B，即十亿。同一个模型通常有多个不同参数的版本。例如，Bert-110M，Bert-340M 分别代表参数量为 1.1 亿和 3.4 亿的 Bert 模型，GPT3-350M，GPT3-1.3B 和 GPT3-175B 分别代表参数量为 3.5亿，13亿和 1750 亿的 GPT3 模型。参数量的数字部分取值在 [1, 1000) 区间（一个 8 亿参数的模型表示为 800M 而非 0.8B，10 亿参数的模型表示为 1B 而非 1000M）。计算机专业的学生小 A 从网上收集了一份模型的列表，他需要将它们按照名称归类排序，并且同一个模型的参数量从小到大排序，生成 “模型名称: 参数量1, 参数量2, ...” 的列表。请你帮他写一个程序实现。

**输入**

第一行为一个正整数 n（n <= 1000），表示有 n 个待整理的模型。

接下来 n 行，每行一个 “模型名称-参数量” 的字符串。模型名称是字母和数字的混合。

**输出**

每行一个 “模型名称: 参数量1, 参数量2, ...” 的字符串，符号均为英文符号，模型名称按字典序排列，参数量按从小到大排序。

样例输入

```
5
GPT-1.3B
Bert-340M
GPT-350M
Bert-110M
GPT-175B
```

样例输出

```
Bert: 110M, 340M
GPT: 350M, 1.3B, 175B
```

提示

tags: string, sort

来源

2023fall zyn



Competitive Programming with Large Reasoning Models, https://arXiv.org/pdf/2502.06807

2025/2/14 13:52，通义千问 不能AC; ChatGPT Reason, DeepThink(R1) 都可以AC。

https://chatgpt.com/ 默认是"Search"，选到"Reason"，可以AC。

```python
def main():
    n = int(input().strip())
    models = {}
    
    for _ in range(n):
        s = input().strip()
        # 按 '-' 分割得到模型名称和参数字符串（如 "1.3B" 或 "110M"）
        model_name, param_str = s.split('-', 1)
        # 数字部分为除去最后一个字符（单位）得到的部分
        num_str = param_str[:-1]
        unit = param_str[-1]
        # 计算实际数值，B 表示十亿，M 表示百万
        value = float(num_str) * (1e9 if unit == 'B' else 1e6)
        # 将模型名称和对应的 (数值, 原始参数字符串) 添加到字典中
        models.setdefault(model_name, []).append((value, param_str))
    
    # 按模型名称字典序输出
    for model_name in sorted(models.keys()):
        # 同一模型下按照数值从小到大排序
        sorted_params = sorted(models[model_name], key=lambda x: x[0])
        # 只取原始参数字符串部分
        params_str = ", ".join(param for _, param in sorted_params)
        print(f"{model_name}: {params_str}")

if __name__ == "__main__":
    main()

```





DeepSeek默认是DeepThink(R1)，能AC。

按照以下步骤来编写Python程序：

1. **读取输入**：首先读取输入的模型数量 `n`，然后读取接下来的 `n` 行，每行包含一个模型名称和参数量的字符串。
2. **解析和存储**：将每个模型名称和参数量解析出来，并存储在一个字典中，字典的键是模型名称，值是一个列表，存储该模型的所有参数量。
3. **排序**：对字典中的模型名称按字典序排序，对每个模型的参数量列表按参数量从小到大排序。
4. **输出**：按照要求的格式输出每个模型及其参数量。

```python
def parse_model(s):
    # 将字符串解析为模型名称和参数量
    parts = s.split('-')
    model_name = parts[0]
    param = parts[1]
    return model_name, param

def sort_params(params):
    # 对参数量进行排序
    # 将参数量转换为统一的单位（M或B）进行比较
    def get_value(param):
        if param[-1] == 'M':
            return float(param[:-1])
        elif param[-1] == 'B':
            return float(param[:-1]) * 1000
        else:
            return 0  # 无效参数
    
    return sorted(params, key=get_value)

def main():
    n = int(input())
    models = {}
    
    # 读取并解析输入
    for _ in range(n):
        s = input().strip()
        model_name, param = parse_model(s)
        if model_name not in models:
            models[model_name] = []
        models[model_name].append(param)
    
    # 对模型名称按字典序排序
    sorted_models = sorted(models.keys())
    
    # 对每个模型的参数量进行排序
    for model in sorted_models:
        sorted_params = sort_params(models[model])
        # 输出结果
        print(f"{model}: {', '.join(sorted_params)}")

if __name__ == "__main__":
    main()
```



# 1 创建云主机

访问 https://clab.pku.edu.cn

> 同时打开入门文档，https://clab.pku.edu.cn/docs/getting-started/introduction
>
> 我是在mac机器操作
>
> 在terminal中，
>
> ls .ssh/id_ed25519.pub



点击”云主机”,点击“创建云主机”按钮

可用域：nova

架构：X86

类别：labs_and_courses, l3,  4CPU, 4.00GiB

启动源：镜像

操作系统：RockyLinux 9.5

> 推荐的镜像是 RockyLinux 9，是一个基于 RHEL 的 Linux 发行版，有着良好的兼容性和稳定性。Ubuntu 24.04.1 和 Ubuntu 20.04 也是非常好的选择，有着良好的社区支持。
>
> 对于新手来说，Ubuntu 或 Linux Mint 可能是最好的起点，而对于寻求最新技术和功能的用户，Fedora 或 Arch Linux 则可能是更好的选择。对于企业级应用，CentOS Stream 或 openSUSE Leap 可以提供所需的支持和稳定性。

从云硬盘启动：是

系统盘：类型SSD，容量40GiB

数据盘：类型SSD，容量60Gib。去掉后面的 随云主机删除



点击页面右下角的“下一步”按钮，进入网络设置

共享网络：pku-new

虚拟网卡：默认值

安全组：默认值



点击页面右下角的“下一步”按钮，进入名称和密钥设置

名称：YouNameOne

登录凭证：默认值

SSH密钥对: 点“创建密钥”，点“导入密钥”，名称：YouNameOne,  公钥：.ssh/id_ed25519.pub 内容贴进来

确认云主机的配置



# 2 连接云主机 

云主机创建完成后，可以点击云主机的名称进入云主机详情页面。在这里可以看到云主机的状态、IP 地址等信息。我的是 10.129.242.98

在terminal中登录云主机

ssh rocky@10.129.242.98

输入yes，回车



## 登陆网关

```python
#!/usr/bin/env python3

import requests
import getpass

# 从命令行获取用户名和密码
username = input("请输入用户名: ")
password = getpass.getpass("请输入密码: ")

url = "https://its4.pku.edu.cn/cas/ITSClient"
payload = {
    'username': username,
    'password': password,
    'iprange': 'free',
    'cmd': 'open'
}
headers = {'Content-type': 'application/x-www-form-urlencoded'}

result = requests.post(url, params=payload, headers=headers)
print(result.text)
```

将程序保存为`login.py`，运行程序，根据提示输入用户名和密码，就可以登陆网关了。

运行程序

```
python login.py
```



## 把60GB数据盘挂上来

此时看不到创建云主机时候设置的容量60Gib的数据盘。

```
$ df -h
```

输出示例：

```
Filesystem      Size  Used Avail Use% Mounted on
devtmpfs        4.0M     0  4.0M   0% /dev
tmpfs           1.8G     0  1.8G   0% /dev/shm
tmpfs           731M  684K  731M   1% /run
efivarfs        256K   19K  233K   8% /sys/firmware/efi/efivars
/dev/sda4        39G  7.4G   32G  19% /
/dev/sda3       936M  257M  680M  28% /boot
/dev/sda2       100M  7.0M   93M   8% /boot/efi
tmpfs           366M     0  366M   0% /run/user/1000
```



使用lsblk检查是否有分区。

```
$ lsblk
```

看到了sdb。输出示例：

```
NAME   MAJ:MIN RM  SIZE RO TYPE MOUNTPOINTS
sda      8:0    0   40G  0 disk 
├─sda1   8:1    0    2M  0 part 
├─sda2   8:2    0  100M  0 part /boot/efi
├─sda3   8:3    0 1000M  0 part /boot
└─sda4   8:4    0 38.9G  0 part /
sdb      8:16   0   60G  0 disk 
sr0     11:0    1  474K  0 rom  
```





### 直接挂载整个磁盘

把 `/dev/sdb` 挂载并访问，下面操作步骤：

#### 步骤 1: 创建文件系统

直接在 `/dev/sdb` 上创建一个文件系统（例如 ext4），而不需要创建分区。

```bash
sudo mkfs.ext4 /dev/sdb
```

注意：此操作会清除磁盘上的所有数据，请确保该磁盘不包含重要数据或者已经备份。

#### 步骤 2: 创建挂载点

选择一个目录作为挂载点，或者创建一个新的目录。

```bash
sudo mkdir -p /mnt/data
```

#### 步骤 3: 挂载磁盘

使用 `mount` 命令将磁盘挂载到指定的挂载点。

```bash
sudo mount /dev/sdb /mnt/data
```

#### 步骤 4: 验证挂载

检查是否成功挂载：

```bash
df -h
```

输出示例：

```
Filesystem      Size  Used Avail Use% Mounted on
...
/dev/sdb         59G   24M   56G   1% /mnt/data
```

#### 步骤 5: 设置开机自动挂载（可选）

为了确保系统重启后自动挂载该磁盘，您需要编辑 `/etc/fstab` 文件。

首先，获取磁盘的 UUID：

```bash
sudo blkid /dev/sdb
```

输出示例：

```
/dev/sdb: UUID="some-unique-id" TYPE="ext4"
```

然后编辑 `/etc/fstab` 文件：

```bash
sudo vi /etc/fstab
```

添加一行如下内容（根据实际情况调整）：

```
UUID=some-unique-id  /mnt/data  ext4  defaults  0  2
```



保存并退出编辑器。

#### 总结

通过上述步骤，您可以直接在 `/dev/sdb` 上创建文件系统并挂载它，而无需创建任何分区。 



# 3 部署大模型&测试写代码

## 安装 ollama

```
curl -fsSL https://ollama.com/install.sh | sh
```

输出示例：

```
>>> Installing ollama to /usr/local
>>> Downloading Linux amd64 bundle
######################################################################## 100.0%
>>> Creating ollama user...
>>> Adding ollama user to render group...
>>> Adding ollama user to video group...
>>> Adding current user to ollama group...
>>> Creating ollama systemd service...
>>> Enabling and starting ollama service...
Created symlink /etc/systemd/system/default.target.wants/ollama.service → /etc/systemd/system/ollama.service.
>>> The Ollama API is now available at 127.0.0.1:11434.
>>> Install complete. Run "ollama" from the command line.
WARNING: No NVIDIA/AMD GPU detected. Ollama will run in CPU-only mode.
```





查看 https://ollama.com 选择相应模型安装

> 基本上4g内存可用的就是llama3 1b和deepseek蒸馏的qwen1.5b的量化版，qwen3b的量化版不知道能不能跑
>
> llama 3先发的8b和70b版本中文不行，后来又发的3.2（包括1b和3b）是做了多语言训练的
>
>  ollama的特点是默认tag是量化版，比如您跑这个就是1b量化的版本，我感觉这样目的是让尽可能多用户至少能把模型跑起来，虽然效果可能差点
>
> ollama run llama3.2:1b-instruct-fp16，是跑原始的1b版本，也能跑起来
>
> 跑的应该是3b量化版本，看来4G内存还是不行



```
[rocky@jensen ~]$ ollama run llama3.2:1b
```

输出示例：

```
pulling manifest 
pulling 74701a8c35f6... 100% ▏ 1.3 GB                         
pulling 966de95ca8a6... 100% ▏ 1.4 KB                         
pulling fcc5a6bec9da... 100% ▏ 7.7 KB                         
pulling a70ff7e570d9... 100% ▏ 6.0 KB                         
pulling 4f659a1e86d7... 100% ▏  485 B                         
verifying sha256 digest 
writing manifest 
success 
```





```
[rocky@jensen ~]$ ollama list
```

输出示例：

```
NAME               ID              SIZE      MODIFIED      
llama3.2:latest    a80c4f17acd5    2.0 GB    2 minutes ago 
```





## 测试写代码

看能否正确给出这个题目代码，但是给出的代码是错误的。 

27300:模型整理, http://cs101.openjudge.cn/practice/27300/

```
$ ollama run llama3.2:1b
```

输出示例：

```
>>> 请给出python: 深度学习模型（尤其是大模型）是近两年计算机学术和业界热门的研究方向。每个模型可以用 “模型名称-参数量” 命名，其中参数量的单位会使用两种：M，即百万；B
... ，即十亿。同一个模型通常有多个不同参数的版本。例如，Bert-110M，Bert-340M 分别代表参数量为 1.1 亿和 3.4 亿的 Bert 模型，GPT3-350M，GPT3-1.3B 和 GPT3-175B 分别代表
... 参数量为 3.5亿，13亿和 1750 亿的 GPT3 模型。参数量的数字部分取值在 [1, 1000) 区间（一个 8 亿参数的模型表示为 800M 而非 0.8B，10 亿参数的模型表示为 1B 而非 1000M
... ）。计算机专业的学生小 A 从网上收集了一份模型的列表，他需要将它们按照名称归类排序，并且同一个模型的参数量从小到大排序，生成 “模型名称: 参数量1, 参数量2, ...” 的
... 列表。请你帮他写一个程序实现。
... 
... 输入
... 第一行为一个正整数 n（n <= 1000），表示有 n 个待整理的模型。
... 
... 接下来 n 行，每行一个 “模型名称-参数量” 的字符串。模型名称是字母和数字的混合。
... 输出
... 每行一个 “模型名称: 参数量1, 参数量2, ...” 的字符串，符号均为英文符号，模型名称按字典序排列，参数量按从小到大排序。
... 样例输入
... 5
... GPT-1.3B
... Bert-340M
... GPT-350M
... Bert-110M
... GPT-175B
... 样例输出
... Bert: 110M, 340M
... GPT: 350M, 1.3B, 175B
... 提示
... tags: string, sort


```





runtime error

```python
import re
from collections import defaultdict

models = {}

def parse_model_name(model):
    name = model.strip()
    params = name.split('-')
    params = [int(param) for param in params]
    model_params = {
        'B': 1e9,
        'M': 1e6
    }
    return (model, tuple(params), dict(model_params))

def parse_line(line):
    match = re.match(r"(\w+)\s*:\s*(\d+\.\d+)", line)
    if match:
        name, params_str = match.groups()
        param_list = [int(param) for param in params_str.split(',')]
        model_name, params_tuple, _ = parse_model_name(name)
        models[model_name] = tuple(sorted(params_tuple, reverse=True))
        return f"{name}: {', '.join(map(str, list(models[model_name])))}"
    else:
        raise ValueError("Invalid line format")

def main():
    n = int(input())
    for _ in range(n):
        line = input().strip()
        print(parse_line(line))

if __name__ == "__main__":
    main()

```



# 参考

基于大模型和知识库构建问答教学工具，https://pku.instructuremedia.com/embed/5741e3a7-ca03-48f6-97f0-3b5e8df42e2a

linux-help，https://pku.instructuremedia.com/embed/06bda1b0-3342-4705-9c77-e279638f1af2

o3斩获IOI金牌冲榜全球TOP 18，自学碾压顶尖程序员！48页技术报告公布，https://mp.weixin.qq.com/s/rHzZqTBhLBrb-FPtHhEYug

> Competitive Programming with Large Reasoning Models, https://arXiv.org/pdf/2502.06807
>
> o3却在无人启发的情况下，通过强化学习中自己摸索出了一些技巧。即o3在推理过程中展现出更具洞察力和深度思考的思维链。对于验证过程较为复杂的问题，o3会采用一种独特的策略：先编写简单的暴力解法，牺牲一定效率来确保正确性，然后将暴力解法的输出与更优化的算法实现进行交叉检查。
>
> ![图片](https://raw.githubusercontent.com/GMyhf/img/main/img/640)


