# 20250218-Week1-虚拟机，Shell&大模型

Updated 0828 GMT+8 Feb 16 2025

2025 spring, Complied by Hongfei Yan



logs：

> Get up and running with large language models
>
> 计概课程看图灵自传改编的电影《模拟游戏》，数算课程看英伟达的AI科普书《黄仁勋：英伟达之芯》，期末考试双百加油！



拥抱大模型，接触大模型，跟上时代的脚步。这是我本地的，70b的也可以跑的，就是机器峰值了，风扇声大，也热。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202502160840413.png" alt="image-20250216084051207" style="zoom:50%;" />





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



# 1虚拟机

## 1.1 创建云主机

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

> Clab PKU 推荐的镜像是 RockyLinux 9，是一个基于 RHEL 的 Linux 发行版，有着良好的兼容性和稳定性。Ubuntu 24.04.1 和 Ubuntu 20.04 也是非常好的选择，有着良好的社区支持。
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



## 1.2 连接云主机 

云主机创建完成后，可以点击云主机的名称进入云主机详情页面。在这里可以看到云主机的状态、IP 地址等信息。我的IP是 `10.129.242.98`。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202502160843962.png" alt="image-20250216084334218" style="zoom:50%;" />

在terminal中登录云主机

```
ssh rocky@10.129.242.98
```

输入yes，回车



### 登陆网关

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



> 要使用Vi编辑器粘贴并保存这段Python程序，并最终执行它，请按照以下步骤操作：
>
> **步骤 1: 打开终端**
>
> 首先，打开你的Linux终端。
>
> **步骤 2: 使用Vi创建新文件**
>
> 在终端中输入以下命令来使用Vi创建一个名为`login.py`的新文件：
> ```bash
> vi login.py
> ```
>
> **步骤 3: 进入插入模式**
>
> 进入Vi编辑器后，按下键盘上的 `i` 键进入插入模式（Insert Mode）。此时，你可以开始编辑文件内容了。
>
> **步骤 4: 粘贴代码**
>
> 将提供的Python代码复制到剪贴板中，然后在Vi编辑器内右键点击选择“粘贴”或者直接使用快捷键 `Ctrl+Shift+V` 来粘贴代码。确保所有代码都被正确地粘贴到了文件中。
>
> **步骤 5: 保存文件**
>
> 完成代码粘贴后，按下 `Esc` 键退出插入模式。然后输入以下命令保存文件并退出Vi编辑器：
> ```
> :wq
> ```
> 这里的 `:` 表示进入命令模式，`w` 是写入（保存）文件，`q` 是退出Vi编辑器。
>
> **步骤 6: 赋予执行权限**
>
> 为了能够运行这个Python脚本，你可能需要给它赋予执行权限。在终端中输入以下命令：
> ```bash
> chmod +x login.py
> ```
>
> **步骤 7: 执行程序**
>
> 最后，在终端中输入以下命令来运行这个Python程序：
> ```bash
> python3 login.py
> ```
> 注意：根据你的系统配置和安装的Python版本，可能需要使用 `python3` 而不是 `python` 来运行脚本。
>
> 现在，根据提示输入用户名和密码，就可以尝试登录网关了。





### 把60GB数据盘挂上来

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





#### 直接挂载整个磁盘

把 `/dev/sdb` 挂载并访问，下面操作步骤：

**步骤 1: 创建文件系统**

直接在 `/dev/sdb` 上创建一个文件系统（例如 ext4），而不需要创建分区。

```bash
sudo mkfs.ext4 /dev/sdb
```

注意：此操作会清除磁盘上的所有数据，请确保该磁盘不包含重要数据或者已经备份。

**步骤 2: 创建挂载点**

选择一个目录作为挂载点，或者创建一个新的目录。

```bash
sudo mkdir -p /mnt/data
```

**步骤 3: 挂载磁盘**

使用 `mount` 命令将磁盘挂载到指定的挂载点。

```bash
sudo mount /dev/sdb /mnt/data
```

**步骤 4: 验证挂载**

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

**步骤 5: 设置开机自动挂载（可选）**

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

**总结**

通过上述步骤，您可以直接在 `/dev/sdb` 上创建文件系统并挂载它，而无需创建任何分区。 



#  2 Linux Shell使用

**定义**：Shell是为用户提供用户界面的软件，通常指命令行界面的解释器。

**功能**：Shell提供文字界面，可以执行程序、编译代码、控制和监测计算机的运行状态。

**重要性**：大多数底层功能的高效执行需要基于Shell，而不是图形界面（GUI）。因此，Shell在计算机基础学习中不可或缺。使用Shell能够帮助你更好地利用你的设备，提高工作效率。

我们主要讨论的是Linux环境下的Bash，也称为BASH (Bourne Again Shell)。在打开BASH后，你会看到命令提示符，它由多个部分组成：用户名、主机名、工作目录和"$"符号作为命令提示符。

在这里是hfyan。打印当前用户还有一种方法，是`whoami`。第二部分是主机名，可以理解为计算机的名字，在这里是jensen。

```
(base) hfyan@HongfeideMac-Studio ~ % ssh rocky@10.129.242.98
Last login: Sat Feb 15 07:50:12 2025 from 162.105.89.132
[rocky@jensen ~]$ whoami
rocky
[rocky@jensen ~]$ pwd
/home/rocky
[rocky@jensen ~]$ 
```

可以使用`pwd` (print working directory) 打印当前工作目录，在这里 `/home/rocky`是工作目录，意思就是我们当前处在这个目录中。其他命令还有 `echo`回显，`cal`是打印今天的日历，`clear`清屏。

## 2.1 快捷键和学习资源

使用快捷键来提升效率，快速编辑命令行内容。

**Ctrl + U**：删除光标前的所有字符。

**Ctrl + K**：删除光标后的所有字符。

**Ctrl + A**：定位到命令的首部。

**Ctrl + E**：定位到命令的尾部。

**Ctrl + R**：在命令历史记录中进行反向搜索。



快捷键可以减少你输入命令的麻烦。设想这样一种情况，键入了一个非常非常长的指令，但是敲到一半的时候，突然发现整个都错了，需要重新写，按退格键或者一直按着，但是这样子删光整一行，可能需要比较长的时间，可以使用快捷键  `ctrl + u`，删除光标下的所有字符一直到行首。与之相对应的 `ctrl + k`，删除光标下的字符直至行尾。

再比如安装软件如果发现 permission denied，因为你不是管理员用户，需要在前面加sudo。使用上下键来定位我们之前输入过的命令，然后 `ctrl + a` 定位到命令的首部，插入sudo，这时候你就可以直接按enter执行了。`ctrl + e` 回到整行的后面。

`trl + r`（reverse search in bash history）是一个非常重要的快捷键。

`ctrl + l` 与`clear`基本等效，但是它前面的东西都清除掉，不会清除当前行输入的命令。



对于学习资源，除了直接在命令后面加上`--help`获取简要信息外，还可以使用man指令查看详细的用户手册。比如说你不知道ls这个指令应该怎么用，可以直接 `ls --help`，更详细的，可以`man ls`。 man命令是“manual”的缩写，用于显示命令的手册页（manual pages），在这样的这个手册界面中，一般可以使用`vi`的指令来去定位浏览，j向下，向上，如果你要在这个手册中搜索一些信息的话，可以先按正斜杠，然后再输入你要搜索的关键词，比如说line，然后按enter，这个时候所有的line都会被高亮，你再按n，也就代表next，就可以慢慢的向下搜索了。对于这些手册，可以用q来退出它，就是quit。

man也好，help也好，都不是非常的易读，也不是能在短时间内能够看完的，所以你就会想太长不看，有这样一个too long didn't read第三方软件。它可以提供简单常用的命令示例，而且只使用一句话来描述。比如说 `tldr ls`，就会看到打印了一些常用的参数组合。比如需要解压缩一个tar文件，参数都会比较长，初学可能都记不太住，`tldr tar`打印一些常见的解压缩以及压缩的指令的用法。

> 在 Linux 的 Shell 中，可以使用以下方法安装 `tldr`（命令行速查工具）：
>
> **方法 ：使用 npm 安装（推荐）**
>
> `npm` 是 Node.js 的包管理器，如果尚未安装，可以先安装 Node.js：
>
> ```
> sudo dnf install nodejs                         # Fedora
> 
> ```
>
> 然后安装 `tldr`：
>
> ```
> sudo npm install -g tldr
> ```



另外一个比较有用的命令叫info，以咨询一些信息。info是什么？叫做GNU Core Utils，是Linux中一些核心的小工具。

先安装

```
sudo npm install -g info
```

如果你运行 `info` 的话，会看到关于这个小工具的一个手册，里面有非常多有意思的工具。如果大家对shell的用法感兴趣的话可以探索，在网上也是有html版的，当你在这些文档中都找不到好用的信息的时候，可以上网搜索。推荐 https://unix.stackexchange.com/, https://stackoverflow.com/，比较容易能够获得有效的信息。



## 2.2 与文件系统交互

在shell中最基础的命令可能是从文件系统进行交互，也就是资源管理器的功能。Linux的文件系统一般遵循文件系统层次化标准FHS，在该标准中系统中的一切事物都被视为文件，包括目录。通常所有用户的加目录都存储在home下，访问文件系统需要路径的概念，从当前工作目录开始的路径称为相对路径，从根目录开始的路径则称为绝对路径。有了路径的概念后，可以开始文件系统的相关操作。

### 文件操作

- **`ls`**：列出目录内容。list directory contents
  - **`ls -a`**：列出所有文件，包括隐藏文件。隐藏文件一般以dot打头。
  - **`ls -l`**：以长列表形式列出文件详细信息。
- **`cd`**：更改工作目录。
  - **`cd ..`**：返回上一级目录。
  - **`cd ~`**：返回家目录。
  - **`cd -`**：返回上一次访问的目录。

- **`mkdir`**：创建目录。
- **`touch`**：创建空白文件或修改文件时间戳。
- **`rm`**：删除文件。remove directory
  - **`rm -r`**：递归删除目录及其内容。
  - **`rm -f`**：强制删除，不提示确认。
- **`mv`**：移动文件或重命名文件。
- **`cp`**：复制文件。
  - **`cp -r`**：递归复制目录及其内容。

### 文件查看和执行

- **`cat`**：打印文件内容。concatenate
- **`less`**：分页查看文件内容。似于手册的方式来观察比较长的文件。可以使用手册中的那种键盘操作，按q退出
- **`head`**、**`tail`**：查看文件的开头或结尾部分。
- **`./`**：执行文件。执行文件采用.slash再加路径的方式。例如：`./bomb`

有关文件的查看，还有wordcount,find,dif等指令。



### 环境变量

难道不应该使用ls的完整路径来执行吗？系统使用环境变量中的path变量来完成这件事。所谓环境变量，一般是指在操作系统中用来指定运行环境的一些参数，比如临时文件夹位置，系统文件夹位置等等。

`echo $HOST` 打印主机名，`echo $PATH`打印环境变量PATH的值，可以得到一些用冒号分隔的绝对路径。当我们进入一个指令的时候，系统会在这些路径中先顺序搜索可执行文件名，如果找到了就执行。

```
(base) hfyan@Mac-mini 2025spring-cs201 % echo $HOST      
Mac-mini.local
(base) hfyan@Mac-mini 2025spring-cs201 % echo $PATH 
/Users/hfyan/miniconda3/bin:/Users/hfyan/miniconda3/condabin:/usr/local/bin:/System/Cryptexes/App/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/local/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/bin:/var/run/com.apple.security.cryptexd/codex.system/bootstrap/usr/appleinternal/bin:/Library/Apple/usr/bin:/Applications/VMware Fusion.app/Contents/Public:/opt/homebrew/bin
(base) hfyan@Mac-mini 2025spring-cs201 % 
```

使用export这样的指令来改写环境变量

- **`export`**：设置环境变量。
- **`which`**：查找命令的路径。

### 文件权限

观察 `ls -l`的输出，由十位组成，第一位呢表示是否目录，剩下三位为一组，分别代表用户，用户组和其他人对这个文件的权限。文件权限包括读、写、执行，分别用rwx来表示，如果是一个短横线则表示没有这个权限。对于目录来说，执行权限就是搜索。

```
(base) hfyan@Mac-mini 2025spring-cs201 % ls -l
total 2296
-rw-r--r--@  1 hfyan  staff   24135 Feb 16 08:14 20250218_running_LLMs.md
-rw-r--r--   1 hfyan  staff   51196 Feb 16 06:23 AI_accepted_locally.md
-rw-r--r--   1 hfyan  staff    4720 Feb 14 00:09 DSA-B_course_introduction.md
-rw-r--r--   1 hfyan  staff   91550 Feb 14 00:09 KMP-SegmentTree-BIT-BinarySearch-radixSort-Retrieval.md
-rw-r--r--   1 hfyan  staff   10054 Feb 14 00:09 README.md
drwxr-xr-x  12 hfyan  staff     384 Feb 16 06:23 cheatsheet
drwxr-xr-x  11 hfyan  staff     352 Feb  6 09:18 code
-rw-r--r--   1 hfyan  staff  156593 Feb 14 00:09 coding_exam_DSA-B.md
drwxr-xr-x  10 hfyan  staff     320 Jan 22 00:24 machine_learning
drwxr-xr-x   4 hfyan  staff     128 Jan 23 08:48 other
-rw-r--r--   1 hfyan  staff   12001 Feb 16 06:23 pre_problem_list_2025spring.md
-rw-r--r--   1 hfyan  staff  223513 Jan 21 01:32 winter_week1_OOP_linear_structure.md
-rw-r--r--   1 hfyan  staff  246411 Jan 24 00:39 winter_week2_tree.md
-rw-r--r--   1 hfyan  staff  216616 Feb  6 09:18 winter_week3_graph.md
-rw-r--r--   1 hfyan  staff  115050 Feb 14 00:09 written_exam_DSA-B.md
(base) hfyan@Mac-mini 2025spring-cs201 % 

```



- **`ls -l`**：查看文件权限。
- **`chmod`**：更改文件权限。
  - **`chmod u+x`**：给用户添加执行权限。

### 文件打包和压缩

谈到权限我们顺道就可以说到tar这个指令，常会用到tar格式的存档文件，它和zip甚至win rar的区别是它可以保留linux中的文件权限。tar是tape archive的缩写，就是在备份文件的时候常会用到磁带机。用tar打包之后一般会再进行压缩，扩展名为tar.gz指经过gzip算法压缩，而tar.bz2则是经过bzip2算法压缩。

- **`tar`**：打包文件。
  - **`tar czvf`**：打包并压缩文件。
  - **`tar xzvf`**：解压文件。

### 文件重定向和管道

- **`>`**：重定向输出到文件。
- **`>>`**：追加输出到文件。
- **`2>`**：重定向错误输出。
- **`|`**：管道，将前一个命令的输出作为后一个命令的输入。



我们介绍了Shell的基本概念、常用命令、快捷键、文件操作、环境变量、文件权限、打包压缩以及重定向和管道。



# 3 大模型安装和测试

https://www.ollama.com 是字符界面。图像界面可以用，https://lmstudio.ai

我的机器是Mac Studio。





<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202502160845440.png" alt="image-20250216084507879" style="zoom:50%;" />

Apple M1 Ultra 是 Apple 芯片系列中的一员，专为高性能需求设计，特别是在 Mac Studio 等设备中使用。M1 Ultra 的配置包括了中央处理器（CPU）、图形处理器（GPU）以及统一内存架构（Unified Memory Architecture, UMA），其中统一内存可供 CPU、GPU 以及其他组件共享。

**关于 GPU 内存**

在 M1 Ultra 中， 64GB 内存实际上是整个系统共享的统一内存容量，这意味着这64GB内存是由CPU、GPU及其他组件共同使用的，而不是专门分配给GPU的独立内存。

- **统一内存架构**：Apple 的设计理念是通过统一内存架构来提升性能和效率。这种架构允许 GPU 和 CPU 访问相同的内存池，减少了数据复制的需求，并且可以更灵活地根据需要分配内存资源。

- **M1 Ultra 的 GPU 资源**：M1 Ultra 配备了一个强大的 48 核心 GPU。尽管没有“专用”的 GPU 显存，但其可以从整个 64GB 统一内存中获取所需的工作内存。这对于许多图形密集型应用来说是非常有利的，因为它避免了传统显存与主存之间可能存在的瓶颈。

因此，当看到 M1 Ultra 配置为 64GB 内存时，实际上是指整个系统的统一内存大小，而 GPU 可以利用这部分内存中的任何部分作为其工作内存，具体取决于当前运行的应用程序及其对资源的需求情况。这种设计极大地提高了灵活性和性能表现，尤其是在处理复杂图形任务或多任务处理场景下。 

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202502160845741.png" alt="image-20250216084526735" style="zoom:50%;" />



## 3.1 安装lm studio及测试

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202502160842762.png" alt="image-20250216084242045" style="zoom: 33%;" />

装好后，图像界面的本地LLM。

![image-20250216084933939](https://raw.githubusercontent.com/GMyhf/img/main/img/202502160849312.png)

可以本地做编程题目。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202502160850940.png" alt="image-20250216085020115" style="zoom:50%;" />



![image-20250216085036709](https://raw.githubusercontent.com/GMyhf/img/main/img/202502160850429.png)



再比如：

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202502160851393.png" alt="image-20250216085110235" style="zoom:50%;" />



![image-20250216085123730](https://raw.githubusercontent.com/GMyhf/img/main/img/202502160851001.png)



模型是通用的，不用GUI，也可以命令行运行。用 https://github.com/ggml-org/llama.cpp

![image-20250216085617901](https://raw.githubusercontent.com/GMyhf/img/main/img/202502160856958.png)



## 3.2安装 ollama及测试

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202502160842974.png" alt="image-20250216084224586" style="zoom: 33%;" />



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

> 基本上4g内存可用的就是llama3 1b和deepseek蒸馏的qwen1.5b的量化版，3b的量化版还是不行
>
> llama 3先发的8b和70b版本中文不行，后来又发的3.2（包括1b和3b）是做了多语言训练的
>
>  ollama的特点是默认tag是量化版，感觉这样目的是让尽可能多用户至少能把模型跑起来，虽然效果可能差点
>
> ollama run llama3.2:1b-instruct-fp16，是跑原始的1b版本，也能跑起来
>



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





**测试写代码**

看能否正确给出这个题目代码，但是1b模型给出的代码是错误的。 

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



https://browser.geekbench.com 有Geekbench AI测试

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202502160849147.png" alt="image-20250216084859159" style="zoom:50%;" />



![image-20250216083942943](https://raw.githubusercontent.com/GMyhf/img/main/img/202502160839008.png)



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202502160848529.png" alt="image-20250216084836890" style="zoom: 33%;" />



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202502160844411.png" alt="image-20250216084415980" style="zoom:50%;" />
