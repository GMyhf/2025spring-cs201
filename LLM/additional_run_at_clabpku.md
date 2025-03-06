



# 在云虚拟机安装Anaconda & 运行鸢尾花卉数据分类

Updated 1655 GMT+8 Mar 6 2025

2025 spring, Complied by Hongfei Yan



> 这份文档，是为了在clab.pku上运行 曹以楷 同学的鸢尾花卉📦程序 additional_CAOYikai.zip
>
> https://github.com/GMyhf/2025spring-cs201/blob/main/LLM/additional_CAOYikai.zip
>
> 



## 1. Installing Anaconda Distribution

https://www.anaconda.com/docs/getting-started/anaconda/install#macos-linux-installation

Linux installer

Download the latest version of Anaconda Distribution by opening a terminal and running one of the following commands (depending on your Linux architecture):



因为安装软件包较大，在/mnt/data盘操作

```
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
```

 % Total  % Received % Xferd Average Speed  Time  Time   Time Current

​                 Dload Upload  Total  Spent  Left Speed

100 1051M 100 1051M  0   0 35.0M   0 0:00:29 0:00:29 --:--:-- 33.5M



[rocky@jensen additional]$ du -hl *.sh

1.1G	Anaconda3-2024.10-1-Linux-x86_64.sh



Install Anaconda Distribution by running one of the following commands (depending on your Linux architecture):



```
bash ./Anaconda3-2024.10-1-Linux-x86_64.sh
```



```
Do you accept the license terms? [yes|no]
>>>      yes

Anaconda3 will now be installed into this location:
/home/rocky/anaconda3

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home/rocky/anaconda3] >>> /mnt/data/anaconda3
...
  zope               pkgs/main/linux-64::zope-1.0-py312h06a4308_1 
  zope.interface     pkgs/main/linux-64::zope.interface-5.4.0-py312h5eee18b_0 
  zstandard          pkgs/main/linux-64::zstandard-0.23.0-py312h2c38b39_0 
  zstd               pkgs/main/linux-64::zstd-1.5.6-hc292b87_0 



Downloading and Extracting Packages:

Preparing transaction: done
Executing transaction: done
installation finished.
Do you wish to update your shell profile to automatically initialize conda?
This will activate conda on startup and change the command prompt when activated.
If you'd prefer that conda's base environment not be activated on startup,
   run the following command when conda is activated:

conda config --set auto_activate_base false

You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[no] >>> yes
no change     /mnt/data/anaconda3/condabin/conda
no change     /mnt/data/anaconda3/bin/conda
no change     /mnt/data/anaconda3/bin/conda-env
no change     /mnt/data/anaconda3/bin/activate
no change     /mnt/data/anaconda3/bin/deactivate
no change     /mnt/data/anaconda3/etc/profile.d/conda.sh
no change     /mnt/data/anaconda3/etc/fish/conf.d/conda.fish
no change     /mnt/data/anaconda3/shell/condabin/Conda.psm1
no change     /mnt/data/anaconda3/shell/condabin/conda-hook.ps1
no change     /mnt/data/anaconda3/lib/python3.12/site-packages/xontrib/conda.xsh
no change     /mnt/data/anaconda3/etc/profile.d/conda.csh
modified      /home/rocky/.bashrc

==> For changes to take effect, close and re-open your current shell. <==

Thank you for installing Anaconda3!
```



```
source ~/.bashrc
```







根据 environment.yml 文件创建一个新的 Conda 环境

```
conda env create -f environment.yml
```

Pip subprocess error: ERROR: Could not find a version that satisfies the requirement torch==2.6.0+cpu (from versions: 1.11.0, 1.12.0, 1.12.1, 1.13.0, 1.13.1, 2.0.0, 2.0.1, 2.1.0, 2.1.1, 2.1.2, 2.2.0, 2.2.1, 2.2.2, 2.3.0, 2.3.1, 2.4.0, 2.4.1, 2.5.0, 2.5.1, 2.6.0) ERROR: No matching distribution found for torch==2.6.0+cpu failed CondaEnvException: Pip failed



### 移除特定标签

尝试移除 `+cpu` 标签，直接使用 `torch==2.6.0` 进行安装。修改你的 `environment.yml` 文件中的相关依赖项：

```
dependencies:
  - ...
  - pip:
      - torch==2.6.0
      - torchaudio==2.6.0
      - torchvision==0.21.0
  - ...
```





```
conda env update --file environment.yml --prune
```

Using cached sympy-1.13.1-py3-none-any.whl (6.2 MB)

Using cached threadpoolctl-3.5.0-py3-none-any.whl (18 kB)

Using cached tomli-2.2.1-py3-none-any.whl (14 kB)

Downloading torch-2.6.0-cp310-cp310-manylinux1_x86_64.whl (766.7 MB)

  ━━━━━━━━                 157.3/766.7 MB 48.5 MB/s eta 0:00:13

Pip subprocess error:

ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device

WARNING: There was an error checking the latest version of pip.

failed

CondaEnvException: Pip failed



```
(base) [rocky@jensen additional]$ df -h /tmp
```

Filesystem   Size Used Avail Use% Mounted on

/dev/sda4    39G  39G 409M 99% /



### 检查并清理临时文件夹

Pip 和其他工具可能会使用系统的临时目录来下载文件。你可以检查和清理这些位置：

查看临时目录：
Bash
深色版本

```
df -h /tmp
```

如果 /tmp 目录位于根分区并且空间不足，可以考虑将它移动到有更多空间的分区如 /mnt/data。
创建新的临时目录：

```
mkdir -p /mnt/data/tmp
export TMPDIR=/mnt/data/tmp
```

然后重新尝试安装命令。
使用 Pip 的 --cache-dir 参数指定缓存目录

你可以指定一个不同的目录作为 Pip 的缓存目录，确保这个目录有足够的空间。

在执行 conda env create 或 pip install 命令之前设置环境变量：

```
export PIP_CACHE_DIR=/mnt/data/pip_cache
mkdir -p $PIP_CACHE_DIR
```



```
conda env update --file environment.yml --prune
```

...

Downloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 21.1/21.1 MB 80.0 MB/s eta 0:00:00

Downloading nvidia_nvtx_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (99 kB)

Downloading triton-3.2.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (253.1 MB)

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 253.1/253.1 MB 34.0 MB/s eta 0:00:00

Installing collected packages: triton, pytz, nvidia-cusparselt-cu12, mpmath, tzdata, typing-extensions, tqdm, tomli, threadpoolctl, sympy, six, pyparsing, pycodestyle, pip, pillow, packaging, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, markupsafe, kiwisolver, joblib, fsspec, fonttools, filelock, cycler, scipy, python-dateutil, nvidia-cusparse-cu12, nvidia-cudnn-cu12, jinja2, contourpy, autopep8, scikit-learn, pandas, nvidia-cusolver-cu12, matplotlib, torch, seaborn, torchvision, torchaudio

 Attempting uninstall: pip

  Found existing installation: pip 25.0

  Uninstalling pip-25.0:

   Successfully uninstalled pip-25.0

Successfully installed autopep8-2.3.2 contourpy-1.3.1 cycler-0.12.1 filelock-3.13.1 fonttools-4.56.0 fsspec-2024.6.1 jinja2-3.1.4 joblib-1.4.2 kiwisolver-1.4.8 markupsafe-2.1.5 matplotlib-3.10.1 mpmath-1.3.0 networkx-3.3 numpy-2.1.2 nvidia-cublas-cu12-12.4.5.8 nvidia-cuda-cupti-cu12-12.4.127 nvidia-cuda-nvrtc-cu12-12.4.127 nvidia-cuda-runtime-cu12-12.4.127 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-cusparselt-cu12-0.6.2 nvidia-nccl-cu12-2.21.5 nvidia-nvjitlink-cu12-12.4.127 nvidia-nvtx-cu12-12.4.127 packaging-24.2 pandas-2.2.3 pillow-11.0.0 pip-25.0.1 pycodestyle-2.12.1 pyparsing-3.2.1 python-dateutil-2.9.0.post0 pytz-2025.1 scikit-learn-1.6.1 scipy-1.15.2 seaborn-0.13.2 six-1.17.0 sympy-1.13.1 threadpoolctl-3.5.0 tomli-2.2.1 torch-2.6.0 torchaudio-2.6.0 torchvision-0.21.0 tqdm-4.67.1 triton-3.2.0 typing-extensions-4.12.2 tzdata-2025.1



done

\#

\# To activate this environment, use

\#

\#   $ conda activate cs201

\#

\# To deactivate an active environment, use

\#

\#   $ conda deactivate



(base) [rocky@jensen additional]$ 



## 2.运行鸢尾花卉数据分类

https://github.com/GMyhf/2025spring-cs201/blob/main/LLM/additional_CAOYikai.zip

> https://github.com/Usercyk/CS201/blob/master/assignments/assignment2.md
>





```
$ conda activate cs201
mkdir -p /home/rocky/assignment2/log/

```



![image-20250306135638535](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250306135638535.png)



![image-20250306135650987](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250306135650987.png)



![image-20250306140127471](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250306140127471.png)





### 结果呈现

在 macOS 上，如果你已经安装了 Visual Studio Code (VS Code)，启动它：

#### 通过 Spotlight 搜索启动

1. 点击屏幕右上角的放大镜图标，或者使用快捷键 `Command + Space` 来打开 Spotlight 搜索。
2. 在搜索框中输入“Visual Studio Code”或简称“VS Code”。
3. 当看到应用图标出现在搜索结果中时，点击它即可启动 VS Code。



#### vscode启动后，如何ssh到云虚拟机？

要在 VS Code 启动后通过 SSH 连接到你的云虚拟机，你需要使用 Visual Studio Code 的远程开发功能。具体步骤如下：

### 前提条件

- 确保你的云虚拟机已启用 SSH 服务，并且你有相应的访问权限（例如：用户名和密码或SSH密钥）。
- 在本地计算机上安装了 Visual Studio Code。
- 安装 Remote - SSH 扩展。

### 步骤

1. **打开 VS Code**:
   - 启动你的 Visual Studio Code 应用程序。

2. **安装 Remote - SSH 扩展**:
   - 如果尚未安装 Remote - SSH 扩展，请按 `Ctrl+Shift+X` 打开扩展视图，然后搜索“Remote - SSH”，找到由 Microsoft 提供的扩展并点击“安装”。

3. **配置 SSH 连接**:
   - 按 `F1` 或 `Ctrl+Shift+P` 打开命令面板，然后输入 `Remote-SSH: Add New SSH Host...` 并选择它。
   - 输入你的 SSH 连接字符串，格式通常为 `ssh user@hostname_or_ip`。比如，如果你的云服务器 IP 地址是 `192.168.1.100`，用户名是 `admin`，则输入 `ssh rocky@10.129.242.98`。
   - 选择用来连接的配置文件位置，默认通常是 `~/.ssh/config`，这取决于你的操作系统和个人偏好。
   - 如果需要，指定 SSH 密钥的位置。如果使用密码登录，则在提示时输入密码。

4. **连接到你的云虚拟机**:
   - 再次按 `F1` 或 `Ctrl+Shift+P`，这次选择 `Remote-SSH: Connect to Host...`。
   - 从列表中选择你刚刚添加的主机，然后按照提示完成连接过程。如果使用 SSH 密钥并且设置了密码保护，此时可能需要输入密钥的密码。

5. **开始使用**:
   - 成功连接后，你可以像操作本地项目一样打开文件夹、编辑文件以及运行应用等。

通过这些步骤，你应该能够成功地从 VS Code 连接到你的云虚拟机进行远程开发工作。记得根据自己的实际情况调整配置细节，如 SSH 密钥路径、用户名、IP 地址等。



![image-20250306160819615](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250306160819615.png)



```
$ mv ~/.vscode-server /mnt/data/
ln -s /mnt/data/.vscode-server ~/

```



![image-20250306161738995](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250306161738995.png)





![image-20250306161850640](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250306161850640.png)













