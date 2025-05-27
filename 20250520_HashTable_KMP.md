# Hash Table & KMP

Updated 1122 GMT+8 May 27, 2025

2025 spring, Complied by Hongfei Yan



# 一、散列表的查找

## 1.1 散列表的基本概念

> 参考：数据结构（C语言版 第2版） (严蔚敏) ，第7章 查找

前面讨论了基于线性结构、树表结构的查找方法，这类查找方法都是以关键字的比较为基础的。

> 线性表是一种具有相同数据类型的有限序列，其特点是每个元素都有唯一的直接前驱和直接后继。换句话说，线性表中的元素之间存在明确的线性关系，每个元素都与其前后相邻的元素相关联。
>
> 线性结构是数据结构中的一种基本结构，它的特点是数据元素之间存在一对一的关系，即除了第一个元素和最后一个元素以外，其他每个元素都有且仅有一个直接前驱和一个直接后继。线性结构包括线性表、栈、队列和串等。
>
> 因此，线性表是线性结构的一种具体实现，它是一种最简单和最常见的线性结构。

在查找过程中只考虑各元素关键字之间的相对大小，记录在存储结构中的位置和其关键字无直接关系，其查找时间与表的长度有关，特别是当结点个数很多时，查找时要大量地与无效结点的关键字进行比较，致使查找速度很慢。如果能在<mark>元素的存储位置和其关键字之间建立某种直接关系</mark>，那么在进行查找时，就无需做比较或做很少次的比较，按照这种关系直接由关键字找到相应的记录。这就是<mark>散列查找法（Hash Search）</mark>的思想，它通过对元素的关键字值进行某种运算，直接求出元素的地址，即使用关键字到地址的直接转换方法，而不需要反复比较。因此，散列查找法又叫杂凑法或散列法。

下面给出散列法中常用的几个术语。

(1) **散列函数和散列地址**：在记录的存储位置p和其关键字 key 之间建立一个确定的对应关系 H，使 `p = H(key)`，称这个对应关系H为散列函数，p为散列地址。

(2) **散列表**：一个有限连续的地址空间，用以存储按散列函数计算得到相应散列地址的数据记录。通常散列表的存储空间是一个一维数组，散列地址是数组的下标。

(3) **冲突和同义词**：对不同的关键字可能得到同一散列地址,即 `key1≠key2`,而 `H(key1) = H(key2)` 这种现象称为<mark>冲突</mark>。具有相同函数值的关键字对该散列函数来说称作同义词，key1与 key2 互称为<mark>同义词</mark>。



例如，在Python语言中，可以针对给定的关键字集合建立一个散列表。假设有一个关键字集合为`S1`，其中包括关键字`main`, `int`, `float`, `while`, `return`, `break`, `switch`, `case`, `do`。为了构建散列表，可以定义一个长度为26的散列表`HT`，其中每个元素是一个长度为8的字符数组。假设我们采用散列函数`H(key)`，该函数将关键字`key`中的第一个字母转换为字母表`{a,b,…,z}`中的序号（序号范围为0~25），即`H(key) = ord(key[0]) - ord('a')`。根据此散列函数构造的散列表`HT`如下所示：

```python
HT = [['' for _ in range(8)] for _ in range(26)]
```

其中，假设关键字`key`的类型为长度为8的字符数组。根据给定的关键字集合和散列函数，可以将关键字插入到相应的散列表位置。



表1

| 0    | 1     | 2    | 3    | 4    | 5     | ...  | 8    | ...  | 12   | ...  | 17     | 18     | ...  | 22    | ...  | 25   |
| ---- | ----- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ------ | ------ | ---- | ----- | ---- | ---- |
|      | break | case | do   |      | float |      | int  |      | main |      | return | switch |      | while |      |      |



假设关键字集合扩充为:

S2 = S1 + {short, default, double, static, for, struct}

如果散列函数不变，新加人的七个关键字经过计算得到：`H(short)=H(static)=H(struct)=18`，`H(default)=H(double)=3`，`H(for)=5`，而 18、3 和5这几个位置均已存放相应的关键字，这就发生了冲突现象，其中 switch、short、static 和 struct 称为同义词；float 和 for 称为同义词；do、default 和 double 称为同义词。

集合S2中的关键字仅有 15 个，仔细分析这 15个关键字的特性，应该不难构造一个散列函数避免冲突。但在实际应用中，理想化的、不产生冲突的散列函数极少存在，这是因为<mark>通常散列表中关键字的取值集合远远大于表空间的地址集</mark>。例如，高级语言的编译程序要对源程序中的标识符建立一张符号表进行管理，多数都采取散列表。在设定散列函数时，考虑的查找关键字集合应包含所有可能产生的关键字，不同的源程序中使用的标识符一般也不相同，如果此语言规定标识符为长度不超过8的、字母开头的字母数字串，字母区分大小写，则标识符取值集合的大小为:
$C_{52}^1 \times C_{62}^7 \times 7! = 1.09 \times 10^{12}$

而一个源程序中出现的标识符是有限的,所以编译程序将散列表的长度设为 1000 足矣。于是要将多达 $10^{12}$个可能的标识符映射到有限的地址上，难免产生冲突。通常，<mark>散列函数是一个多对一的映射，所以冲突是不可避免的</mark>，只能通过选择一个“好”的散列函数使得在一定程度上减少冲突。而一旦发生冲突，就必须采取相应措施及时予以解决。
综上所述，散列查找法主要研究以下两方面的问题:

(1) 如何构造散列函数；
(2) 如何处理冲突。



## 1.2 散列函数的构造方法

构造散列函数的方法很多，一般来说，应根据具体问题选用不同的散列函数，通常要考虑以下因素:

(1) 散列表的长度；
(2) 关键字的长度；
(3) 关键字的分布情况；
(4) 计算散列函数所需的时间；
(5) 记录的查找频率。

构造一个“好”的散列函数应遵循以下两条原则：(1) 函数计算要简单，每一关键字只能有一个散列地址与之对应；(2) 函数的值域需在表长的范围内，计算出的散列地址的分布应均匀，尽可能减少冲突。下面介绍构造散列函数的几种常用方法。

### 1.2.1 数字分析法

如果事先知道关键字集合，且每个关键字的位数比散列表的地址码位数多，每个关键字由n位数组成，如`k1k2,…kn`，则可以从关键字中提取数字分布比较均匀的若干位作为散列地址。

例如，有 80个记录，其关键字为8位十进制数。假设散列表的表长为100，则可取两位十进制数组成散列地址，选取的原则是分析这80个关键字，使得到的散列地址尽量避免产生冲突。假设这 80个关键字中的一部分如下所列:

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240331231703615.png" alt="image-20240331231703615" style="zoom:50%;" />

对关键字全体的分析中可以发现：第①、②位都是“81”，第③位只可能取3或 4，第⑧位可能取 2、5或7，因此这4位都不可取。由于中间的4位可看成是近乎随机的，因此可取其中任意两位，或取其中两位与另外两位的叠加求和后舍去进位作为散列地址。

数字分析法的适用情况：事先必须明确知道所有的关键字每一位上各种数字的分布情况。

在实际应用中，例如，同一出版社出版的所有图书，其ISBN号的前几位都是相同的，因此，若数据表只包含同一出版社的图书，构造散列函数时可以利用这种数字分析排除ISBN 号的前几位数字。



### 1.2.2 平方取中法

通常在选定散列函数时不一定能知道关键字的全部情况，取其中哪几位也不一定合适，而一个数平方后的中间几位数和数的每一位都相关，如果取关键字平方后的中间几位或其组合作为散列地址，则使随机分布的关键字得到的散列地址也是随机的，具体所取的位数由表长决定。<mark>平方取中法是一种较常用的构造散列函数的方法</mark>。

例如，为源程序中的标识符建立一个散列表，假设标识符为字母开头的字母数字串。假设人为约定每个标识的内部编码规则如下：把字母在字母表中的位置序号作为该字母的内部编码，如 I 的内部编码为 09，D 的内部编码为 04，A 的内部编码为 01。数字直接用其自身作为内部编码，如 1的内部编码为 01，2 的内部编码为 02。根据以上编码规则，可知“IDA1”的内部编码为09040101，同理可以得到“IDB2”、“XID3”和“YID4”的内部编码。之后分别对内部编码进行平方运算，再取出第7位到第9位作为其相应标识符的散列地址，如表 2所示。

表2 标识符及其散列地址

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240331232054937.png" alt="image-20240331232054937" style="zoom: 67%;" />



### 1.2.3 折叠法将

关键字分割成位数相同的几部分（最后一部分的位数可以不同），然后取这几部分的叠加和（舍去进位）作为散列地址，这种方法称为<mark>折叠法</mark>。根据数位叠加的方式，可以把折叠法分为移位叠加和边界叠加两种。移位叠加是将分割后每一部分的最低位对齐，然后相加；边界叠加是将两个相邻的部分沿边界来回折叠，然后对齐相加。

例如，当散列表长为 1000 时，关键字`key=45387765213`，从左到右按3 位数一段分割，可以得到 4个部分:453、877、652、13。分别采用移位叠加和边界叠加，求得散列地址为 995 和914，如图 1 所示。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240331232407042.png" alt="image-20240331232407042" style="zoom:50%;" />



<center>图 1由折叠法求得散列地址</center>



<mark>折叠法的适用情况</mark>：适合于散列地址的位数较少，而关键字的位数较多，且难于直接从关键字中找到取值较分散的几位。



### 1.2.4 除留余数法

假设散列表表长为 m，选择一个不大于m 的数p，用p去除关键字，除后所得余数为散列地址，即
H(key) = key%p

这个方法的关键是选取适当的p，一般情况下，可以<mark>选p为小于表长的最大质数</mark>。例如，表长m=100，可取p=97。

除留余数法计算简单，适用范围非常广，是最常用的构造散列函数的方法。它不仅可以对关键字直接取模，也可在折叠、平方取中等运算之后取模，这样能够保证散列地址一定落在散列表的地址空间中。



### 1.2.5 总结哈希函数的选取

取自 刘汝家、黄亮《算法艺术与信息学竞赛》2004年，P96。

对于数值来说：

1）直接取余数（一般选取的除数，最好是个质数，这样冲突少些）。<mark>容易产生分布不均匀的情况</mark>。

2）平方取中法：即计算关键值平方，再取中间r位形成一个大小为2^r的表。好很多，因为几乎所有位都对结果产生了影响。但是它的<mark>计算量大，一般也较少使用</mark>。



对于字符串，常用方法有：

1）折叠法：即把所有字符的ASCII码加起来。

2）采用ELFhash函数，即（它用于UNIX的“可执行链接格式，ELF”中）。这是一个有用的HASH函数，它对长短字符串都很有效。推荐把它作为字符串的HASH函数。

下面是基于ELFhash算法的字符串哈希函数的Python代码示例：

```python
def ELFhash(string):
    hash_value = 0
    x = 0

    for char in string:
        hash_value = (hash_value << 4) + ord(char)
        x = hash_value & 0xF0000000

        if x != 0:
            hash_value ^= (x >> 24)
        
        hash_value &= ~x
    
    return hash_value

# 测试
string = "Hello World"
print("Hash value of '{}' is: {}".format(string, ELFhash(string)))

# output: Hash value of 'Hello World' is: 186219373
```

在这个代码中，`ELFhash` 函数接受一个字符串作为参数，并计算其哈希值。它使用了ELFhash算法，通过遍历字符串的每个字符，将其ASCII码值加入到哈希值中，并进行一系列位运算来得到最终的哈希值。

> 这段代码定义并实现了一个经典的哈希函数：**ELF Hash（Extended Linear Feedback Hash）**，主要用于将字符串映射为整数哈希值。ELF Hash 最初用于 Unix 的 ELF 格式，也出现在很多编译器或链接器中。以下是详细解读：
>
> ------
>
> **初始化变量**
>
> ```python
> hash_value = 0
> x = 0
> ```
>
> - `hash_value`: 存储当前计算的哈希值。
> - `x`: 用于临时存储高位掩码，用来防止哈希溢出。
>
> ------
>
> **主循环**
>
> ```python
> for char in string:
>     hash_value = (hash_value << 4) + ord(char)
> ```
>
> - 每次迭代，将哈希值左移4位（相当于乘以16），并加上当前字符的 ASCII 值（`ord(char)`）。
> - 这种方式<mark>让哈希值对字符的顺序和内容都很敏感</mark>。
>
> ```python
>     x = hash_value & 0xF0000000
> ```
>
> - 提取哈希值的高4位（前4个十六进制位），用于判断是否存在溢出风险。
>
> ```python
>     if x != 0:
>         hash_value ^= (x >> 24)
> ```
>
> - 如果高位 `x` 不为0，右移24位后与当前哈希值异或，<mark>混淆高位对低位的影响，防止模式重复</mark>。
>
> ```python
>     hash_value &= ~x
> ```
>
> - <mark>清除高位，防止哈希值超出范围（限制在28位以内），提高分布的均匀性</mark>。
>
> ------
>
> **总结**
>
> ELF Hash 的特点是：
>
> - 高效；
> - 利用位操作进行散列值的混合；
> - 对字符串内容非常敏感（即使轻微变化也会导致哈希值大变）；
> - 适合用于符号表、哈希表等场景。



## 1.3 处理冲突的方法

选择一个“好”的散列函数可以在一定程度上减少冲突，但在实际应用中，很难完全避免发生冲突，所以选择一个有效的处理冲突的方法是散列法的另一个关键问题。创建散列表和查找散列表都会遇到冲突，两种情况下处理冲突的方法应该一致。下面以创建散列表为例，来说明处理冲突的方法。

处理冲突的方法与散列表本身的组织形式有关。按组织形式的不同，通常分两大类：<mark>开放地址法和链地址法</mark>。



### 1.3.1 开放地址法（闭散列法）

开放地址法的基本思想是：把记录都存储在散列表数组中，当某一记录关键字 key 的初始散列地址 H0 = H(key)发生冲突时，以 H0 为基础，采取合适方法计算得到另一个地址 H1，如果 H1 仍然发生冲突，以 H1 为基础再求下一个地址 H2，若 H2 仍然冲突，再求得 H3。依次类推，直至 Hk 不发生冲突为止，则 Hk 为该记录在表中的散列地址。

这种方法在寻找 ”下一个” 空的散列地址时，<mark>原来的数组空间对所有的元素都是开放的所以称为开放地址法</mark>。通常把寻找 “下一个” 空位的过程称为**探测**，上述方法可用如下公式表示：

$Hi=(H(key) +di)%m	i=1,2,…,k(k≤m-l)$

其中，H(key)为散列函数，m 为散列表表长，d为增量序列。根据d取值的不同，可以分为以下3种探测方法。



(1) 线性探测法
di = 1, 2, 3, …, m-1

这种探测方法可以将散列表假想成一个循环表，发生冲突时，从冲突地址的下一单元顺序寻找空单元，如果到最后一个位置也没找到空单元，则回到表头开始继续查找，直到找到一个空位，就把此元素放入此空位中。如果找不到空位，则说明散列表已满，需要进行溢出处理。



(2) 二次探测法
$d_i =1^2, -1^2, 2^2,-2^2,3^2,.…, +k^2,-k^2 \space (k \le m/2)$​



(3)伪随机探测法

di = 伪随机数序列
例如，散列表的长度为 11，散列函数 H(key)=key%11，假设表中已填有关键字分别为 17、60、29 的记录，如图2(a)所示。现有第四个记录，其关键字为38，由散列函数得到散列地址为 5，产生冲突。

若用线性探测法处理时，得到下一个地址6，仍冲突；再求下一个地址7，仍冲突；直到散列地址为8的位置为“空”时为止，处理冲突的过程结束，38填入散列表中序号为8的位置，如图2(b)所示。

若用二次探测法，散列地址5冲突后，得到下一个地址6，仍冲突；再求得下一个地址 4，无冲突，38填入序号为4的位置，如图 2(c)所示。

若用伪随机探测法，假设产生的伪随机数为9，则计算下一个散列地址为(5+9)%11=3，所以38 填入序号为3 的位置，如图 2(d)所示。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240331233504061.png" alt="image-20240331233504061" style="zoom:50%;" />



<center>图 2 用开放地址法处理冲突时，关键字为38的记录插入前后的散列表</center>

从上述线性探测法处理的过程中可以看到一个现象：当表中i, i+1, i+2位置上已填有记录时，下一个散列地址为i、i+1、i+2和i+3的记录都将填入i+3的位置，这种在处理冲突过程中发生的两个第一个散列地址不同的记录争夺同一个后继散列地址的现象称作“**二次聚集**”(或称作“**堆积**”)，即在处理同义词的冲突过程中又添加了非同义词的冲突。

可以看出，上述三种处理方法各有优缺点。<mark>线性探测法的优点</mark>是：只要散列表未填满，总能找到一个不发生冲突的地址。<mark>缺点</mark>是：会产生“二次聚集”现象。而<mark>二次探测法和伪随机探测法的优点</mark>是：可以避免“二次聚集”现象。<mark>缺点</mark>也很显然：不能保证一定找到不发生冲突的地址。



### 1.3.2 链地址法（开散列法）

链地址法的基本思想是：把具有相同散列地址的记录放在同一个单链表中，称为同义词链表。有 m个散列地址就有m 个单链表，同时用数组 HT[0…m-1]存放各个链表的头指针，凡是散列地址为i的记录都以结点方式插入到以 HT[]为头结点的单链表中。

【例】 已知一组关键字为 (19,14, 23, 1, 68, 20, 84, 27, 55, 11, 10, 79)，设散列函数 `H(key)=key%13`，用链地址法处理冲突，试构造这组关键字的散列表。

由散列函数 H(key)=key %13 得知散列地址的值域为 0~12，故整个散列表有 13 个单链表组成，用数组 HT[0..12]存放各个链表的头指针。如散列地址均为1的同义词 14、1、27、79 构成一个单链表，链表的头指针保存在 HT[1]中，同理，可以构造其他几个单链表，整个散列表的结构如图 3 所示。



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240331233821990.png" alt="image-20240331233821990" style="zoom:50%;" />



<center>图 3 用链地址法处理冲突时的散列表</center>

这种构造方法在具体实现时，依次计算各个关键字的散列地址，然后根据散列地址将关键字插入到相应的链表中。



> 处理散列表冲突的常见方法包括以下几种：
>
> 1. 链地址法（Chaining）：使用链表来处理冲突。每个散列桶（哈希桶）中存储一个链表，具有相同散列值的元素会链接在同一个链表上。
>
> 2. 开放地址法（Open Addressing）：
>    - 线性探测（Linear Probing）：如果发生冲突，就线性地探测下一个可用的槽位，直到找到一个空槽位或者达到散列表的末尾。
>    - 二次探测（Quadratic Probing）：如果发生冲突，就使用二次探测来查找下一个可用的槽位，避免线性探测中的聚集效应。
>    - 双重散列（Double Hashing）：如果发生冲突，就使用第二个散列函数来计算下一个槽位的位置，直到找到一个空槽位或者达到散列表的末尾。
>
> 3. 再散列（Rehashing）：当散列表的**装载因子（load factor）**超过一定阈值时，进行扩容操作，重新调整散列函数和散列桶的数量，以减少冲突的概率。
>
> 4. 建立公共溢出区（Public Overflow Area）：将冲突的元素存储在一个公共的溢出区域，而不是在散列桶中。在进行查找时，需要遍历溢出区域。
>
> 这些方法各有优缺点，适用于不同的应用场景。选择合适的处理冲突方法取决于数据集的特点、散列表的大小以及性能需求。



**双重散列（Double Hashing）**是一种处理散列表冲突的方法，它使用两个散列函数来计算冲突时下一个可用的槽位位置。下面是双重散列的一个示例：

假设有一个散列表，大小为10，使用双重散列来处理冲突。我们定义两个散列函数：

1. 第一个散列函数 `hash1(key)`：将关键字 `key` 转换为散列值，使用一种合适的散列算法，比如取模运算。
2. 第二个散列函数 `hash2(key)`：将关键字 `key` 转换为一个正整数，在本例中，我们使用简单的散列函数 `hash2(key) = 7 - (key % 7)`。

现在，通过以下步骤来插入一个关键字 `key` 到散列表中：

1. 使用第一个散列函数 `hash1(key)` 计算关键字 `key` 的初始散列值 `hash_value = hash1(key)`。
2. 如果散列表中的槽位 `hash_value` 是空的，则将关键字 `key` 插入到该槽位中。
3. 如果槽位 `hash_value` 不为空，表示发生了冲突。在这种情况下，我们**使用第二个散列函数 `hash2(key)` 来计算关键字 `key` 的步长（step）**。
4. 通过计算 `step = hash2(key)`，我们将跳过 `step` 个槽位，继续在散列表中查找下一个槽位。
5. 重复步骤 3 和步骤 4，直到找到一个空槽位，将关键字 `key` 插入到该槽位中。

如果散列表已满而且仍然无法找到空槽位，那么插入操作将失败。

双重散列使用两个散列函数来计算步长，这样可以<mark>避免线性探测中的聚集效应</mark>，提高散列表的性能。每个关键字都有唯一的步长序列，因此它可以在散列表中的不同位置进行探测，减少冲突的可能性。



**笔试例题：**

有一个散列表如下图所示，其散列函数为 `h(key)=key mod 13`，该散列表使用再散列函数`H2(key)=key mod 3` 解决碰撞，问从表中检索出关键码 38 需进行几次比较（ B ）。

| 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 26   | 38   |      |      | 17   |      |      | 33   |      | 48   |      |      | 25   |

A: 1  B: 2  C: 3  D: 4

> 🔢 第一步：计算关键码 38 的初始散列地址
>
> 使用 `h(38) = 38 mod 13 = 12`
>
> 所以要去下标为 **12** 的位置找它。
>
> 查看表中下标 12 的值是 **25**，不是 38。说明发生了碰撞，继续查找。
>
> ---
>
> 🔁 第二步：使用再散列函数解决冲突
>
> 当前哈希地址为 `pos = 12`
>
> 再散列步长为：
>
> ```
> step = H2(38) = 38 mod 3 = 2
> ```
>
> 所以每次探测的公式为：
>
> ```
> pos = (pos + step) % size_of_table
> ```
>
> 散列表长度是 13。
>
> 我们开始探测：
>
> ---
>
> 🔍 探测过程（线性再散列）
>
> 1. 初始 pos = 12  
>    - 表中值为 25 ≠ 38 → 比较 1 次  
>    - 更新 pos = (12 + 2) % 13 = 1
>
> 2. 新 pos = 1  
>    - 表中值为 38 == 38 → 找到！比较 2 次  
>
> ---
>
> ✅ 最终结论：
>
> - 在检索过程中进行了 **2 次比较**。
> - 正确答案是：**B: 2**



## 1.5 程序实现

字符串构建简单的散列函数。针对异序词，这个散列函数总是得到相同的散列值。要弥补这一点，可以用字符位置作为权重因子，

```python
def hash(a_string, table_size):
    sum = 0
    for pos in range(len(a_string)):
        sum = sum + (pos+1) * ord(a_string[pos])

    return sum%table_size

print(hash('abba', 11))
```



使用两个列表创建HashTable类，以此实现映射抽象数据类型。其中，名为slots的列表用于存储键，名为data的列表用于存储值。两个列表中的键与值一一对应。在本节的例子中，散列表的初始大小是11。尽管初始大小可以任意指定，但选用一个素数很重要，这样做可以尽可能地提高冲突处理算法的效率。

hashfunction实现了简单的取余函数。处理冲突时，采用“加1”再散列函数的线性探测法。put函数假设，除非键已经在self.slots中，否则总是可以分配一个空槽。该函数计算初始的散列值，如果对应的槽中已有元素，就循环运行rehash函数，直到遇见一个空槽。如果槽中已有这个键，就用新值替换旧值。

同理，get函数也先计算初始散列值。如果值不在初始散列值对应的槽中，就使用rehash确定下一个位置。注意，第46行确保搜索最终一定能结束，因为不会回到初始槽。如果遇到初始槽，就说明已经检查完所有可能的槽，并且元素必定不存在。

HashTable类的最后两个方法提供了额外的字典功能。<mark>重载`__getitem__`和`__setitem__`，以通过[]进行访问。这意味着创建HashTable类之后，就可以使用熟悉的索引运算符了</mark>。



```python
class HashTable:
    def __init__(self):
        self.size = 11
        self.slots = [None] * self.size
        self.data = [None] * self.size

    def put(self,key,data):
        hashvalue = self.hashfunction(key,len(self.slots))

        if self.slots[hashvalue] == None:
            self.slots[hashvalue] = key
            self.data[hashvalue] = data
        else:
            if self.slots[hashvalue] == key:
                self.data[hashvalue] = data #replace
            else:
                nextslot = self.rehash(hashvalue,len(self.slots))
                while self.slots[nextslot] != None and self.slots[nextslot] != key:
                    nextslot = self.rehash(nextslot,len(self.slots))

                if self.slots[nextslot] == None:
                    self.slots[nextslot] = key
                    self.data[nextslot] = data
                else:
                    self.data[nextslot] = data #replace

    def hashfunction(self,key,size):
        return key%size

    def rehash(self,oldhash,size):
        return (oldhash+1)%size

    def get(self,key):
        startslot = self.hashfunction(key,len(self.slots))

        data = None
        stop = False
        found = False
        position = startslot
        while self.slots[position] != None and not found and not stop:
                if self.slots[position] == key:
                    found = True
                    data = self.data[position]
                else:
                    position=self.rehash(position,len(self.slots))
                    if position == startslot:
                        stop = True
        return data

    def __getitem__(self,key):
        return self.get(key)

    def __setitem__(self,key,data):
        self.put(key,data)


H=HashTable()
H[54]="cat"
H[26]="dog"
H[93]="lion"
H[17]="tiger"
H[77]="bird"
H[31]="cow"
H[44]="goat"
H[55]="pig"
H[20]="chicken"
print(H.slots)
print(H.data)


print(H[20])
print(H[17])

H[20] = 'duck'
print(H[20])

print(H.data)

print(H[99])

"""
[77, 44, 55, 20, 26, 93, 17, None, None, 31, 54]
['bird', 'goat', 'pig', 'chicken', 'dog', 'lion', 'tiger', None, None, 'cow', 'cat']
chicken
tiger
duck
['bird', 'goat', 'pig', 'duck', 'dog', 'lion', 'tiger', None, None, 'cow', 'cat']
None
"""
```



注意，在11个槽中，有9个被占用了。占用率被称作<mark>载荷因子（ load factor）</mark>，记作 λ，定义如下。

 $\lambda = \frac {元素个数}{散列表大小}$ 

在本例中, $\lambda = \frac {9}{11}$.







## 1.6 散列表的查找

在散列表上进行查找的过程和创建散列表的过程基本一致。



## 1.7 编程题目

### 练习17968: 整型关键字的散列映射

http://cs101.openjudge.cn/practice/17968/

给定一系列整型关键字和素数P，用除留余数法定义的散列函数H（key)=key%M，将关键字映射到长度为M的散列表中，用线性探查法解决冲突

**输入**

输入第一行首先给出两个正整数N（N<=1000）和M（>=N的最小素数），分别为待插入的关键字总数以及散列表的长度。
第二行给出N个整型的关键字。数字之间以空格分隔。

**输出**

在一行内输出每个整型关键字的在散列表中的位置。数字间以空格分隔。

样例输入

```
4 5
24 13 66 77
```

样例输出

```
4 3 1 2
```



这个题目的<mark>输入数据可能不是标准形式，特殊处理，整体读入 sys.stdin.read</mark>

```python
def insert_hash_table(keys, M):
    table = [0.5] * M  # 用 0.5 表示空位
    result = []

    for key in keys:
        index = key % M
        i = index

        while True:
            if table[i] == 0.5 or table[i] == key:
                result.append(i)
                table[i] = key
                break
            i = (i + 1) % M

    return result

# 使用标准输入读取数据
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
keys = list(map(int, data[2:2 + N]))

positions = insert_hash_table(keys, M)
print(*positions)

```



### 练习17975: 用二次探查法建立散列表

http://cs101.openjudge.cn/practice/17975/

给定一系列整型关键字和素数P，用除留余数法定义的散列函数H（key)=key%M，将关键字映射到长度为M的散列表中，用二次探查法解决冲突.

本题不涉及删除，且保证表长不小于关键字总数的2倍，即没有插入失败的可能。

**输入**

输入第一行首先给出两个正整数N（N<=1000）和M（一般为>=2N的最小素数），分别为待插入的关键字总数以及散列表的长度。
第二行给出N个整型的关键字。数字之间以空格分隔。

**输出**

在一行内输出每个整型关键字的在散列表中的位置。数字间以空格分隔。

样例输入

```
5 11
24 13 35 15 14
```

样例输出

```
2 3 1 4 7 
```

提示

探查增量序列依次为：$1^2，-1^2，2^2 ，-2^2，....,^2$表示平方



<mark>需要用这样接收数据。因为输入数据可能分行了，不是题面描述的形式。OJ上面有的题目是给C++设计的，细节考虑不周全。</mark>

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]
```



```python
# 2200015507 王一粟
# n, m = map(int, input().split())
# num_list = [int(i) for i in input().split()]
import sys
input = sys.stdin.read
data = input().split()
index = 0
n = int(data[index])
index += 1
m = int(data[index])
index += 1
num_list = [int(i) for i in data[index:index+n]]

mylist = [0.5] * m

def generate_result():
    for num in num_list:
        pos = num % m
        current = mylist[pos]
        if current == 0.5 or current == num:
            mylist[pos] = num
            yield pos
        else:
            sign = 1
            cnt = 1
            while True:
                now = pos + sign * (cnt ** 2)
                current = mylist[now % m]
                if current == 0.5 or current == num:
                    mylist[now % m] = num
                    yield now % m
                    break
                sign *= -1
                if sign == 1:
                    cnt += 1

result = generate_result()
print(*result)
```







# 二、KMP



https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm

在计算机科学中，**Knuth-Morris-Pratt 算法**（简称 **KMP 算法**）是一种字符串查找算法。该算法通过观察到这样一个关键点来查找主文本字符串 `S` 中是否存在一个“单词”或子字符串 `W`：<mark>当发生字符不匹配时，单词 `W` 本身已经包含了足够的信息，可以确定下一次可能的匹配位置，从而跳过对先前已匹配字符的重复检查</mark>。

> KMP 利用已经匹配的信息避免无谓的重复比较，实现了 O(n + m) 的字符串匹配效率。

该算法最初由 James H. Morris 构思，并在几周后被 Donald Knuth 从自动机理论的角度独立发现。Morris 和 Vaughan Pratt 于 1970 年发表了一篇技术报告。三人于 1977 年联合发表了这一算法。与此同时，在 1969 年，Matiyasevich 在研究一个基于二元字母表的字符串模式匹配识别问题时，利用二维图灵机设计出了一个类似的算法。这是<mark>第一个实现线性时间复杂度的字符串匹配算法</mark>。

> In computer science, the **Knuth–Morris–Pratt algorithm** (or **KMP algorithm**) is a string-searching algorithm that searches for occurrences of a "word" `W` within a main "text string" `S` by employing the observation that when a mismatch occurs, the word itself embodies sufficient information to determine where the next match could begin, thus bypassing re-examination of previously matched characters.
>
> The algorithm was conceived by James H. Morris and independently discovered by Donald Knuth "a few weeks later" from automata theory. Morris and Vaughan Pratt published a technical report in 1970. The three also published the algorithm jointly in 1977. Independently, in 1969, Matiyasevich discovered a similar algorithm, coded by a two-dimensional Turing machine, while studying a string-pattern-matching recognition problem over a binary alphabet. This was the first linear-time algorithm for string matching.



KMP Algorithm for Pattern Searching

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231107135044605.png" alt="image-20231107135044605" style="zoom: 33%;" />



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231107135333487.png" alt="image-20231107135333487" style="zoom:50%;" />



**Generative AI is experimental**. Info quality may vary.

Knuth-Morris-Pratt（KMP）算法是**一种用于在文本字符串中查找单词的计算机科学算法**。该算法从左到右依次比较字符。

当出现字符不匹配时，算法会使用一个预处理表（称为“前缀表”）来跳过不必要的字符比较。

**KMP 算法的工作原理**

- 该算法会在<mark>模式串</mark>中寻找被称为 <mark>LPS（Longest Prefix which is also Suffix，最长前缀后缀）</mark>的重复子串，并将这些 LPS 信息存储在一个数组中。
- 算法从左到右逐个比较字符。
- 当发生不匹配时，算法使用一个预处理好的表（称为“前缀表”）来跳过字符比较。
- 算法预先计算一个前缀函数，帮助确定每次发生不匹配时可以在<mark>模式串</mark>中跳过多少字符。
- 相比暴力搜索方法，KMP 算法利用先前比较的信息，避免了不必要的字符比较，从而提高了效率。

**KMP 算法的优势**

- KMP 算法可以高效地帮助你在大量文本中找到特定的模式串。
- KMP 算法可以使你的文本编辑任务更快、更高效。
- KMP 算法保证了**100% 的可靠性**。

> The Knuth–Morris–Pratt (KMP) algorithm is **a computer science algorithm that searches for words in a text string**. The algorithm compares characters from left to right. 
>
> When a mismatch occurs, the algorithm uses a preprocessed table called a "Prefix Table" to skip character comparisons.
>
> How the KMP algorithm works
>
> - The algorithm finds repeated substrings called LPS in the pattern and stores LPS information in an array.
> - The algorithm compares characters from left to right.
> - When a mismatch occurs, the algorithm uses a preprocessed table called a "Prefix Table" to skip character comparisons.
> - The algorithm precomputes a prefix function that helps determine the number of characters to skip in the pattern whenever a mismatch occurs.
> - The algorithm improves upon the brute force method by utilizing information from previous comparisons to avoid unnecessary character comparisons.
>
> Benefits of the KMP algorithm
>
> - The KMP algorithm efficiently helps you find a specific pattern within a large body of text.
> - The KMP algorithm makes your text editing tasks quicker and more efficient.
> - The KMP algorithm guarantees 100% reliability.





**Preprocessing Overview:**

- KMP algorithm preprocesses pat[] and constructs an auxiliary **lps[]** of size **m** (same as the size of the pattern) which is used to skip characters while matching.

- Name **lps** indicates the <mark>longest proper prefix</mark> which is also a suffix. A proper prefix is a prefix with a whole string not allowed. For example, prefixes of “ABC” are “”, “A”, “AB” and “ABC”. Proper prefixes are “”, “A” and “AB”. Suffixes of the string are “”, “C”, “BC”, and “ABC”. 真前缀（proper prefix）是一个串除该串自身外的其他前缀。

- We search for lps in subpatterns. More clearly we ==focus on sub-strings of patterns that are both prefix and suffix==.

- For each sub-pattern pat[0..i] where i = 0 to m-1, lps[i] stores the length of the maximum matching proper prefix which is also a suffix of the sub-pattern pat[0..i].

  > LPS表是一个数组，其中的每个元素表示模式字符串中<mark>当前位置之前</mark>的子串的最长前缀后缀的长度。

>   lps[i] = the longest proper prefix of pat[0..i] which is also a suffix of pat[0..i]. 
>
>   <mark>核心概念：最长前缀后缀（LPS 表）</mark>
>
>   - **LPS（Longest Prefix which is also Suffix）表**：对模式串 `pattern` 的每个前缀子串，记录它的“最长相等前后缀”的长度。
>   - 它的作用是：<mark>**当匹配失败时，指针无需回退主串的位置，只需调整模式串的位置即可继续匹配**</。

**Note:** lps[i] could also be defined as the longest prefix which is also a proper suffix. We need to use it properly in one place to make sure that <mark>the whole substring is not considered</mark>.

Examples of lps[] construction:

> For the pattern “AAAA”, lps[] is [0, 1, 2, 3]
>
> For the pattern “ABCDE”, lps[] is [0, 0, 0, 0, 0]
>
> For the pattern “AABAACAABAA”, lps[] is [0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5]
>
> For the pattern “AAACAAAAAC”, lps[] is [0, 1, 2, 0, 1, 2, 3, 3, 3, 4] 
>
> For the pattern “AAABAAA”, lps[] is [0, 1, 2, 0, 1, 2, 3]



KMP（Knuth-Morris-Pratt）算法是一种利用双指针和动态规划的字符串匹配算法。

```python
""""
compute_lps 函数用于计算模式字符串的LPS表。LPS表是一个数组，
其中的每个元素表示模式字符串中当前位置之前的子串的最长前缀后缀的长度。
该函数使用了两个指针 length 和 i，从模式字符串的第二个字符开始遍历。
"""
def compute_lps(pattern):
    """
    计算pattern字符串的最长前缀后缀（Longest Proper Prefix which is also Suffix）表
    :param pattern: 模式字符串
    :return: lps表
    """

    m = len(pattern)
    lps = [0] * m  # 初始化lps数组
    length = 0  # 当前最长前后缀长度
    for i in range(1, m):  # 注意i从1开始，lps[0]永远是0
        while length > 0 and pattern[i] != pattern[length]:
            length = lps[length - 1]  # 回退到上一个有效前后缀长度
        if pattern[i] == pattern[length]:
            length += 1
        lps[i] = length

    return lps

def kmp_search(text, pattern):
    n = len(text)
    m = len(pattern)
    if m == 0:
        return 0
    lps = compute_lps(pattern)
    matches = []

    # 在 text 中查找 pattern
    j = 0  # 模式串指针
    for i in range(n):  # 主串指针
        while j > 0 and text[i] != pattern[j]:
            j = lps[j - 1]  # 模式串回退
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            matches.append(i - j + 1)  # 匹配成功
            j = lps[j - 1]  # 查找下一个匹配

    return matches


text = "ABABABABCABABABABCABABABABC"
pattern = "ABABCABAB"
index = kmp_search(text, pattern)
print("pos matched：", index)
# pos matched： [4, 13]

```



KMP 是一种利用双指针和动态规划的字符串匹配算法。

- **双指针**：确实存在，一个指针在主串 `text` 上（`i`），另一个在模式串 `pattern` 上（`j`）。
- **动态规划**：广义上讲，LPS 的构造有递推性质，有人将其类比为动态规划的表构建过程（类似于状态转移），但它并不是真正的 DP 算法，只是一个预处理表。

------

✅ 总结

| 项目     | 说明                                      |
| -------- | ----------------------------------------- |
| 主要目标 | 在主串中高效地查找子串                    |
| 核心工具 | LPS 表：用于跳过无效比较                  |
| 优点     | 匹配失败时主串不回退，时间复杂度 O(n + m) |
| 适用场景 | 文本搜索、DNA序列分析、代码查重等         |



## 关于 kmp 算法中 next 数组的周期性质

**🌟 引理：**

对于某一字符串 $S[1∼i]$，在它的 `next[i]` 的候选值中，若存在某一 `next[i]` 使得：

> 注意这个i是从1开始的，写代码通常从0开始。

$i \mod  (i−next[i])=0$

那么：

- $S[1∼(i−next[i])]$ 是 $S[1∼i]$ 的**最小循环元**（最小周期子串）；
- $K= \frac{i}{i−next[i]}$ 是这个循环元在 $S[1∼i]$ 中出现的次数。

------

这个引理揭示了 KMP 算法中 `next` 数组与字符串的**周期性质**之间的联系，是字符串处理中的一个经典思想。我们来详细解释它的含义与推导过程。

🧠 **解释与推导**

KMP 算法中 `next[i]` 表示：在 $S[1∼i]$ 中，最长的**真前缀 = 真后缀**的长度。

设：

- $p=i−next[i]$

也就是说，当前字符串 $S[1∼i]$ 的最长相等前后缀长度是 $next[i]$，剩下的部分（中间不能匹配的部分）长度为 p。

如果 $i \mod  p=0$，也就是 i 是 p 的整数倍，意味着我们可以将长度为 p 的子串重复 $K = \frac{i}{p}$ 次，刚好构成整个 $S[1∼i]$。

这说明：

- S[1∼i] 是由某个长度为 p 的子串重复 K 次构成；
- 即 S[1∼p] 是 S[1∼i] 的**最小循环节（最小周期）**；
- K 是循环次数。

------

✅ **举个例子**

考虑字符串 `ababab`，即 S=a b a b a b

构造其 `next` 数组：

| i    | S[i] | next[i] |
| ---- | ---- | ------- |
| 1    | a    | 0       |
| 2    | b    | 0       |
| 3    | a    | 1       |
| 4    | b    | 2       |
| 5    | a    | 3       |
| 6    | b    | 4       |

对 i=6，有 next[6] = 4，则：

- p = i−next[i] = 6−4=2
- i mod  p = 6 mod  2=0，满足条件
- 所以 `ab`（即 S[1∼2]）是 `ababab` 的最小循环元；
- $K=\frac62=3$，循环了 3 次。

------

🔁 用途：判断字符串是否由某个子串重复构成

基于这个引理，可以用来快速判断一个字符串是否可以由某个子串重复得到。例如：

```python
def is_repeated_pattern(s: str) -> bool:
    n = len(s)
    next = [0] * (n + 1)
    j = 0
    for i in range(2, n + 1):
        while j > 0 and s[j] != s[i-1]:
            j = next[j]
        if s[j] == s[i - 1]:
            j += 1
        next[i] = j

    p = n - next[n]
    return n % p == 0 and n != p

print(is_repeated_pattern("ababab"))  # True
```

------

📌 总结

引理中的结论可以归纳如下：

> 如果一个字符串 $S[1∼i]$ 的 `next[i]` 满足 $i \mod  (i−next[i])==0$，
> 则 $S[1∼(i−next[i])]$ 是它的最小循环节，
> 且重复次数为 $K= \frac{i}{i−next[i]}$。

这种性质在字符串压缩、周期检测、重复匹配等应用中非常重要。



参考：https://www.acwing.com/solution/content/4614/

引理：
对于某一字符串 `S[1～i]`，在它众多的`next[i]`的“候选项”中，如果存在某一个`next[i]`，使得: `i%(i-nex[i])==0`，那么 `S[1～ (i−next[i])]` 可以为 `S[1～i]` 的循环元而` i/(i−next[i])` 即是它的循环次数 K。

证明如下：

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231107111654773.png" alt="image-20231107111654773" style="zoom: 50%;" />

如果在紧挨着之前框选的子串后面再框选一个长度为 m 的小子串(绿色部分)，同样的道理，

可以得到：`S[m～b]=S[b～c]`
又因为：`S[1～m]=S[m～b]`
所以：`S[1～m]=S[m～b]=S[b～c]`



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/5c8ef2df2845d.png" alt="img" style="zoom:75%;" />

如果一直这样框选下去，无限推进，总会有一个尽头。当满足` i % m==0` 时，刚好可以分出 K 个这样的小子串，且形成循环(`K=i/m`)。



### 练习02406: 字符串乘方

KMP, http://cs101.openjudge.cn/practice/02406/

给定两个字符串a和b,我们定义`a*b`为他们的连接。例如，如果a=”abc” 而b=”def”， 则`a*b=”abcdef”`。 如果我们将连接考虑成乘法，一个非负整数的乘方将用一种通常的方式定义：a^0^=””(空字符串)，a^(n+1)^=a*(a^n^)。

**输入**

每一个测试样例是一行可打印的字符作为输入，用s表示。s的长度至少为1，且不会超过一百万。最后的测试样例后面将是一个点号作为一行。

**输出**

对于每一个s，你应该打印最大的n，使得存在一个a，让$s=a^n$

样例输入

```
abcd
aaaa
ababab
.
```

样例输出

```
1
4
3
```

提示: 本问题输入量很大，请用scanf代替cin，从而避免超时。

来源: Waterloo local 2002.07.01



```python
'''
使用KMP算法的部分知识，当字符串的长度能被提取的"base字符串"的长度整除时，
即可判断s可以被表示为a^n的形式，此时的n就是s的长度除以"base字符串"的长度。
'''

import sys

while True:
    s = sys.stdin.readline().strip()
    if s == '.':
        break
    n = len(s)
    next = [0] * len(s)
    j = 0
    for i in range(1, n):
        while j > 0 and s[i] != s[j]:
            j = next[j - 1]
        if s[i] == s[j]:
            j += 1
        next[i] = j
    p = len(s) - next[-1]
    if n % p == 0:
        print(n // p)
    else:
        print(1)
```



### 练习01961: 前缀中的周期

KMP, http://cs101.openjudge.cn/practice/01961/

http://poj.org/problem?id=1961

For each prefix of a given string S with N characters (each character has an ASCII code between 97 and 126, inclusive), we want to know whether the prefix is a periodic string. That is, for each $i \ (2 \le i \le N)$ we want to know the largest K > 1 (if there is one) such that the prefix of S with length i can be written as $A^K$ ,that is A concatenated K times, for some string A. Of course, we also want to know the period K.



一个字符串的前缀是从第一个字符开始的连续若干个字符，例如"abaab"共有5个前缀，分别是a, ab, aba, abaa,  abaab。

我们希望知道一个N位字符串S的前缀是否具有循环节。换言之，对于每一个从头开始的长度为 i （i 大于1）的前缀，是否由重复出现的子串A组成，即 AAA...A （A重复出现K次，K 大于 1）。如果存在，请找出最短的循环节对应的K值（也就是这个前缀串的所有可能重复节中，最大的K值）。

**输入**

输入包括多组测试数据。每组测试数据包括两行。
第一行包括字符串S的长度N（2 <= N <= 1 000 000）。
第二行包括字符串S。
输入数据以只包括一个0的行作为结尾。

**输出**

对于每组测试数据，第一行输出 "Test case #“ 和测试数据的编号。
接下来的每一行，输出前缀长度i和重复次数K，中间用一个空格隔开。前缀长度需要升序排列。
在每组测试数据的最后输出一个空行。

样例输入

```
3
aaa
12
aabaabaabaab
0
```

样例输出

```
Test case #1
2 2
3 3

Test case #2
2 2
6 2
9 3
12 4
```



【POJ1961】period，https://www.cnblogs.com/ve-2021/p/9744139.html

如果一个字符串S是由一个字符串T重复K次构成的，则称T是S的<mark>循环元</mark>。使K出现最大的字符串T称为S的最小循环元，此时的K称为最大循环次数。

现在给定一个长度为N的字符串S，对S的每一个前缀S[1~i]，如果它的最大循环次数大于1，则输出该循环的最小循环元长度和最大循环次数。



题解思路：
1）与自己的前缀进行匹配，与KMP中的next数组的定义相同。next数组的定义是：字符串中以i结尾的子串与该字符串的前缀能匹配的最大长度。
2）将字符串S与自身进行匹配，对于每个前缀，能匹配的条件即是：S[i-next[i]+1 \~ i]与S[1~next[i]]是相等的，并且不存在更大的next满足条件。
3）当i-next[i]能整除i时，S[1 \~ i-next[i]]就是S[1 ~ i]的最小循环元。它的最大循环次数就是i/(i - next[i])。



这是刘汝佳《算法竞赛入门经典训练指南》上的原题（p213），用KMP构造状态转移表。在3.3.2 KMP算法。

```python
'''
gpt
这是一个字符串匹配问题，通常使用KMP算法（Knuth-Morris-Pratt算法）来解决。
使用了 Knuth-Morris-Pratt 算法来寻找字符串的所有前缀，并检查它们是否由重复的子串组成，
如果是的话，就打印出前缀的长度和最大重复次数。
'''

# 得到字符串s的前缀值列表
def kmp_next(s):
  	# kmp算法计算最长相等前后缀
    next = [0] * len(s)
    j = 0
    for i in range(1, len(s)):
        while s[i] != s[j] and j > 0:
            j = next[j - 1]
        if s[i] == s[j]:
            j += 1
        next[i] = j
    return next


def main():
    case = 0
    while True:
        n = int(input().strip())
        if n == 0:
            break
        s = input().strip()
        case += 1
        print("Test case #{}".format(case))
        next = kmp_next(s)
        for i in range(2, len(s) + 1):
            k = i - next[i - 1]		# 可能的重复子串的长度
            if (i % k == 0) and i // k > 1:
                print(i, i // k)
        print()


if __name__ == "__main__":
    main()

```

