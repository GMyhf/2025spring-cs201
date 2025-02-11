# KMP-BinarySearch-radixSort-Retrieval

Updated 2028 GMT+8 Feb 11, 2025

2024 spring, Complied by Hongfei Yan



**Logs：**

created on Apr 6, 2024





# 一、KMP（Knuth-Morris-Pratt）



https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm

In computer science, the **Knuth–Morris–Pratt algorithm** (or **KMP algorithm**) is a string-searching algorithm that searches for occurrences of a "word" `W` within a main "text string" `S` by employing the observation that when a mismatch occurs, the word itself embodies sufficient information to determine where the next match could begin, thus bypassing re-examination of previously matched characters.

The algorithm was conceived by James H. Morris and independently discovered by Donald Knuth "a few weeks later" from automata theory. Morris and Vaughan Pratt published a technical report in 1970. The three also published the algorithm jointly in 1977. Independently, in 1969, Matiyasevich discovered a similar algorithm, coded by a two-dimensional Turing machine, while studying a string-pattern-matching recognition problem over a binary alphabet. This was the first linear-time algorithm for string matching.



KMP Algorithm for Pattern Searching

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231107135044605.png" alt="image-20231107135044605" style="zoom: 33%;" />



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231107135333487.png" alt="image-20231107135333487" style="zoom:50%;" />



**Generative AI is experimental**. Info quality may vary.

The Knuth–Morris–Pratt (KMP) algorithm is **a computer science algorithm that searches for words in a text string**. The algorithm compares characters from left to right. 

When a mismatch occurs, the algorithm uses a preprocessed table called a "Prefix Table" to skip character comparisons.

How the KMP algorithm works

- The algorithm finds repeated substrings called LPS in the pattern and stores LPS information in an array.
- The algorithm compares characters from left to right.
- When a mismatch occurs, the algorithm uses a preprocessed table called a "Prefix Table" to skip character comparisons.
- The algorithm precomputes a prefix function that helps determine the number of characters to skip in the pattern whenever a mismatch occurs.
- The algorithm improves upon the brute force method by utilizing information from previous comparisons to avoid unnecessary character comparisons.

Benefits of the KMP algorithm

- The KMP algorithm efficiently helps you find a specific pattern within a large body of text.
- The KMP algorithm makes your text editing tasks quicker and more efficient.
- The KMP algorithm guarantees 100% reliability.





**Preprocessing Overview:**

- KMP algorithm preprocesses pat[] and constructs an auxiliary **lps[]** of size **m** (same as the size of the pattern) which is used to skip characters while matching.
- Name **lps** indicates the longest proper prefix which is also a suffix. A proper prefix is a prefix with a whole string not allowed. For example, prefixes of “ABC” are “”, “A”, “AB” and “ABC”. Proper prefixes are “”, “A” and “AB”. Suffixes of the string are “”, “C”, “BC”, and “ABC”. 真前缀（proper prefix）是一个串除该串自身外的其他前缀。
- We search for lps in subpatterns. More clearly we ==focus on sub-strings of patterns that are both prefix and suffix==.
- For each sub-pattern pat[0..i] where i = 0 to m-1, lps[i] stores the length of the maximum matching proper prefix which is also a suffix of the sub-pattern pat[0..i].

>   lps[i] = the longest proper prefix of pat[0..i] which is also a suffix of pat[0..i]. 

**Note:** lps[i] could also be defined as the longest prefix which is also a proper suffix. We need to use it properly in one place to make sure that the whole substring is not considered.

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
    lps = [0] * m
    length = 0
    for i in range(1, m):
        while length > 0 and pattern[i] != pattern[length]:
            length = lps[length - 1]    # 跳过前面已经比较过的部分
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

    j = 0  # j是pattern的索引
    for i in range(n):  # i是text的索引
        while j > 0 and text[i] != pattern[j]:
            j = lps[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            matches.append(i - j + 1)
            j = lps[j - 1]
    return matches


text = "ABABABABCABABABABCABABABABC"
pattern = "ABABCABAB"
index = kmp_search(text, pattern)
print("pos matched：", index)
# pos matched： [4, 13]


```



## 关于 kmp 算法中 next 数组的周期性质

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



### 02406: 字符串乘方

http://cs101.openjudge.cn/practice/02406/

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
gpt
使用KMP算法的部分知识，当字符串的长度能被提取的"base字符串"的长度整除时，
即可判断s可以被表示为a^n的形式，此时的n就是s的长度除以"base字符串"的长度。

'''

import sys
while True:
    s = sys.stdin.readline().strip()
    if s == '.':
        break
    len_s = len(s)
    next = [0] * len(s)
    j = 0
    for i in range(1, len_s):
        while j > 0 and s[i] != s[j]:
            j = next[j - 1]
        if s[i] == s[j]:
            j += 1
        next[i] = j
    base_len = len(s)-next[-1]
    if len(s) % base_len == 0:
        print(len_s // base_len)
    else:
        print(1)

```



### 01961: 前缀中的周期

http://cs101.openjudge.cn/practice/01961/

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

如果一个字符串S是由一个字符串T重复K次构成的，则称T是S的循环元。使K出现最大的字符串T称为S的最小循环元，此时的K称为最大循环次数。

现在给定一个长度为N的字符串S，对S的每一个前缀S[1~i],如果它的最大循环次数大于1，则输出该循环的最小循环元长度和最大循环次数。



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





# 二、二分法



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231212104106505.png" alt="image-20231212104106505" style="zoom:50%;" />

数院胡睿诚：这就是个求最小值的最大值或者最大值的最小值的一个套路。

求最值转化为判定对不对，判定问题是可以用贪心解决的，然后用二分只用判定log次。



## 08210: 河中跳房子/石头

binary search/greedy, http://cs101.openjudge.cn/practice/08210

每年奶牛们都要举办各种特殊版本的跳房子比赛，包括在河里从一个岩石跳到另一个岩石。这项激动人心的活动在一条长长的笔直河道中进行，在起点和离起点L远 (1 ≤ *L*≤ 1,000,000,000) 的终点处均有一个岩石。在起点和终点之间，有*N* (0 ≤ *N* ≤ 50,000) 个岩石，每个岩石与起点的距离分别为$Di (0 < Di < L)$。

在比赛过程中，奶牛轮流从起点出发，尝试到达终点，每一步只能从一个岩石跳到另一个岩石。当然，实力不济的奶牛是没有办法完成目标的。

农夫约翰为他的奶牛们感到自豪并且年年都观看了这项比赛。但随着时间的推移，看着其他农夫的胆小奶牛们在相距很近的岩石之间缓慢前行，他感到非常厌烦。他计划移走一些岩石，使得从起点到终点的过程中，最短的跳跃距离最长。他可以移走除起点和终点外的至多*M* (0 ≤ *M* ≤ *N*) 个岩石。

请帮助约翰确定移走这些岩石后，最长可能的最短跳跃距离是多少？



**输入**

第一行包含三个整数L, N, M，相邻两个整数之间用单个空格隔开。
接下来N行，每行一个整数，表示每个岩石与起点的距离。岩石按与起点距离从近到远给出，且不会有两个岩石出现在同一个位置。

**输出**

一个整数，最长可能的最短跳跃距离。

样例输入

```
25 5 2
2
11
14
17
21
```

样例输出

```
4
```

提示：在移除位于2和14的两个岩石之后，最短跳跃距离为4（从17到21或从21到25）。



二分法思路参考：https://blog.csdn.net/gyxx1998/article/details/103831426

**用两分法去推求最长可能的最短跳跃距离**。
最初，待求结果的可能范围是[0，L]的全程区间，因此暂定取其半程(L/2)，作为当前的最短跳跃距离，以这个标准进行岩石的筛选。
**筛选过程**是：
先以起点为基点，如果从基点到第1块岩石的距离小于这个最短跳跃距离，则移除第1块岩石，再看接下来那块岩石（原序号是第2块），如果还够不上最小跳跃距离，就继续移除。。。直至找到一块距离基点超过最小跳跃距离的岩石，保留这块岩石，并将它作为新的基点，再重复前面过程，逐一考察和移除在它之后的那些距离不足的岩石，直至找到下一个基点予以保留。。。
当这个筛选过程最终结束时，那些幸存下来的基点，彼此之间的距离肯定是大于当前设定的最短跳跃距离的。
这个时候要看一下被移除岩石的总数：

- 如果总数>M，则说明被移除的岩石数量太多了（已超过上限值），进而说明当前设定的最小跳跃距离(即L/2)是过大的，其真实值应该是在[0, L/2]之间，故暂定这个区间的中值(L/4)作为接下来的最短跳跃距离，并以其为标准重新开始一次岩石筛选过程。。。
- 如果总数≤M，则说明被移除的岩石数量并未超过上限值，进而说明当前设定的最小跳跃距离(即L/2)很可能过小，准确值应该是在[L/2, L]之间，故暂定这个区间的中值(3/4L)作为接下来的最短跳跃距离

```python
L,n,m = map(int,input().split())
rock = [0]
for i in range(n):
    rock.append(int(input()))
rock.append(L)

def check(x):
    num = 0
    now = 0
    for i in range(1, n+2):
        if rock[i] - now < x:
            num += 1
        else:
            now = rock[i]
            
    if num > m:
        return True
    else:
        return False

# https://github.com/python/cpython/blob/main/Lib/bisect.py
'''
2022fall-cs101，刘子鹏，元培。
源码的二分查找逻辑是给定一个可行的下界和不可行的上界，通过二分查找，将范围缩小同时保持下界可行而区间内上界不符合，
但这种最后print(lo-1)的写法的基础是最后夹出来一个不可行的上界，但其实L在这种情况下有可能是可行的
（考虑所有可以移除所有岩石的情况），所以我觉得应该将上界修改为不可能的 L+1 的逻辑才是正确。
例如：
25 5 5
1
2
3
4
5

应该输出 25
'''
# lo, hi = 0, L
lo, hi = 0, L+1
ans = -1
while lo < hi:
    mid = (lo + hi) // 2
    
    if check(mid):
        hi = mid
    else:               # 返回False，有可能是num==m
        ans = mid       # 如果num==m, mid就是答案
        lo = mid + 1
        
#print(lo-1)
print(ans)
```





## 04135: 月度开销

binary search/greedy , http://cs101.openjudge.cn/practice/04135

农夫约翰是一个精明的会计师。他意识到自己可能没有足够的钱来维持农场的运转了。他计算出并记录下了接下来 *N* (1 ≤ *N* ≤ 100,000) 天里每天需要的开销。

约翰打算为连续的*M* (1 ≤ *M* ≤ *N*) 个财政周期创建预算案，他把一个财政周期命名为fajo月。每个fajo月包含一天或连续的多天，每天被恰好包含在一个fajo月里。

约翰的目标是合理安排每个fajo月包含的天数，使得开销最多的fajo月的开销尽可能少。

**输入**

第一行包含两个整数N,M，用单个空格隔开。
接下来N行，每行包含一个1到10000之间的整数，按顺序给出接下来N天里每天的开销。

**输出**

一个整数，即最大月度开销的最小值。

样例输入

```
7 5
100
400
300
100
500
101
400
```

样例输出

```
500
```

提示：若约翰将前两天作为一个月，第三、四两天作为一个月，最后三天每天作为一个月，则最大月度开销为500。其他任何分配方案都会比这个值更大。



在所给的N天开销中寻找连续M天的最小和，即为最大月度开销的最小值。

与 `OJ08210：河中跳房子`  一样都是二分+贪心判断，但注意这道题目是最大值求最小。

参考 bisect 源码的二分查找写法，https://github.com/python/cpython/blob/main/Lib/bisect.py ，两个题目的代码均进行了规整。
因为其中涉及到 num==m 的情况，有点复杂。二者思路一样，细节有点不一样。

```python
n,m = map(int, input().split())
expenditure = []
for _ in range(n):
    expenditure.append(int(input()))

def check(x):
    num, s = 1, 0
    for i in range(n):
        if s + expenditure[i] > x:
            s = expenditure[i]
            num += 1
        else:
            s += expenditure[i]
    
    return [False, True][num > m]

# https://github.com/python/cpython/blob/main/Lib/bisect.py
lo = max(expenditure)
# hi = sum(expenditure)
hi = sum(expenditure) + 1
ans = 1
while lo < hi:
    mid = (lo + hi) // 2
    if check(mid):      # 返回True，是因为num>m，是确定不合适
        lo = mid + 1    # 所以lo可以置为 mid + 1。
    else:
        ans = mid    # 如果num==m, mid就是答案
        hi = mid
        
#print(lo)
print(ans)
```



为了练习递归，写出了下面代码

```python
n, m = map(int, input().split())
expenditure = [int(input()) for _ in range(n)]

left,right = max(expenditure), sum(expenditure)

def check(x):
    num, s = 1, 0
    for i in range(n):
        if s + expenditure[i] > x:
            s = expenditure[i]
            num += 1
        else:
            s += expenditure[i]
    
    return [False, True][num > m]

res = 0

def binary_search(lo, hi):
    if lo >= hi:
        global res
        res = lo
        return
    
    mid = (lo + hi) // 2
    #print(mid)
    if check(mid):
        lo = mid + 1
        binary_search(lo, hi)
    else:
        hi = mid
        binary_search(lo, hi)
        
binary_search(left, right)
print(res)
```



2021fall-cs101，郑天宇。

一开始难以想到用二分法来解决此题，主要是因为长时间被从正面直接解决问题的思维所禁锢，忘记了**==对于有限的问题，其实可以采用尝试的方法来解决==**。这可能就是“计算思维”的生动体现吧，也可以说是计算概论课教会我们的一个全新的思考问题的方式。

2021fall-cs101，韩萱。居然还能这么做...自己真的想不出来，还是“先完成，再完美”，直接看题解比较好，不然自己想是真的做不完的。

2021fall-cs101，欧阳韵妍。

解题思路：这道题前前后后花了大概3h+（如果考试碰到这种题希望我能及时止损马上放弃），看到老师分享的叶晨熙同学的作业中提到“两球之间的最小磁力”问题的题解有助于理解二分搜索，去找了这道题的题解，看完之后果然有了一点思路，体会到了二分搜索其实就相当于一个往空隙里“插板”的问题，只不过可以运用折半的方法代替一步步挪动每个板子，从而降低时间复杂度。不过虽然有了大致思路但是还是不知道怎么具体实现，于是去仔仔细细地啃了几遍题解。def 的check 函数就是得出在确定了两板之间最多能放多少开销后的一种插板方法；两板之间能放的开销的最大值的最大值（maxmax）一开始为开销总和，两板之间能放的开销的最大值的最小值minmax）一开始为开销中的最大值，我们的目标就是尽可能缩小这个maxmax。如果通过每次减去1 来缩小maxmax 就会超时，那么这时候就使用二分方法，看看  (maxmax+minmax)//2 能不能行，如果可以，大于  (maxmax+minmax)//2的步骤就能全部省略了，maxmax 直接变为  (maxmax+minmax)//2；如果不可以，那么让minmax 变成  (maxmax+minmax)//2+1，同样可以砍掉一半【为什么可以砍掉一半可以这样想：按照check（）的定义，如果输出了False 代表板子太多了，那么“两板之间能放的开销的最大值”（这里即middle）太小了，所以最后不可能采用小于middle 的开销，即maxmax不可能为小于middle 的值，那么这时候就可以把小于middle 的值都砍掉】

感觉二分法是用于在一个大范围里面通过范围的缩小来定位的一种缩短搜素次数的方法。

2021fall-cs101，王紫琪。【月度开销】强烈建议把 欧阳韵妍 同学的思路放进题解！对于看懂代码有很大帮助（拯救了我的头发）

```python
n, m = map(int, input().split())
L = list(int(input()) for x in range(n))

def check(x):
    num, cut = 1, 0
    for i in range(n):
        if cut + L[i] > x:
            num += 1
            cut = L[i]  #在L[i]左边插一个板，L[i]属于新的fajo月
        else:
            cut += L[i]
    
    if num > m:
        return False
    else:
        return True

maxmax = sum(L)
minmax = max(L)
while minmax < maxmax:
    middle = (maxmax + minmax) // 2
    if check(middle):   #表明这种插法可行，那么看看更小的插法可不可以
        maxmax = middle
    else:
        minmax = middle + 1#这种插法不可行，改变minmax看看下一种插法可不可以

print(maxmax)
```







# 三、基数排序

基数排序是一种非比较型整数排序算法，其原理是将整数按位数切割成不同的数字，然后按每个位数分别比较。由于整数也可以表达字符串（比如名字或日期）和特定格式的浮点数，所以基数排序也不是只能使用于整数。



```python
def radixSort(arr):
    max_value = max(arr)
    digit = 1
    while digit <= max_value:
        temp = [[] for _ in range(10)]
        for i in arr:
            t = i // digit % 10
            temp[t].append(i)
        arr.clear()
        for bucket in temp:
            arr.extend(bucket)
        digit *= 10
    return arr

arr = [170, 45, 75, 90, 802, 24, 2, 66]
ans = radixSort(arr)
print(*ans)

# Output:
# 2 24 45 66 75 90 170 802
```



这个程序是一个实现基数排序（Radix Sort）的函数。基数排序是一种非比较型的排序算法，它根据数字的位数来对数字进行排序。

下面是对程序的解读：

1. `radixSort` 函数接受一个整数列表 `arr` 作为输入，并返回排序后的列表。
2. 在函数中，首先找出列表中的最大值 `max_value`，以确定需要排序的数字的最大位数。
3. 然后，通过 `digit` 变量来表示当前处理的位数，初始化为 1。在每次迭代中，`digit` 的值会乘以 10，以处理下一个更高位的数字。
4. 在每次迭代中，创建一个包含 10 个空列表的临时列表 `temp`，用于存储每个数字在当前位数上的分组情况。
5. 对于列表中的每个数字 `i`，计算其在当前位数上的值 `t`（通过取整除和取模操作），然后将数字 `i` 存入对应的桶中。
6. 在填充完所有桶之后，将桶中的数字按照顺序取出，重新放入原始列表 `arr` 中。这样就完成了对当前位数的排序。
7. 继续迭代，直到处理完所有位数为止。
8. 最后，返回排序后的列表 `arr`。

通过基数排序，可以有效地对整数列表进行排序，时间复杂度为 O(d * (n + k))，其中 d 是最大位数，n 是数字个数，k 是基数（这里是 10）。



**Complexity Analysis of Radix Sort**

Time Complexity:

- Radix sort is a non-comparative integer sorting algorithm that sorts data with integer keys by grouping the keys by the individual digits which share the same significant position and value. It has a time complexity of O(d \* (n + b)), where d is the number of digits, n is the number of elements, and b is the base of the number system being used.
- In practical implementations, radix sort is often faster than other comparison-based sorting algorithms, such as quicksort or merge sort, for large datasets, especially when the keys have many digits. However, its time complexity grows linearly with the number of digits, and so it is not as efficient for small datasets.

Auxiliary Space:

- Radix sort also has a space complexity of O(n + b), where n is the number of elements and b is the base of the number system. This space complexity comes from the need to create buckets for each digit value and to copy the elements back to the original array after each digit has been sorted.





# *四、字典与检索

## 06640: 倒排索引

http://cs101.openjudge.cn/2024sp_routine/06640/

给定一些文档，要求求出某些单词的倒排表。

对于一个单词，它的倒排表的内容为出现这个单词的文档编号。

**输入**

第一行包含一个数N，1 <= N <= 1000，表示文档数。
接下来N行，每行第一个数ci，表示第i个文档的单词数。接下来跟着ci个用空格隔开的单词，表示第i个文档包含的单词。文档从1开始编号。1 <= ci <= 100。
接下来一行包含一个数M，1 <= M <= 1000，表示查询数。
接下来M行，每行包含一个单词，表示需要输出倒排表的单词。
每个单词全部由小写字母组成，长度不会超过256个字符，大多数不会超过10个字符。

**输出**

对于每一个进行查询的单词，输出它的倒排表，文档编号按从小到大排序。
如果倒排表为空，输出"NOT FOUND"。

样例输入

```
3
2 hello world
4 the world is great
2 great news
4
hello
world
great
pku
```

样例输出

```
1
1 2
2 3
NOT FOUND
```



要实现一个程序来创建和查询倒排索引，可以使用 字典结构来高效地完成任务。以下是具体的步骤：

1. 首先，解析输入，为每个单词构建倒排索引，即记录每个单词出现在哪些文档中。
2. 使用字典存储倒排索引，其中键为单词，值为一个有序列表，列表中包含出现该单词的文档编号。
3. 对于每个查询，检查字典中是否存在该单词，如果存在，则返回升序文档编号列表；如果不存在，则返回 "NOT FOUND"。

```python
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()

    n = int(data[0])
    index = 1
    inverted_index = {}   # 构建倒排索引
    for i in range(1, n + 1):
        parts = data[index].split()
        doc_id = i
        num_words = int(parts[0])
        words = parts[1:num_words + 1]
        for word in words:
            if word not in inverted_index:
                inverted_index[word] = []
            inverted_index[word].append(doc_id)
        index += 1

    m = int(data[index])
    index += 1
    results = []

    # 查询倒排索引
    for _ in range(m):
        query = data[index]
        index += 1
        if query in inverted_index:
            results.append(" ".join(map(str, sorted(inverted_index[query]))))
        else:
            results.append("NOT FOUND")

    # 输出查询结果
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
```



## 04093: 倒排索引查询

http://cs101.openjudge.cn/practice/04093/

现在已经对一些文档求出了倒排索引，对于一些词得出了这些词在哪些文档中出现的列表。

要求对于倒排索引实现一些简单的查询，即查询某些词同时出现，或者有些词出现有些词不出现的文档有哪些。

**输入**

第一行包含一个数N，1 <= N <= 100，表示倒排索引表的数目。
接下来N行，每行第一个数ci，表示这个词出现在了多少个文档中。接下来跟着ci个数，表示出现在的文档编号，编号不一定有序。1 <= ci <= 1000，文档编号为32位整数。
接下来一行包含一个数M，1 <= M <= 100，表示查询的数目。
接下来M行每行N个数，每个数表示这个词要不要出现，1表示出现，-1表示不出现，0表示无所谓。数据保证每行至少出现一个1。

**输出**

共M行，每行对应一个查询。输出查询到的文档编号，按照编号升序输出。
如果查不到任何文档，输出"NOT FOUND"。

样例输入

```
3
3 1 2 3
1 2
1 3
3
1 1 1
1 -1 0
1 -1 -1
```

样例输出

```
NOT FOUND
1 3
1
```



在实际搜索引擎在处理基于倒排索引的查询时，搜索引擎确实会优先关注各个查询词的倒排表的合并和交集处理，而不是直接准备未出现文档的集合。这种方法更有效，特别是在处理大规模数据集时，因为它允许系统动态地调整和优化查询过程，特别是在有复杂查询逻辑（如多个词的组合、词的排除等）时。详细解释一下搜索引擎如何使用倒排索引来处理查询：

倒排索引查询的核心概念

1. 倒排索引结构：
   - 对于每个词（token），都有一个关联的文档列表，这个列表通常是按文档编号排序的。
   - 每个文档在列表中可能还会有附加信息，如词频、位置信息等。
2. 处理查询：
   - 单词查询：对于单个词的查询，搜索引擎直接返回该词的倒排列表。
   - 多词交集查询：对于包含多个词的查询，搜索引擎找到每个词的倒排列表，然后计算这些列表的交集。
     这个交集代表了所有查询词都出现的文档集合。
   - 复杂逻辑处理：对于包含逻辑运算（AND, OR, NOT）的查询，搜索引擎会结合使用集合的
     交集（AND）、并集（OR）和差集（NOT）操作来处理查询。特别是在处理 NOT 逻辑时，
     它并不是去查找那些未出现词的文档集合，而是从已经确定的结果集中排除含有这个词的文档。

更贴近实际搜索引擎的处理实现，如下：

```python
import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1

word_documents = []

# 读取每个词的倒排索引
for _ in range(N):
    ci = int(data[index])
    index += 1
    documents = sorted(map(int, data[index:index + ci]))
    index += ci
    word_documents.append(documents)

M = int(data[index])
index += 1

results = []

# 处理每个查询
for _ in range(M):
    query = list(map(int, data[index:index + N]))
    index += N

    # 集合存储各词的文档集合（使用交集获取所有词都出现的文档）
    included_docs = []
    excluded_docs = set()

    # 解析查询条件
    for i in range(N):
        if query[i] == 1:
            included_docs.append(word_documents[i])
        elif query[i] == -1:
            excluded_docs.update(word_documents[i])

    # 仅在有包含词时计算交集
    if included_docs:
        result_set = set(included_docs[0])
        for docs in included_docs[1:]:
            result_set.intersection_update(docs)
        result_set.difference_update(excluded_docs)
        final_docs = sorted(result_set)
        results.append(" ".join(map(str, final_docs)) if final_docs else "NOT FOUND")
    else:
        results.append("NOT FOUND")

# 输出所有查询结果
for result in results:
    print(result)
```



# *五、B-trees

2-3 树、2-3-4 树、B 树和 B+ 树

**2-3 Tree**:

- A 2-3 tree is a type of balanced search tree where each node can have either 2 or 3 children.
- In a 2-3 tree:
  - Every internal node has either 2 children and 1 data element, or 3 children and 2 data elements.
  - The leaves are all at the same level.
- Insertions and deletions in a 2-3 tree may cause tree restructuring to maintain the balance.

**2-3-4 Tree**:

- A 2-3-4 tree is a generalization of the 2-3 tree where nodes can have either 2, 3, or 4 children.
- In a 2-3-4 tree:
  - Every internal node has either 2, 3, or 4 children and 1, 2, or 3 data elements, respectively.
  - The leaves are all at the same level.
- Like the 2-3 tree, insertions and deletions may cause restructuring to maintain balance.

**B-Tree**:

- A B-tree is a self-balancing tree data structure that maintains sorted data and allows for efficient search, insertion, and deletion operations.
- In a B-tree:
  - Each node contains multiple keys and pointers to child nodes.
  - Nodes can have a variable number of keys within a certain range, determined by the order of the B-tree.
  - B-trees are balanced and ensure that all leaves are at the same level.
- B-trees are commonly used in databases and file systems for their ability to handle large amounts of data efficiently.

**B+ Tree**:

- A B+ tree is a variation of the B-tree with additional features optimized for disk storage systems.
- In a B+ tree:
  - Data entries are stored only in leaf nodes.
  - Internal nodes store keys and pointers to child nodes but do not store actual data.
  - Leaf nodes are linked together in a linked list, making range queries efficient.
- B+ trees are commonly used in database systems because of their efficiency in disk-based storage and their ability to handle range queries effectively.

These tree structures are fundamental in computer science and are widely used in various applications where efficient data storage and retrieval are essential.





Here's a brief tutorial on B-trees and B+ trees, along with Python implementations:

### B-Tree Tutorial:

1. **Introduction**:
   - A B-tree is a self-balancing tree data structure that maintains sorted data and allows for efficient search, insertion, and deletion operations.
   - Each node contains multiple keys and pointers to child nodes.
   - Nodes can have a variable number of keys within a certain range, determined by the order of the B-tree.
   - B-trees are balanced and ensure that all leaves are at the same level.

2. **Operations**:
   - **Search**: Starts from the root and recursively searches down the tree to find the target key.
   - **Insertion**: Starts from the root and recursively inserts the key into the appropriate leaf node. If the leaf node is full, it may split, and the median key is pushed up to the parent node.
   - **Deletion**: Starts from the root and recursively deletes the key from the appropriate leaf node. If deletion causes underflow, nodes may merge or borrow keys from neighboring nodes to maintain balance.

### B-Tree Python Implementation:

B树是一种自平衡的树，主要用于系统中大量数据的存储。它可以保持数据排序，并允许进行高效的插入、删除和查找操作。

B树的主要特性如下：

1. 所有叶子节点都在同一层。
2. 每个节点中的元素从小到大排列，节点当中k个元素正好是k+1个孩子指针的分界点。
3. 非叶子节点的孩子节点数目介于 t 和 2t 之间，其中 t 是树的阶数。每个非根节点至少有 t-1 个关键字，至多有 2t-1 个关键字。
4. 根节点至少有两个孩子，除非它是一个叶节点（即树中只有一个节点）。

以下是一个简单的B树的Python实现：

```python
class BTreeNode:
    def __init__(self, leaf=False):
        self.leaf = leaf
        self.keys = []
        self.child = []

class BTree:
    def __init__(self, t):
        self.root = BTreeNode(True)
        self.t = t

    def insert(self, k):
        root = self.root
        if len(root.keys) == (2*self.t) - 1:
            temp = BTreeNode()
            self.root = temp
            temp.child.insert(0, root)
            self.split_child(temp, 0)
            self.insert_non_full(temp, k)
        else:
            self.insert_non_full(root, k)

    def insert_non_full(self, x, k):
        i = len(x.keys) - 1
        if x.leaf:
            x.keys.append((None, None))
            while i >= 0 and k < x.keys[i]:
                x.keys[i+1] = x.keys[i]
                i -= 1
            x.keys[i+1] = k
        else:
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            if len(x.child[i].keys) == (2*self.t) - 1:
                self.split_child(x, i)
                if k > x.keys[i]:
                    i += 1
            self.insert_non_full(x.child[i], k)

    def split_child(self, x, i):
        t = self.t
        y = x.child[i]
        z = BTreeNode(y.leaf)
        x.child.insert(i+1, z)
        x.keys.insert(i, y.keys[t-1])
        z.keys = y.keys[t: (2*t) - 1]
        y.keys = y.keys[0: t-1]
        if not y.leaf:
            z.child = y.child[t: 2*t]
            y.child = y.child[0: t-1]

# 创建一个阶数为3的B树
btree = BTree(3)

# 插入一些键
keys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for key in keys:
    btree.insert(key)

# 打印B树的根节点的键
print(btree.root.keys)
#output: [(3, 6)] 
```

这个代码实现了B树的基本操作，包括插入和分裂子节点。但是，它并没有实现删除操作，这是一个更复杂的问题，需要更多的代码来处理各种情况。





### B+ Tree Tutorial:

1. **Introduction**:
   - A B+ tree is a variation of the B-tree with additional features optimized for disk storage systems.
   - Data entries are stored only in leaf nodes.
   - Internal nodes store keys and pointers to child nodes but do not store actual data.
   - Leaf nodes are linked together in a linked list, making range queries efficient.

2. **Operations**:
   - Operations are similar to B-trees but are optimized for disk access patterns, making them suitable for database systems.

### 





# 参考

20231107_KMP.md

principal.md
