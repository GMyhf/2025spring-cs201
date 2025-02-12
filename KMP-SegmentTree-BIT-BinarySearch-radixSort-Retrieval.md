# KMP-SegmentTree-BIT-BinarySearch-radixSort-Retrieval

Updated 2101 GMT+8 Feb 12, 2025

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





# 二、线段树、树状数组

> 2025/2/11 线段树、树状数组在计概课程中放在附录部分，数算可以加深理解。

理解时间复杂度 $O(1)$ 和 $O(n)$ 权衡处理方法，有的题目 $O(n^2)$ 算法超时，需要把时间复杂度降到$O(nLogn)$才能AC。

例如：27018:康托展开，http://cs101.openjudge.cn/practice/27018/



线段树（Segment Tree）和树状数组（Binary Indexed Tree）的区别和联系：

1）时间复杂度相同, 但是树状数组的常数优于线段树。

2）树状数组的作用被线段树完全涵盖, 凡是可以使用树状数组解决的问题, 使用线段树一定可以解决, 但是线段树能够解决的问题树状数组未必能够解决。

3）树状数组的代码量比线段树小很多。



Segment Tree and Its Applications

https://www.baeldung.com/cs/segment-trees#:~:text=The%20segment%20tree%20is%20a,structure%20such%20as%20an%20array.

The segment tree is a type of data structure from computational geometry. [Bentley](https://en.wikipedia.org/wiki/Bentley–Ottmann_algorithm) proposed this well-known technique in 1977. A segment tree is essentially a binary tree in whose nodes we store the information about the segments of a linear data structure such as an array.

> 区间树是一种来自计算几何的数据结构。Bentley 在 1977 年提出了这一著名的技术。区间树本质上是一棵二叉树，在其节点中存储了关于线性数据结构（如数组）的区段信息。

Fenwick tree

https://en.wikipedia.org/wiki/Fenwick_tree#:~:text=A%20Fenwick%20tree%20or%20binary,in%20an%20array%20of%20values.&text=This%20structure%20was%20proposed%20by,further%20modification%20published%20in%201992.

A **Fenwick tree** or **binary indexed tree** **(BIT)** is a data structure that can efficiently update values and calculate [prefix sums](https://en.wikipedia.org/wiki/Prefix_sum) in an array of values.

This structure was proposed by Boris Ryabko in 1989 with a further modification published in 1992. It has subsequently become known under the name Fenwick tree after Peter Fenwick, who described this structure in his 1994 article.

> Fenwick 树 或 二叉索引树 (BIT) 是一种数据结构，可以高效地更新数组中的值并计算前缀和。
>
> 这种结构由 Boris Ryabko 于 1989 年提出，并在 1992 年进行了进一步的修改。此后，这种结构以其在 1994 年的文章中描述它的 Peter Fenwick 的名字而广为人知，被称为 Fenwick 树。



## 2.1 Segment tree | Efficient implementation

https://www.geeksforgeeks.org/segment-tree-efficient-implementation/

Let us consider the following problem to understand Segment Trees without recursion.
We have an array $arr[0 . . . n-1]$. We should be able to, 

1. Find the sum of elements from index `l` to `r` where $0 \leq l \leq r \leq n-1$
2. Change the value of a specified element of the array to a new value `x`. We need to do $arr[i] = x$ where $0 \leq i \leq n-1$. 

A **simple solution** is to run a loop from `l` to `r` and calculate the sum of elements in the given range. To update a value, simply do $arr[i] = x$. The first operation takes **O(n)** time and the second operation takes **O(1)** time.

> **简单解决方案** 是从 `l` 到 `r` 运行一个循环，计算给定范围内的元素之和。要更新一个值，只需执行 `arr[i] = x`。第一个操作（查询）的时间复杂度为 **O(n)**，第二个操作（更新）的时间复杂度为 **O(1)**。

**Another solution** is to create another array and store the sum from start to `i` at the ith index in this array. The sum of a given range can now be calculated in O(1) time, but the update operation takes O(n) time now. This works well if the number of query operations is large and there are very few updates.

> **另一种解决方案** 是创建另一个数组，并在该数组的第 `i` 个索引处存储从起始位置到 `i` 的元素之和。现在可以在 O(1) 时间内计算给定范围的和，但更新操作现在需要 O(n) 时间。如果查询操作的数量很大而更新操作很少，这种方法效果很好。

What if the number of queries and updates are equal? Can we perform both the operations in O(log n) time once given the array? We can use a [Segment Tree](https://www.geeksforgeeks.org/segment-tree-set-1-sum-of-given-range/) to do both operations in O(logn) time. We have discussed the complete implementation of segment trees in our [previous](https://www.geeksforgeeks.org/segment-tree-set-1-sum-of-given-range/) post. In this post, we will discuss the easier and yet efficient implementation of segment trees than in the previous post.

> 但如果查询和更新操作的数量相等呢？我们能否在给定数组的情况下，使两个操作都在 O(log n) 时间内完成？我们可以使用 线段树 来在 O(log n) 时间内完成这两个操作。我们在之前的帖子中详细讨论了线段树的完整实现。在这篇文章中，我们将讨论比之前更简单且高效的线段树实现方法。

Consider the array and segment tree as shown below:  叶子是数组值，非叶是和

![img](https://media.geeksforgeeks.org/wp-content/uploads/excl.png)



You can see from the above image that the original array is at the bottom and is 0-indexed with 16 elements. The tree contains a total of 31 nodes where the leaf nodes or the elements of the original array start from node 16. So, we can easily construct a segment tree for this array using a `2*N` sized array where N is the number of elements in the original array. The leaf nodes will start from index N in this array and will go up to index `(2 * N – 1)`. Therefore, the element at index `i` in the original array will be at index `(i + N)` in the segment tree array. Now to calculate the parents, we will start from the index `(N – 1)` and move upward. For index `i` , the left child will be at `(2 * i)` and the right child will be at `(2*i + 1)` index. So the values at nodes at `(2 * i)` and `(2*i + 1)` are combined at i-th node to construct the tree. 

> 从上图可以看出，原始数组位于底部，是 0 索引的，包含 16 个元素。树总共有 31 个节点，其中叶节点或原始数组的元素从节点 16 开始。因此，我们可以使用一个大小为 2*N 的数组轻松构建这个数组的线段树，其中 N 是原始数组中的元素数量。叶节点将从该数组的索引 N 开始，一直到索引 (2 * N - 1)。因此，原始数组中索引 i 处的元素将在线段树数组中的索引 (i + N) 处。现在，为了计算父节点，我们将从索引 (N - 1) 开始向上移动。对于索引 i，左孩子将位于 (2 * i) 索引处，右孩子将位于 (2 * i + 1) 索引处。因此，节点 (2 * i) 和 (2 * i + 1) 处的值将在 i 索引处组合以构建树。

As you can see in the above figure, we can query in this tree in an interval `[L,R)` with left index (L) included and right (R) excluded.
We will implement all of these multiplication and addition operations using bitwise operators.

> 如上图所示，我们可以在区间 [L, R) 中查询这棵树，其中左索引 L 包含在内，右索引 R 排除在外。
> 我们将使用位运算符实现所有的乘法和加法操作。

Let us have a look at the complete implementation: 

```python
# Python3 Code Addition 

# limit for array size 
N = 100000; 

# Max size of tree 
tree = [0] * (2 * N); 

# function to build the tree 
def build(arr) : 

	# insert leaf nodes in tree 
	for i in range(n) : 
		tree[n + i] = arr[i]; 
	
	# build the tree by calculating parents 
	for i in range(n - 1, 0, -1) : 
    # tree[i] = tree[2*i] + tree[2*i+1]
		tree[i] = tree[i << 1] + tree[i << 1 | 1]; 	

# function to update a tree node 
def updateTreeNode(p, value) : 
	
	# set value at position p 
	tree[p + n] = value; 
	p = p + n; 
	
	# move upward and update parents 
	i = p; 
	
	while i > 1 : 
		
		tree[i >> 1] = tree[i] + tree[i ^ 1]; 
		i >>= 1; 

# function to get sum on interval [l, r) 
def query(l, r) : 

	res = 0; 
	
	# loop to find the sum in the range 
	l += n; 
	r += n; 
	
	while l < r : 
	
		if (l & 1) : 
			res += tree[l]; 
			l += 1
	
		if (r & 1) : 
			r -= 1; 
			res += tree[r]; 
			
		l >>= 1; 
		r >>= 1
	
	return res; 

if __name__ == "__main__" : 

	a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; 

	n = len(a); 
	
	build(a); 
	
	# print the sum in range(1,2) index-based 
	print(query(1, 3)); 
	
	# modify element at 2nd index 
	updateTreeNode(2, 1); 
	
	# print the sum in range(1,2) index-based 
	print(query(1, 3)); 

```



**Output:** 

```
5
3
```

Yes! That is all. The complete implementation of the segment tree includes the query and update functions. Let us now understand how each of the functions works: 


1. The picture makes it clear that the leaf nodes are stored at i+n, so we can clearly insert all leaf nodes directly. 

   > 图片清楚地表明叶节点存储在i+n的位置，因此我们可以直接明确地插入所有叶节点。

2. The next step is to build the tree and it takes O(n) time. The parent always has its less index than its children, so we just process all the nodes in decreasing order, calculating the value of the parent node. If the code inside the build function to calculate parents seems confusing, then you can see this code. It is equivalent to that inside the build function. 

   > 下一步是构建树，这需要O(n)的时间。父节点的索引总是小于其子节点的索引，所以我们只需按递减顺序处理所有节点，计算父节点的值。如果构建函数中用于计算父节点的代码看起来令人困惑，那么你可以参考这段代码。它与构建函数内部的代码等效。

   `tree[i] = tree[2*i] + tree[2*i+1]`

 

3. Updating a value at any position is also simple and the time taken will be proportional to the height （“高度”这个概念，其实就是从下往上度量，树这种数据结构的高度是从最底层开始计数，并且计数的起点是0） of the tree. We only update values in the parents of the given node which is being changed. So to get the parent, we just go up to the parent node, which is `p/2` or `p>>1`, for node `p`. `p^1` turns `(2*i`) to `(2*i + 1)` and vice versa to get the second child of p.

   > 在任意位置更新一个值也非常简单，所需时间将与树的高度成正比。我们只更新给定节点（即正在更改的节点）的父节点中的值。为了得到父节点，我们只需向上移动到节点p的父节点，该父节点为p/2或p>>1。p^1将`(2*i)`转换为`(2*i + 1)`反之亦然，以获得p的第二个子节点。

4. Computing the sum also works in $O(Logn)$ time. If we work through an interval of [3,11), we need to calculate only for nodes 19,26,12, and 5 in that order.  要演示这个索引上行的求和过程，前面程序数组是12个元素，图示是16个元素，需要稍作修改。增加了print输出，便于调试。



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/202310312148391.png" alt="image-20231031214814445" style="zoom:50%;" />



The idea behind the query function is whether we should include an element in the sum or whether we should include its parent. Let’s look at the image once again for proper understanding. 

![img](https://media.geeksforgeeks.org/wp-content/uploads/excl.png)

Consider that `L` is the left border of an interval and `R` is the right border of the interval `[L,R)`. It is clear from the image that if `L` is odd, then it means that it is the right child of its parent and our interval includes only `L` and not the parent. So we will simply include this node to sum and move to the parent of its next node by doing `L = (L+1)/2`. Now, if L is even, then it is the left child of its parent and the interval includes its parent also unless the right borders interfere. Similar conditions are applied to the right border also for faster computation. We will stop this iteration once the left and right borders meet.

> 假设`L`是一个区间的左边界，而`R`是区间`[L,R)`的右边界。从图中可以明显看出，如果`L`是奇数，这意味着它是其父节点的右孩子，并且我们的区间仅包含`L`而不包括其父节点。因此，我们将简单地把这个节点加到总和中，并通过执行`L = (L+1)/2`移动到下一个节点的父节点。现在，如果`L`是偶数，那么它是其父节点的左孩子，除非右边界干涉，否则区间也包括其父节点。对于右边界也有类似的条件，以便更快地计算。一旦左右边界相遇，我们就会停止这次迭代。

The theoretical time complexities of both previous implementation and this implementation is the same, but practically, it is found to be much more efficient as there are no recursive calls. We simply iterate over the elements that we need. Also, this is very easy to implement.

> 这两种实现的理论时间复杂度是相同的，但在实际应用中，后者被发现要高效得多，因为没有递归调用。我们只是迭代我们需要的元素。此外，这种方法非常容易实现。

**Time Complexities:**

- Tree Construction: O( n )
- Query in Range: O( Log n )
- Updating an element: O( Log n ).

**Auxiliary Space:** O(2*N)



### 示例1364A: A. XXXXX

brute force/data structures/number theory/two pointers, 1200, https://codeforces.com/problemset/problem/1364/A

Ehab loves number theory, but for some reason he hates the number 𝑥. Given an array 𝑎, find the length of its longest subarray such that the sum of its elements **isn't** divisible by 𝑥, or determine that such subarray doesn't exist.

An array 𝑎 is a subarray of an array 𝑏 if 𝑎 can be obtained from 𝑏 by deletion of several (possibly, zero or all) elements from the beginning and several (possibly, zero or all) elements from the end.

**Input**

The first line contains an integer 𝑡 (1≤𝑡≤5) — the number of test cases you need to solve. The description of the test cases follows.

The first line of each test case contains 2 integers 𝑛 and 𝑥 (1≤𝑛≤10^5^, 1≤𝑥≤10^4^) — the number of elements in the array 𝑎 and the number that Ehab hates.

The second line contains 𝑛 space-separated integers $𝑎_1, 𝑎_2, ……, 𝑎_𝑛 (0≤𝑎_𝑖≤10^4)$ — the elements of the array 𝑎.

**Output**

For each testcase, print the length of the longest subarray whose sum isn't divisible by 𝑥. If there's no such subarray, print −1.

Example

input

```
3
3 3
1 2 3
3 4
1 2 3
2 2
0 6
```

output

```
2
3
-1
```

Note

In the first test case, the subarray \[2,3\] has sum of elements 5, which isn't divisible by 3.

In the second test case, the sum of elements of the whole array is 6, which isn't divisible by 4.

In the third test case, all subarrays have an even sum, so the answer is −1.



Pypy3 可以AC。使用tree segment，时间复杂度是O(n*logn)

```python
# CF 1364A
 
# def prefix_sum(nums):
#     prefix = []
#     total = 0
#     for num in nums:
#         total += num
#         prefix.append(total)
#     return prefix
 
# def suffix_sum(nums):
#     suffix = []
#     total = 0
#     # 首先将列表反转
#     reversed_nums = nums[::-1]
#     for num in reversed_nums:
#         total += num
#         suffix.append(total)
#     # 将结果反转回来
#     suffix.reverse()
#     return suffix
 
 
t = int(input())
ans = []
for _ in range(t):
    n, x = map(int, input().split())
    a = [int(i) for i in input().split()]


# Segment tree | Efficient implementation
# https://www.geeksforgeeks.org/segment-tree-efficient-implementation/

    # Max size of tree 
    tree = [0] * (2 * n); 

    def build(arr) : 

        # insert leaf nodes in tree 
        for i in range(n) : 
            tree[n + i] = arr[i]; 
        
        # build the tree by calculating parents 
        for i in range(n - 1, 0, -1) : 
            tree[i] = tree[i << 1] + tree[i << 1 | 1]; 

    # function to update a tree node 
    def updateTreeNode(p, value) : 
        
        # set value at position p 
        tree[p + n] = value; 
        p = p + n; 
        
        # move upward and update parents 
        i = p; 
        
        while i > 1 : 
            
            tree[i >> 1] = tree[i] + tree[i ^ 1]; 
            i >>= 1; 

    # function to get sum on interval [l, r) 
    def query(l, r) : 

        res = 0; 
        
        # loop to find the sum in the range 
        l += n; 
        r += n; 
        
        while l < r : 
        
            if (l & 1) : 
                res += tree[l]; 
                l += 1
        
            if (r & 1) : 
                r -= 1; 
                res += tree[r]; 
                
            l >>= 1; 
            r >>= 1
        
        return res; 
    #aprefix_sum = prefix_sum(a)
    #asuffix_sum = suffix_sum(a)
 
    build([i%x for i in a]);
    
    left = 0
    right = n - 1
    if right == 0:
        if a[0] % x !=0:
            print(1)
        else:
            print(-1)
        continue
 
    leftmax = 0
    rightmax = 0
    while left != right:
        #total = asuffix_sum[left]
        total = query(left, right+1)
        if total % x != 0:
            leftmax = right - left + 1
            break
        else:
            left += 1
 
    left = 0
    right = n - 1
    while left != right:
        #total = aprefix_sum[right]
        total = query(left, right+1)
        if total % x != 0:
            rightmax = right - left + 1
            break
        else:
            right -= 1
    
    if leftmax == 0 and rightmax == 0:
        #print(-1)
        ans.append(-1)
    else:
        #print(max(leftmax, rightmax))
        ans.append(max(leftmax, rightmax))

print('\n'.join(map(str,ans)))
```



如果用sum求和，O(n^2)，pypy3也会在test3 超时。







### Benifits of segment tree usage

https://www.geeksforgeeks.org/segment-tree-sum-of-given-range/

**Range Queries:** One of the main use cases of segment trees is to perform range queries on an array in an efficient manner. The query function in the segment tree can return the ==minimum, maximum, sum, or any other aggregation== of elements within a specified range in the array in O(log n) time.

> **区间查询：** 线段树的主要用途之一是以高效的方式对数组进行区间查询。线段树中的查询函数可以在O(log n)时间内返回指定区间内元素的**最小值、最大值、和或其他聚合结果**。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231031140857139.png" alt="image-20231031140857139" style="zoom:50%;" />



假设根节点下标从0开始，左子节点 = 2\*父节点+1，右子节点  = 2\*父节点+2

二叉树的父子节点位置关系，https://zhuanlan.zhihu.com/p/339763580

```python
class SegmentTree:
	def __init__(self, array):
		self.size = len(array)
		self.tree = [0] * (4 * self.size)
		self.build_tree(array, 0, 0, self.size - 1)

	def build_tree(self, array, tree_index, left, right):
		if left == right:
			self.tree[tree_index] = array[left]
			return
		mid = (left + right) // 2
		self.build_tree(array, 2 * tree_index + 1, left, mid)
		self.build_tree(array, 2 * tree_index + 2, mid + 1, right)
		self.tree[tree_index] = min(self.tree[2 * tree_index + 1], self.tree[2 * tree_index + 2])

	def query(self, tree_index, left, right, query_left, query_right):
		if query_left <= left and right <= query_right:
			return self.tree[tree_index]
		mid = (left + right) // 2
		min_value = float('inf')
		if query_left <= mid:
			min_value = min(min_value, self.query(2 * tree_index + 1, left, mid, query_left, query_right))
		if query_right > mid:
			min_value = min(min_value, self.query(2 * tree_index + 2, mid + 1, right, query_left, query_right))
		return min_value

	def query_range(self, left, right):
		return self.query(0, 0, self.size - 1, left, right)


if __name__ == '__main__':
	array = [1, 3, 2, 5, 4, 6]
	st = SegmentTree(array)
	print(st.query_range(1, 5)) # 2

```

如果要返回区间最大值，只需要修改第14、20、22、24行程序为求最大相应代码

```python
        #self.tree[tree_index] = min(self.tree[2 * tree_index + 1], self.tree[2 * tree_index + 2])
        self.tree[tree_index] = max(self.tree[2 * tree_index + 1], self.tree[2 * tree_index + 2])
...
				#min_value = float('inf')
        min_value = -float('inf')
        if query_left <= mid:
            #min_value = min(min_value, self.query(2 * tree_index + 1, left, mid, query_left, query_right))
            min_value = max(min_value, self.query(2 * tree_index + 1, left, mid, query_left, query_right))
        if query_right > mid:
            #min_value = min(min_value, self.query(2 * tree_index + 2, mid + 1, right, query_left, query_right))
            min_value = max(min_value, self.query(2 * tree_index + 2, mid + 1, right, query_left, query_right))
        return min_value
   ....
   print(st.query_range(1, 5)) # 6   
      
```

如果要返回区间 求和，只需要修改第14、20、22、24行程序为求和代码。



## 2.2 树状数组（Binary Indexed Tree）

树状数组或二叉索引树（英语：Binary Indexed Tree），又以其发明者命名为Fenwick树，最早由Peter M. Fenwick于1994年以A New Data Structure for Cumulative Frequency Tables为题发表。其初衷是解决数据压缩里的累积频率（Cumulative Frequency）的计算问题，现多用于高效计算数列的前缀和， 区间和。



**Binary Indexed Tree or Fenwick Tree**

https://www.geeksforgeeks.org/binary-indexed-tree-or-fenwick-tree-2/

Let us consider the following problem to understand Binary Indexed Tree.
We have an array $arr[0 . . . n-1]$. We would like to 
**1** Compute the sum of the first i elements. 
**2** Modify the value of a specified element of the array arr[i] = x where $0 \leq i \leq n-1$.
A **simple solution** is to run a loop from 0 to i-1 and calculate the sum of the elements. To update a value, simply do arr[i] = x. The first operation takes O(n) time and the second operation takes O(1) time. Another simple solution is to create an extra array and store the sum of the first i-th elements at the i-th index in this new array. The sum of a given range can now be calculated in O(1) time, but the update operation takes O(n) time now. This works well if there are a large number of query operations but a very few number of update operations.
**Could we perform both the query and update operations in O(log n) time?** 
One efficient solution is to use [Segment Tree](https://www.geeksforgeeks.org/segment-tree-set-1-sum-of-given-range/) that performs both operations in O(Logn) time.
An alternative solution is Binary Indexed Tree, which also achieves O(Logn) time complexity for both operations. Compared with Segment Tree, Binary Indexed Tree requires less space and is easier to implement.



> 让我们考虑以下问题来理解二叉索引树（Binary Indexed Tree, BIT）：
> 我们有一个数组 $arr[0 . . . n-1]$。我们希望实现两个操作：
>
> 1. 计算前i个元素的和。
> 2. 修改数组中指定位置的值，即设置 $arr[i] = x$，其中 $0 \leq i \leq n-1$。
>
> 一个简单的解决方案是从0到i-1遍历并计算这些元素的总和。要更新一个值，只需执行 $arr[i] = x$。第一个操作的时间复杂度为O(n)，而第二个操作的时间复杂度为O(1)。另一种简单的解决方案是创建一个额外的数组，并在这个新数组的第i个位置存储前i个元素的总和。这样，给定范围的和可以在O(1)时间内计算出来，但是更新操作现在需要O(n)时间。当查询操作非常多而更新操作非常少时，这种方法表现良好。
>
> **我们能否在O(log n)时间内同时完成查询和更新操作呢？**
> 一种高效的解决方案是使用段树（Segment Tree），它能够在O(log n)时间内完成这两个操作。
> 另一种解决方案是二叉索引树（Binary Indexed Tree，也称作Fenwick Tree），同样能够以O(log n)的时间复杂度完成查询和更新操作。与段树相比，二叉索引树所需的空间更少，且实现起来更加简单。



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231031141452788.png" alt="image-20231031141452788" style="zoom:50%;" />



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231031141531597.png" alt="image-20231031141531597" style="zoom:50%;" />



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231031141548736.png" alt="image-20231031141548736" style="zoom:50%;" />

**Representation** 
Binary Indexed Tree is represented as an array. Let the array be BITree[]. Each node of the Binary Indexed Tree stores the sum of some elements of the input array. The size of the Binary Indexed Tree is equal to the size of the input array, denoted as n. In the code below, we use a size of n+1 for ease of implementation.

> **表示方式**
> 二叉索引树用数组形式表示。设该数组为BITree[]。二叉索引树的每个节点存储了输入数组某些元素的和。二叉索引树的大小等于输入数组的大小，记为n。在下面的代码中，为了便于实现，我们使用n+1的大小。

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231031141831067.png" alt="image-20231031141831067" style="zoom:50%;" />



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231031141629059.png" alt="image-20231031141629059" style="zoom:50%;" />



**Construction** 
We initialize all the values in BITree[] as 0. Then we call update() for all the indexes, the update() operation is discussed below.

> **构建**
> 我们首先将BITree[]中的所有值初始化为0。然后对所有的索引调用update()函数，下面将讨论update()操作的具体内容。

**Operations** 


> ***getSum(x): Returns the sum of the sub-array arr[0,…,x]*** 
> // Returns the sum of the sub-array arr[0,…,x] using BITree[0..n], which is constructed from arr[0..n-1] 
>
> 1) Initialize the output sum as 0, the current index as x+1. 
> 2) Do following while the current index is greater than 0. 
>
> …a) Add BITree[index] to sum 
> …b) Go to the parent of BITree[index]. The parent can be obtained by removing 
> the last set bit from the current index, i.e., index = index – (index & (-index)) 
>
> 3) Return sum.

 

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/BITSum.png" alt="BITSum" style="zoom: 67%;" />



getsum(7)

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231031142037881.png" alt="image-20231031142037881" style="zoom:50%;" />

getsum(8)

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231031142146355.png" alt="image-20231031142146355" style="zoom:50%;" />



**整数的二进制表示常用的方式之一是使用补码**

补码是一种表示有符号整数的方法，它将负数的二进制表示转换为正数的二进制表示。补码的优势在于可以使用相同的算术运算规则来处理正数和负数，而不需要特殊的操作。

在补码表示中，最高位用于表示符号位，0表示正数，1表示负数。其他位表示数值部分。

具体将一个整数转换为补码的步骤如下：

1. 如果整数是正数，则补码等于二进制表示本身。
2. 如果整数是负数，则需要先将其绝对值转换为二进制，然后取反，最后加1。

例如，假设要将-5转换为补码：

1. 5的二进制表示为00000101。

2. 将其取反得到11111010。

3. 加1得到11111011，这就是-5的补码表示。

   

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231031142210011.png" alt="image-20231031142210011" style="zoom:50%;" />



The diagram above provides an example of how getSum() is working. Here are some important observations.
BITree[0] is a dummy node. 
BITree[y] is the parent of BITree[x], if and only if y can be obtained by removing the last set bit from the binary representation of x, that is y = x – (x & (-x)).
The child node BITree[x] of the node BITree[y] stores the sum of the elements between y(inclusive) and x(exclusive): arr[y,…,x). 

> 上图提供了一个getSum()如何工作的例子。这里有一些重要的观察点：
>
> - BITree[0]是一个虚拟节点。
> - 如果仅通过从x的二进制表示中移除最后一个设置位（即最右边的1）可以得到y，则BITree[y]是BITree[x]的父节点，这可以表示为 y = x – (x & (-x))。
> - 节点BITree[y]的子节点BITree[x]存储了从y（包括y）到x（不包括x）之间元素的和：arr[y,...,x)。


> ***update(x, val): Updates the Binary Indexed Tree (BIT) by performing arr[index] += val*** 
> // Note that the update(x, val) operation will not change arr[]. It only makes changes to BITree[] 
>
> 1) Initialize the current index as x+1. 
> 2) Do the following while the current index is smaller than or equal to n. 
>
> …a) Add the val to BITree[index] 
> …b) Go to next element of BITree[index]. The next element can be obtained by incrementing the last set bit of the current index, i.e., index = index + (index & (-index))

 

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/BITUpdate12.png" alt="BITUpdate1" style="zoom:67%;" />

update(4, 10)

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231031142428708.png" alt="image-20231031142428708" style="zoom:50%;" />



The update function needs to make sure that all the BITree nodes which contain arr[i] within their ranges being updated. We loop over such nodes in the BITree by repeatedly adding the decimal number corresponding to the last set bit of the current index.
**How does Binary Indexed Tree work?** 
The idea is based on the fact that all positive integers can be represented as the sum of powers of 2. For example 19 can be represented as 16 + 2 + 1. Every node of the BITree stores the sum of n elements where n is a power of 2. For example, in the first diagram above (the diagram for getSum()), the sum of the first 12 elements can be obtained by the sum of the last 4 elements (from 9 to 12) plus the sum of 8 elements (from 1 to 8). The number of set bits in the binary representation of a number n is O(Logn). Therefore, we traverse at-most O(Logn) nodes in both getSum() and update() operations. The time complexity of the construction is O(nLogn) as it calls update() for all n elements. 
**Implementation:** 
Following are the implementations of Binary Indexed Tree.

> 更新函数需要确保所有包含arr[i]在其范围内的BITree节点都被更新。我们通过不断向当前索引添加其最后一位设置位对应的十进制数，在BITree中循环遍历这些节点。
> **二叉索引树是如何工作的？**
> 这个想法基于所有正整数都可以表示为2的幂的和这一事实。例如，19可以表示为16 + 2 + 1。BITree的每个节点都存储n个元素的和，其中n是2的幂。例如，在上面的第一个图（getSum()的图示）中，前12个元素的和可以通过最后4个元素（从9到12）的和加上前8个元素（从1到8）的和得到。一个数n的二进制表示中设置位的数量是O(Logn)。因此，在getSum()和update()操作中，我们最多遍历O(Logn)个节点。构建的时间复杂度为O(nLogn)，因为它为所有n个元素调用了update()。
> **实现：**
> 以下是二叉索引树的实现。

```python
# Python implementation of Binary Indexed Tree 

# Returns sum of arr[0..index]. This function assumes 
# that the array is preprocessed and partial sums of 
# array elements are stored in BITree[]. 
def getsum(BITTree,i): 
	s = 0 #initialize result 

	# index in BITree[] is 1 more than the index in arr[] 
	i = i+1

	# Traverse ancestors of BITree[index] 
	while i > 0: 

		# Add current element of BITree to sum 
		s += BITTree[i] 

		# Move index to parent node in getSum View 
		i -= i & (-i) 
	return s 

# Updates a node in Binary Index Tree (BITree) at given index 
# in BITree. The given value 'val' is added to BITree[i] and 
# all of its ancestors in tree. 
def updatebit(BITTree , n , i ,v): 

	# index in BITree[] is 1 more than the index in arr[] 
	i += 1

	# Traverse all ancestors and add 'val' 
	while i <= n: 

		# Add 'val' to current node of BI Tree 
		BITTree[i] += v 

		# Update index to that of parent in update View 
		i += i & (-i) 


# Constructs and returns a Binary Indexed Tree for given 
# array of size n. 
def construct(arr, n): 

	# Create and initialize BITree[] as 0 
	BITTree = [0]*(n+1) 

	# Store the actual values in BITree[] using update() 
	for i in range(n): 
		updatebit(BITTree, n, i, arr[i]) 

	# Uncomment below lines to see contents of BITree[] 
	#for i in range(1,n+1): 
	#	 print BITTree[i], 
	return BITTree 


# Driver code to test above methods 
freq = [2, 1, 1, 3, 2, 3, 4, 5, 6, 7, 8, 9] 
BITTree = construct(freq,len(freq)) 
print("Sum of elements in arr[0..5] is " + str(getsum(BITTree,5))) 
freq[3] += 6
updatebit(BITTree, len(freq), 3, 6) 
print("Sum of elements in arr[0..5]"+
					" after update is " + str(getsum(BITTree,5))) 

# This code is contributed by Raju Varshney 
 
```

**Output**

```
Sum of elements in arr[0..5] is 12
Sum of elements in arr[0..5] after update is 18
```

**Time Complexity:** O(NLogN)
**Auxiliary Space:** O(N)

**Can we extend the Binary Indexed Tree to computing the sum of a range in O(Logn) time?** 
Yes. rangeSum(l, r) = getSum(r) – getSum(l-1).
**Applications:** 
The implementation of the arithmetic coding algorithm. The development of the Binary Indexed Tree was primarily motivated by its application in this case. See [this ](http://en.wikipedia.org/wiki/Fenwick_tree#Applications)for more details.
**Example Problems:** 
[Count inversions in an array | Set 3 (Using BIT)](https://www.geeksforgeeks.org/count-inversions-array-set-3-using-bit/) 
[Two Dimensional Binary Indexed Tree or Fenwick Tree](https://www.geeksforgeeks.org/two-dimensional-binary-indexed-tree-or-fenwick-tree/) 
[Counting Triangles in a Rectangular space using BIT](https://www.geeksforgeeks.org/counting-triangles-in-a-rectangular-space-using-2d-bit/)

**References:** 
http://en.wikipedia.org/wiki/Fenwick_tree 
http://community.topcoder.com/tc?module=Static&d1=tutorials&d2=binaryIndexedTrees



[力扣307] 线段树&树状数组，https://zhuanlan.zhihu.com/p/126539401



## 示例LeetCode307.区域和检索 - 数组可修改

https://leetcode.cn/problems/range-sum-query-mutable/

给你一个数组 `nums` ，请你完成两类查询。

1. 其中一类查询要求 **更新** 数组 `nums` 下标对应的值
2. 另一类查询要求返回数组 `nums` 中索引 `left` 和索引 `right` 之间（ **包含** ）的nums元素的 **和** ，其中 `left <= right`

实现 `NumArray` 类：

- `NumArray(int[] nums)` 用整数数组 `nums` 初始化对象
- `void update(int index, int val)` 将 `nums[index]` 的值 **更新** 为 `val`
- `int sumRange(int left, int right)` 返回数组 `nums` 中索引 `left` 和索引 `right` 之间（ **包含** ）的nums元素的 **和** （即，`nums[left] + nums[left + 1], ..., nums[right]`）

 

**示例 1：**

```
输入：
["NumArray", "sumRange", "update", "sumRange"]
[[[1, 3, 5]], [0, 2], [1, 2], [0, 2]]
输出：
[null, 9, null, 8]

解释：
NumArray numArray = new NumArray([1, 3, 5]);
numArray.sumRange(0, 2); // 返回 1 + 3 + 5 = 9
numArray.update(1, 2);   // nums = [1,2,5]
numArray.sumRange(0, 2); // 返回 1 + 2 + 5 = 8
```



## 示例27018: 康托展开

http://cs101.openjudge.cn/practice/27018/

总时间限制: 3000ms 单个测试点时间限制: 2000ms 内存限制: 90112kB
描述
求 1∼N 的一个给定全排列在所有 1∼N 全排列中的排名。结果对 998244353取模。

**输入**
第一行一个正整数 N。

第二行 N 个正整数，表示 1∼N 的一种全排列。
**输出**
一行一个非负整数，表示答案对 998244353 取模的值。
样例输入

```
Sample1 in:
3
2 1 3

Sample1 output:
3
```

样例输出

```
Sample2 in:
4
1 2 4 3

Sample2 output:
2
```

提示: 对于100%数据，$1≤N≤1000000$。
来源: https://www.luogu.com.cn/problem/P5367



思路：容易想到的方法是把所有排列求出来后再进行排序，但事实上有更简单高效的算法来解决这个问题，那就是康托展开。

> **康托展开**是一个全排列到一个自然数的双射，常用于构建特定哈希表时的空间压缩。 康托展开的实质是计算当前排列在所有由小到大全排列中的次序编号，因此是可逆的。即由全排列可得到其次序编号（康托展开），由次序编号可以得到对应的第几个全排列（逆康托展开）。
>
> 康托展开的**表达式为**：
>
> $X＝a_n×(n-1)!＋a_{n-1}×(n-2)!＋…＋a_i×(i-1)!＋…＋a_2×1!＋a_1×0!$
>
> 其中：X 为比当前排列小的全排列个数（X+1即为当前排列的次序编号）；n 表示全排列表达式的字符串长度；$a_i$ 表示原排列表达式中的第 i 位（由右往左数），前面（其右侧） i-1 位数有多少个数的值比它小。

例如求 5 2 3 4 1 在 {1, 2, 3, 4, 5} 生成的排列中的次序可以按如下步骤计算。
从右往左数，i 是5时候，其右侧比5小的数有1、2、3、4这4个数，所以有4×4！。
是2，比2小的数有1一个数，所以有 1×3！。
是3，比3小的数有1一个数，为1×2！。
是4，比4小的数有1一个数，为1×1！。
最后一位数右侧没有比它小的数，为 0×0！＝0。
则 4×4！＋1×3！＋1×2！＋1×1！＝105。
这个 X 只是这个排列之前的排列数，而题目要求这个排列的位置，即 5 2 3 4 1排在第 106 位。

同理，4 3 5 2 1的排列数：3×4!＋2×3!＋2×2!＋1×1!＝89，即 4 3 5 2 1 排在第90位。
因为比4小的数有3个：3、2、1；比3小的数有2个：2、1；比5小的数有2个：2、1；比2小的数有1个：1。

参考代码如下。



```python
MOD = 998244353								# Time Limit Exceeded, 内存7140KB, 时间18924ms
fac = [1]

def cantor_expand(a, n):
    ans = 0
    
    for i in range(1, n + 1):
        count = 0
        for j in range(i + 1, n + 1):
            if a[j] < a[i]:
                count += 1				# 计算有几个比他小的数
        ans = (ans + (count * fac[n - i]) % MOD) % MOD
    return ans + 1

a = [0]
N = int(input())		# 用大写N，因为spyder的debug，执行下一条指令的命令是 n/next。与变量n冲突。

for i in range(1, N + 1):
    fac.append((fac[i - 1] * i) % MOD)		# 整数除法具有分配律

*perm, = map(int, input().split())
a.extend(perm)

print(cantor_expand(a, N))
```



用C++也是超时

```c++
#include<iostream>							// Time Limit Exceeded, 内存960KB, 时间1986ms
using namespace std;

const long long MOD = 998244353;
long long fac[1000005]={1};

int cantor_expand (int a[],int n){
    int i, j, count;
    long long ans = 0 ;

    for(i = 1; i <= n; i ++){
        count = 0;
        for(j = i + 1; j <= n; j ++){
            if(a[j] < a[i]) count ++;						// 计算有几个比它小的数
        }
        ans = (ans + (count * fac[n-i]) % MOD ) % MOD;
    }
    return ans + 1;
}


int a[1000005];

int main()
{
  int N;
  //cin >> N;
  scanf("%d", &N);
  for (int i=1; i<=N; i++){
      fac[i] = (fac[i-1]*i)%MOD;
  }

  for (int i=1; i<=N; i++)
      //cin >> a[i];
      scanf("%d",&a[i]);
  cout << cantor_expand(a,N) << endl;
  return 0;
}
```



### 优化

康托展开用 $O(n^2)$ 算法超时，需要把时间复杂度降到$O(nLogn)$。“计算有几个比他小的数”，时间复杂度由 $O(n)$ 降到 $O(Logn)$。

### 树状数组（Binary Indexed Tree）

实现树状数组的核心部分，包括了三个重要的操作：lowbit、修改和求和。

1. lowbit函数：`lowbit(x)` 是用来计算 `x` 的二进制表示中最低位的 `1` 所对应的值。它的运算规则是利用位运算 `(x & -x)` 来获取 `x` 的最低位 `1` 所对应的值。例如，`lowbit(6)` 的结果是 `2`，因为 `6` 的二进制表示为 `110`，最低位的 `1` 所对应的值是 `2`。

   > `-x` 是 `x` 的补码表示。
   >
   > 对于正整数 `x`，`-x` 的二进制表示是 `x` 的二进制表示取反后加 1。
   >
   > `6` 的二进制表示为 `110`，取反得到 `001`，加 1 得到 `010`。
   >
   > `-6` 的二进制表示为 `11111111111111111111111111111010`（假设 32 位整数）。
   >
   > `6 & -6` 的结果：
   >
   > `110` 与 `11111111111111111111111111111010` 按位与运算，结果为 `010`，即 `2`。

2. update函数：这个函数用于修改树状数组中某个位置的值。参数 `x` 表示要修改的位置，参数 `y` 表示要增加/减少的值。函数使用一个循环将 `x` 的所有对应位置上的值都加上 `y`。具体的操作是首先将 `x` 位置上的值与 `y` 相加，然后通过 `lowbit` 函数找到 `x` 的下一个需要修改的位置，将该位置上的值也加上 `y`，然后继续找下一个位置，直到修改完所有需要修改的位置为止。这样就完成了数组的修改。

3. getsum函数：这个函数用于求解树状数组中某个范围的前缀和。参数 `x` 表示要求解前缀和的位置。函数使用一个循环将 `x` 的所有对应位置上的值累加起来，然后通过 `lowbit` 函数找到 `x` 的上一个位置（即最后一个需要累加的位置），再将该位置上的值累加起来，然后继续找上一个位置，直到累加完所有需要累加的位置为止。这样就得到了从位置 `1` 到位置 `x` 的前缀和。

这就是树状数组的核心操作，通过使用这三个函数，我们可以实现树状数组的各种功能，如求解区间和、单点修改等。

```python
n, MOD, ans = int(input()), 998244353, 1						# 内存69832KB, 时间2847ms
a, fac = list(map(int, input().split())), [1]

tree = [0] * (n + 1)

def lowbit(x):
    return x & -x

def update(x, y):
    while x <= n:
        tree[x] += y
        x += lowbit(x)

def getsum(x):
    tot = 0
    while x:
        tot += tree[x]
        x -= lowbit(x)
    return tot


for i in range(1, n):
    fac.append(fac[i-1] * i % MOD)

for i in range(1, n + 1):
    cnt = getsum(a[i-1])
    update(a[i-1], 1)
    ans = (ans + ((a[i-1] - 1 - cnt) * fac[n - i]) % MOD) % MOD
    
print(ans)
```



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231029152322373.png" alt="image-20231029152322373" style="zoom:67%;" />



### 线段树（Segment tree）

线段树 segment tree 来计算第i位右边比该数还要小的数的个数。

```python
n, MOD, ans = int(input()), 998244353, 1					# 内存69900KB, 时间5162ms
a, fac = list(map(int, input().split())), [1]

tree = [0] * (2*n)


def build(arr):

    # insert leaf nodes in tree
    for i in range(n):
        tree[n + i] = arr[i]

    # build the tree by calculating parents
    for i in range(n - 1, 0, -1):
        tree[i] = tree[i << 1] + tree[i << 1 | 1]


# function to update a tree node
def updateTreeNode(p, value):

    # set value at position p
    tree[p + n] = value
    p = p + n

    # move upward and update parents
    i = p
    while i > 1:

        tree[i >> 1] = tree[i] + tree[i ^ 1]
        i >>= 1


# function to get sum on interval [l, r)
def query(l, r):

    res = 0

    l += n
    r += n

    while l < r:

        if (l & 1):
            res += tree[l]
            l += 1

        if (r & 1):
            r -= 1
            res += tree[r]

        l >>= 1
        r >>= 1

    return res


#build([0]*n)

for i in range(1, n):
    fac.append(fac[i-1] * i % MOD)

for i in range(1, n + 1):
    cnt = query(0, a[i-1])
    updateTreeNode(a[i-1]-1, 1)
    
    ans = (ans + (a[i-1] -1 - cnt) * fac[n - i]) % MOD
    
print(ans)

```



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20231029161854925.png" alt="image-20231029161854925" style="zoom: 50%;" />







# 三、二分法



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



```python
def maxMinJump(L, N, M, rocks):
    # 先将岩石位置排序，并加入起点和终点
    rocks = [0] + rocks + [L]

    left, right = 0, L + 1  # 可能的最小跳跃距离范围。所以二分是在 [left, right) 区间内进行的

    def canAchieve(min_dist):
        """ 判断是否能移除至多 M 个岩石，使最短跳跃距离至少为 min_dist """
        removed = 0  # 记录移除的岩石数量
        prev = 0  # 记录上一个岩石位置（起点）

        for i in range(1, len(rocks)):
            if rocks[i] - rocks[prev] < min_dist:
                removed += 1
                if removed > M:
                    return False  # 不能满足要求
            else:
                prev = i  # 更新上一个岩石位置

        return True  # 可以满足要求

    # 二分查找最长可能的最短跳跃距离
    ans = -1
    while left < right:
        mid = (left + right) // 2  # 取中间偏右值
        if canAchieve(mid):
            ans = mid   # 记录可行的 `mid`
            left = mid + 1  # 继续尝试更大的值
        else:
            right = mid

    return ans


# 读取输入
L, N, M = map(int, input().split())
rocks = [int(input()) for _ in range(N)]

# 计算并输出答案
print(maxMinJump(L, N, M, rocks))

```





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
lo, hi = 0, L+1 # 左闭右开写法
ans = -1
while lo < hi:
    mid = (lo + hi) // 2
    
    if check(mid):
        hi = mid
    else:               
        ans = mid       # 记录可行的 `mid`
        lo = mid + 1		# 继续尝试更大的值
        
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





```python
def minMaxMonthlyExpense(N, M, expenses):
    def can_split(max_expense):
        """ 判断是否能合并至多 M 个花费，使最大花费不超过 max_expense """
        months = 1  # 记录当前使用的月份数
        current_sum = 0 # 当前月的开销
        for cost in expenses:
            if current_sum + cost > max_expense:
                months += 1
                if months > M:
                    return False
                current_sum = cost
            else:
                current_sum += cost
        return True

    # 可能的最小开销范围。所以二分是在 [left, right) 区间内进行的
    left, right = max(expenses), sum(expenses) + 1
    ans = -1
    while left < right: # 二分查找最小的 "最大月度开销"
        mid = (left + right) // 2
        if can_split(mid):
            ans = mid   # 记录可行的 `mid`
            right = mid # 继续尝试更小的值
        else:
            left = mid + 1
    return ans

# 读取输入
N, M = map(int, input().split())
expenses = [int(input()) for _ in range(N)]

# 计算并输出答案
print(minMaxMonthlyExpense(N, M, expenses))
```





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
        ans = mid   # 记录可行的 `mid`
        hi = mid		# 继续尝试更小的值
        
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







# 四、基数排序

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





# 五、字典与检索

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



# 六、B-trees

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
