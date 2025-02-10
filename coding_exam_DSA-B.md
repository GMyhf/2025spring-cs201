# Coding Exam 上机考试

Updated 1742 GMT+8 Feb 10 2025  
2025 winter, Complied by Hongfei Yan



# cs201 2024 Final Exam

http://cs101.openjudge.cn/cs2012024feclass12/



E27933:Okabe and Boxes

http://cs101.openjudge.cn/practice/27933/

E28186:分糖果

http://cs101.openjudge.cn/practice/28186/

M28012:受限条件下可到达节点的数目

http://cs101.openjudge.cn/practice/28012/

M28013:堆路径

http://cs101.openjudge.cn/practice/28013/

M28193:谣言

http://cs101.openjudge.cn/practice/28193/

M28276:判断等价关系是否成立

http://cs101.openjudge.cn/practice/28276/





# 20240508 cs201 2024 Mock Exam

http://cs101.openjudge.cn/20240508mockexam/



E02808:校门外的树

http://cs101.openjudge.cn/practice/02808/

E20449:是否被5整除

http://cs101.openjudge.cn/practice/20449/

M01258:Agri-Net

http://cs101.openjudge.cn/practice/01258/

M27635:判断无向图是否连通有无回路

http://cs101.openjudge.cn/practice/27635/

T27947:动态中位数

http://cs101.openjudge.cn/practice/27947/

T28190:奶牛排队

http://cs101.openjudge.cn/practice/28190/





# 20240403 cs201 2024 Mock Exam

http://cs101.openjudge.cn/20240403mockexam/



E27706:逐词倒放

http://cs101.openjudge.cn/practice/27706/

E27951:机器翻译

http://cs101.openjudge.cn/practice/27951/

M27932:Less or Equal

http://cs101.openjudge.cn/practice/27932/

M27948:FBI树

http://cs101.openjudge.cn/practice/27948/

T27925:小组队列

http://cs101.openjudge.cn/practice/27925/

T27928:遍历树

http://cs101.openjudge.cn/practice/27928/







# 20240306 cs201 2024 Mock Exam

http://cs101.openjudge.cn/20240306mockexam/



E02945:拦截导弹

http://cs101.openjudge.cn/practice/02945/

E04147:汉诺塔问题(Tower of Hanoi)

http://cs101.openjudge.cn/practice/04147/

M03253:约瑟夫问题No.2

http://cs101.openjudge.cn/practice/03253/

M21554:排队做实验(greedy)v0.2

http://cs101.openjudge.cn/practice/21554/

T19963:买学区房

http://cs101.openjudge.cn/practice/19963/

T27300:模型整理
http://cs101.openjudge.cn/practice/27300/









http://cs101.openjudge.cn/practice//



# 数算B必会简单题

Updated 0020 GMT+8 Jun 4, 2024

2024 spring, Complied by Hongfei Yan



取自gw班，http://dsbpython.openjudge.cn/easyprbs/



| 题目                                  | tags             |
| ------------------------------------- | ---------------- |
| 22782: PKU版爱消除                    | stack            |
| 26590: 检测括号嵌套                   | stack            |
| 26571: 我想完成数算作业：代码         | disjoint set     |
| 20169: 排队                           | disjoint set     |
| 24744: 想要插队的Y君                  | Linked List      |
| 25143/27638: 求二叉树的高度和叶子数目 | tree             |
| 25155: 深度优先遍历一个无向图         | dfs              |
| 22508: 最小奖金方案                   | topological sort |



## 001: PKU版爱消除

http://dsbpython.openjudge.cn/easyprbs/001/

你有⼀个字符串S，⼤小写区分，⼀旦⾥⾯出现连续的PKU三个字符，就会消除。问最终稳定下来以后，这个字符串是什么样的？

输入

⼀⾏,⼀个字符串S，表⽰消除前的字符串。
字符串S的⻓度不超过100000，且只包含⼤小写字⺟。

输出

⼀⾏,⼀个字符串T，表⽰消除后的稳定字符串

样例输入

```
TopSchoolPPKPPKUKUUPKUku
```

样例输出

```
TopSchoolPku
```

提示

请注意看样例。PKU消除后导致出现连续PKU，还要继续消除
比如APKPKUUB，消除中间PKU后，又得到PKU，就接着消除得到AB
此题用栈解决

来源

Chen Jiali



```python
s = input()
stack = []
for c in s:
    if c == "U":
        if len(stack) >= 2 and stack[-1] == "K" and stack[-2] == "P":
            stack.pop()
            stack.pop()
        else:
            stack.append(c)
    else:
        stack.append(c)
print("".join(stack))
```



## 002: 检测括号嵌套

http://dsbpython.openjudge.cn/easyprbs/002/

字符串中可能有3种成对的括号，"( )"、"[ ]"、"{}"。请判断字符串的括号是否都正确配对以及有无括号嵌套。无括号也算正确配对。括号交叉算不正确配对，例如"1234[78)ab]"就不算正确配对。一对括号被包含在另一对括号里面，例如"12(ab[8])"就算括号嵌套。括号嵌套不影响配对的正确性。 给定一个字符串: 如果括号没有正确配对，则输出 "ERROR" 如果正确配对了，且有括号嵌套现象，则输出"YES" 如果正确配对了，但是没有括号嵌套现象，则输出"NO"   

输入

一个字符串，长度不超过5000,仅由 ( ) [ ] { } 和小写英文字母以及数字构成

输出

根据实际情况输出 ERROR, YES 或NO

样例输入

```
样例1:
[](){}
样例2:
[(a)]bv[]
样例3:
[[(])]{}
```

样例输出

```
样例1:
NO
样例2:
YES
样例3:
ERROR
```





```python
def check_brackets(s):
    stack = []
    nested = False
    pairs = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in pairs.values():
            stack.append(ch)
        elif ch in pairs.keys():
            if not stack or stack.pop() != pairs[ch]:
                return "ERROR"
            if stack:
                nested = True
    if stack:
        return "ERROR"
    return "YES" if nested else "NO"

s = input()
print(check_brackets(s))
```





## 003: 我想完成数算作业：代码

当卷王小D睡前意识到室友们每天熬夜吐槽的是自己也选了的课时，他距离早八随堂交的ddl只剩下了不到4小时。已经debug一晚上无果的小D有心要分无力做题，于是决定直接抄一份室友的作业完事。万万没想到，他们作业里完全一致的错误，引发了一场全面的作业查重……

假设a和b作业雷同，b和c作业雷同，则a和c作业雷同。所有抄袭现象都会被发现，且雷同的作业只有一份独立完成的原版，请输出独立完成作业的人数

输入

第一行输入两个正整数表示班上的人数n与总比对数m，接下来m行每行均为两个1-n中的整数i和j，表明第i个同学与第j个同学的作业雷同。

输出

独立完成作业的人数

样例输入

```
3 2
1 2
1 3
样例2：
4 2
2 4
1 3
```

样例输出

```
样例1：
1
样例2:
2
```





```python
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if xroot != yroot:
        parent[xroot] = yroot

n, m = map(int, input().split())
parent = list(range(n + 1))
for _ in range(m):
    i, j = map(int, input().split())
    union(parent, i, j)

count = sum(i == parent[i] for i in range(1, n + 1))
print(count)
```





## 004: 排队

http://cs101.openjudge.cn/practice/20169/

操场上有好多好多同学在玩耍，体育老师冲了过来，要求他们排队。同学们纪律实在太散漫了，老师不得不来手动整队：
"A，你站在B的后面。" 
"C，你站在D的后面。"
"B，你站在D的后面。哦，去D队伍的最后面。" 

更形式化地，初始时刻，操场上有 n 位同学，自成一列。每次操作，老师的指令是 "x y"，表示 x 所在的队列排到 y 所在的队列的后面，即 x 的队首排在 y 的队尾的后面。（如果 x 与 y 已经在同一队列，请忽略该指令） 最终的队列数量远远小于 n，老师很满意。请你输出最终时刻每位同学所在队列的队首（排头），老师想记录每位同学的排头，方便找人。

输入

第一行一个整数 T (T≤5)，表示测试数据组数。 接下来 T 组测试数据，对于每组数据，第一行两个整数 n 和 m (n,m≤30000)，紧跟着 m 行每行两个整数 
x 和 y (1≤x,y≤n)。

输出

共 T 行。 每行 n 个整数，表示每位同学的排头。

样例输入

```
2
4 2
1 2
3 4
5 4
1 2
2 3
4 5
1 3
```

样例输出

```
2 2 4 4
3 3 3 5 5
```





```python
def getRoot(a):
    if parent[a] != a:
        parent[a] = getRoot(parent[a])
    return parent[a]


def merge(a, b):
    pa = getRoot(a)
    pb = getRoot(b)
    if pa != pb:
        parent[pa] = parent[pb]


t = int(input())
for i in range(t):
    n, m = map(int, input().split())
    parent = [i for i in range(n + 10)]
    for i in range(m):
        x, y = map(int, input().split())
        merge(x, y)
    for i in range(1, n + 1):
        print(getRoot(i), end=" ")
    # 注意，一定不能写成 print(parent[i],end= " ")
    # 因为只有执行路径压缩getRoot(i)以后，parent[i]才会是i的树根
    print()
```





## 005: 想要插队的Y君

http://dsbpython.openjudge.cn/easyprbs/005/

很遗憾，一意孤行的Y君没有理会你告诉他的饮食计划并很快吃完了他的粮食储备。
但好在他捡到了一张校园卡，凭这个他可以偷偷混入领取物资的队伍。
为了不被志愿者察觉自己是只猫，他想要插到队伍的最中央。（插入后若有偶数个元素则选取靠后的位置）
于是他又找到了你，希望你能帮他修改志愿者写好的代码，在发放顺序的中间加上他的学号6。
你虽然不理解志愿者为什么要用链表来写这份代码，但为了不被发现只得在此基础上进行修改：

```
class Node:
	def __init__(self, data, next=None):
		self.data, self.next = data, next

class LinkList:
	def __init__(self):
		self.head = None

	def initList(self, data):
		self.head = Node(data[0])
		p = self.head
		for i in data[1:]:
			node = Node(i)
			p.next = node
			p = p.next

	def insertCat(self):
// 在此处补充你的代码
########            
	def printLk(self):
		p = self.head
		while p:
			print(p.data, end=" ")
			p = p.next
		print()

lst = list(map(int,input().split()))
lkList = LinkList()
lkList.initList(lst)
lkList.insertCat()
lkList.printLk()
```

输入

一行，若干个整数，组成一个链表。

输出

一行，在链表中间位置插入数字6后得到的新链表

样例输入

```
### 样例输入1
8 1 0 9 7 5
### 样例输入2
1 2 3
```

样例输出

```
### 样例输出1
8 1 0 6 9 7 5
### 样例输出2
1 2 6 3
```

来源

Lou Yuke





```python
class Node:
	def __init__(self, data, next=None):
		self.data, self.next = data, next

class LinkList:
	def __init__(self):
		self.head = None

	def initList(self, data):
		self.head = Node(data[0])
		p = self.head
		for i in data[1:]:
			node = Node(i)
			p.next = node
			p = p.next

	def insertCat(self):
#your code starts here
		ptr = self.head
		total = 0
		while ptr is not None:
			total += 1
			ptr = ptr.next
		if total % 2 == 0:
			pos = total // 2
		else:
			pos = total // 2 + 1
		ptr = self.head
		for i in range(pos-1):
			ptr = ptr.next
		nd = Node(6)
		nd.next = ptr.next
		ptr.next = nd
########            
	def printLk(self):
		p = self.head
		while p:
			print(p.data, end=" ")
			p = p.next
		print()

lst = list(map(int,input().split()))
lkList = LinkList()
lkList.initList(lst)
lkList.insertCat()
lkList.printLk()
```



```python
# 求二叉树的高度和叶子数目	2022-09-06 20:36:28
class BinaryTree:
    def __init__(self, data, left=None, right=None):
        self.data, self.left, self.right = data, left, right

    def addLeft(self, tree):  # tree是一个二叉树
        self.left = tree

    def addRight(self, tree):  # tree是一个二叉树
        self.right = tree

    def preorderTraversal(self, op):  # 前序遍历,对本题无用 op是函数，表示访问操作
        op(self)  # 访问根结点
        if self.left:  # 左子树不为空
            self.left.preorderTraversal(op)  # 遍历左子树
        if self.right:
            self.right.preorderTraversal(op)  # 遍历右子

    def inorderTraversal(self, op):  # 中序遍历， 对本题无用
        if self.left:
            self.left.inorderTraversal(op)
        op(self)
        if self.right:
            self.right.inorderTraversal(op)

    def postorderTraversal(self, op):  # 后序遍历， 对本题无用
        if self.left:
            self.left.postorderTraversal(op)
        if self.right:
            self.right.postorderTraversal(op)
        op(self)

    def bfsTraversal(self, op):  # 按层次遍历，对本题无用
        import collections
        dq = collections.deque()
        dq.append(self)
        while len(dq) > 0:
            nd = dq.popleft()
            op(nd)
            if nd.left:
                dq.append(nd.left)
            if nd.right:
                dq.append(nd.right)

    def countLevels(self):  # 算有多少层结点
        def count(root):
            if root is None:
                return 0
            return 1 + max(count(root.left), count(root.right))

        return count(self)

    def countLeaves(self):  # 算叶子数目
        def count(root):
            if root.left is None and root.right is None:
                return 1
            elif root.left is not None and root.right is None:
                return count(root.left)
            elif root.left is None and root.right is not None:
                return count(root.right)
            else:
                return count(root.right) + count(root.left)

        return count(self)

    def countWidth(self):  # 求宽度，对本题无用
        dt = {}

        def traversal(root, level):
            if root is None:
                return
            dt[level] = dt.get(level, 0) + 1
            traversal(root.left, level + 1)
            traversal(root.right, level + 1)

        traversal(self, 0)
        width = 0
        for x in dt.items():
            width = max(width, x[1])
        return width


def buildTree(n):
    nodes = [BinaryTree(None) for i in range(n)]
    isRoot = [True] * n
    # 树描述： 结点编号从0开始
    # 1 2
    # -1 -1
    # -1 -1
    for i in range(n):
        L, R = map(int, input().split())
        nd = i
        nodes[nd].data = nd
        if L != -1:
            nodes[nd].left = nodes[L]
            isRoot[L] = False
        if R != -1:
            nodes[nd].right = nodes[R]
            isRoot[R] = False
    for i in range(n):
        if isRoot[i]:
            return nodes[i]
    return None


n = int(input())
tree = buildTree(n)
print(tree.countLevels() - 1, tree.countLeaves())

```



## 006: 求二叉树的高度和叶子数目

http://cs101.openjudge.cn/dsapre/27638/

http://dsbpython.openjudge.cn/easyprbs/006/

给定一棵二叉树，求该二叉树的高度和叶子数目

二叉树高度定义：从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的结点数减1为树的高度。只有一个结点的二叉树，高度是0。





输入

第一行是一个整数n，表示二叉树的结点个数。二叉树结点编号从0到n-1。n <= 100
接下来有n行，依次对应二叉树的编号为0,1,2....n-1的节点。
每行有两个整数，分别表示该节点的左儿子和右儿子的编号。如果第一个（第二个）数为-1则表示没有左（右）儿子

输出

在一行中输出2个整数，分别表示二叉树的高度和叶子结点个数

样例输入

```
3
-1 -1
0 2
-1 -1
```

样例输出

```
1 2
```

来源

Guo Wei



由于输入无法分辨谁为根节点，所以写寻找根节点语句。

```python
class TreeNode:
    def __init__(self):
        self.left = None
        self.right = None

def tree_height(node):
    if node is None:
        return -1  # 根据定义，空树高度为-1
    return max(tree_height(node.left), tree_height(node.right)) + 1

def count_leaves(node):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)

n = int(input())  # 读取节点数量
nodes = [TreeNode() for _ in range(n)]
has_parent = [False] * n  # 用来标记节点是否有父节点

for i in range(n):
    left_index, right_index = map(int, input().split())
    if left_index != -1:
        nodes[i].left = nodes[left_index]
        has_parent[left_index] = True
    if right_index != -1:
        #print(right_index)
        nodes[i].right = nodes[right_index]
        has_parent[right_index] = True

# 寻找根节点，也就是没有父节点的节点
root_index = has_parent.index(False)
root = nodes[root_index]

# 计算高度和叶子节点数
height = tree_height(root)
leaves = count_leaves(root)

print(f"{height} {leaves}")
```





注意：需要找根节点

```python
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def build_tree(node_descriptions):
    nodes = {i: TreeNode(i) for i in range(len(node_descriptions))}
    child_set = set()

    for i, (left, right) in enumerate(node_descriptions):
        if left != -1:
            nodes[i].left = nodes[left]
            child_set.add(left)
        if right != -1:
            nodes[i].right = nodes[right]
            child_set.add(right)

    # Root is the node that is not anyone's child
    root = next(node for node in nodes.values() if node.val not in child_set)
    return root


def tree_height_and_leaf_count(root):
    if not root:
        return 0, 0  # height is 0 for empty tree, no leaves

    def dfs(node):
        if not node:
            return -1, 0

        if not node.left and not node.right:
            return 0, 1

        left_height, left_leaves = dfs(node.left)
        right_height, right_leaves = dfs(node.right)

        current_height = 1 + max(left_height, right_height)
        current_leaves = left_leaves + right_leaves

        return current_height, current_leaves

    height, leaf_count = dfs(root)
    return height, leaf_count


n = int(input())
node_descriptions = [tuple(map(int, input().split())) for _ in range(n)]

root = build_tree(node_descriptions)
height, leaf_count = tree_height_and_leaf_count(root)

print(height, leaf_count)
```





## 007: 深度优先遍历一个无向图

http://dsbpython.openjudge.cn/easyprbs/007/

输出无向图深度优先遍历序列

 

 

输入

第一行是整数n和m(0 < n <=16)，表示无向图有n个顶点，m条边，顶点编号0到n-1。接下来m行，每行两个整数a,b，表示顶点a,b之间有一条边。

输出

任意一个深度优先遍历序列

样例输入

```
9 9
0 1
0 2
3 0
2 1
1 5
1 4
4 5
6 3
8 7
```

样例输出

```
0 1 2 4 5 3 6 8 7
```

提示

题目需要Special Judge。所以输出错误答案也可能导致Runtime Error

来源

Guo Wei



```python
def dfs(graph, visited, node):
    visited[node] = True
    print(node, end=" ")

    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs(graph, visited, neighbor)

def main():
    n, m = map(int, input().split())
    graph = [[] for _ in range(n)]
    visited = [False] * n

    for _ in range(m):
        a, b = map(int, input().split())
        graph[a].append(b)
        graph[b].append(a)

    for i in range(n):
        if not visited[i]:
            dfs(graph, visited, i)

if __name__ == "__main__":
    main()
```



```python
def dfsTravel(G,op): #G是邻接表
	def dfs(v):
		visited[v] = True
		op(v)
		for u in G[v]:
			if not visited[u]:
				dfs(u)
	n = len(G)  # 顶点数目
	visited = [False for i in range(n)]
	for i in range(n):  # 顶点编号0到n-1
		if not visited[i]:
			dfs(i)

n,m = map(int,input().split())
G = [[] for i in range(n)]
for i in range(m):
	s,e = map(int,input().split())
	G[s].append(e)
	G[e].append(s)
dfsTravel(G,lambda x:print(x,end = " "))
```





## 008: 最小奖金方案

http://dsbpython.openjudge.cn/easyprbs/008/

现在有n个队伍参加了比赛，他们进行了m次PK。现在赛事方需要给他们颁奖（奖金为整数），已知参加比赛就可获得100元，由于比赛双方会比较自己的奖金，所以获胜方的奖金一定要比败方奖金高。请问赛事方要准备的最小奖金为多少？奖金数额一定是整数。

输入

一组数据，第一行是两个整数n(1≤n≤1000)和m(0≤m≤2000)，分别代表n个队伍和m次pk，队伍编号从0到n-1。接下来m行是pk信息，具体信息a，b，代表编号为a的队伍打败了编号为b的队伍。
输入保证队伍之间的pk战胜关系不会形成有向环

输出

给出最小奖金w

样例输入

```
5 6
1 0
2 0
3 0
4 1
4 2
4 3
```

样例输出

```
505
```

来源

陈鑫



```python
import collections
n,m = map(int,input().split())
G = [[] for i in range(n)]
award = [0 for i in range(n)]
inDegree = [0 for i in range(n)]

for i in range(m):
	a,b = map(int,input().split())
	G[b].append(a)
	inDegree[a] += 1
q = collections.deque()
for i in range(n):
	if inDegree[i] == 0:
		q.append(i)
		award[i] = 100
while len(q) > 0:
	u = q.popleft()
	for v in G[u]:
		inDegree[v] -= 1
		award[v] = max(award[v],award[u] + 1)
		if inDegree[v] == 0:
			q.append(v)
total = sum(award)
print(total)
```











# 2024北京大学计算机学院优秀大学生暑期夏令营机试

http://bailian.openjudge.cn/xly2024062702/

2024-06-27 18:00~20:00



| 题目               | tags             |
| ------------------ | ---------------- |
| 金币               | implementation   |
| 话题焦点人物       | Implementation   |
| 算24               | bruteforce       |
| 怪盗基德的滑翔翼   | dp               |
| 合并编码           | Stack            |
| Sorting It All Out | topological sort |
| A Bug's Life       | disjoint set     |
| 正方形破坏者       | IDA*             |







## 02000: 金币

http://cs101.openjudge.cn/practice/02000/

国王将金币作为工资，发放给忠诚的骑士。第一天，骑士收到一枚金币；之后两天（第二天和第三天）里，每天收到两枚金币；之后三天（第四、五、六天）里，每天收到三枚金币；之后四天（第七、八、九、十天）里，每天收到四枚金币……这种工资发放模式会一直这样延续下去：当连续N天每天收到N枚金币后，骑士会在之后的连续N+1天里，每天收到N+1枚金币（N为任意正整数）。

你需要编写一个程序，确定从第一天开始的给定天数内，骑士一共获得了多少金币。

输入

输入包含至少一行，但不多于21行。除最后一行外，输入的每行是一组输入数据，包含一个整数（范围1到10000），表示天数。输入的最后一行为0，表示输入结束。

输出

对每个数据输出一行，包含该数据对应天数和总金币数，用单个空格隔开。

样例输入

```
10
6
7
11
15
16
100
10000
1000
21
22
0
```

样例输出

```
10 30
6 14
7 18
11 35
15 55
16 61
100 945
10000 942820
1000 29820
21 91
22 98
```

来源

Rocky Mountain 2004



```python
def calculate_gold_days(days):
    total_gold = 0
    days_count = 1
    while days > 0:
        for _ in range(days_count):
            if days == 0:
                break
            total_gold += days_count
            days -= 1
        days_count += 1
    return total_gold

while True:
    days = int(input())
    if days == 0:
        break
    total_gold = calculate_gold_days(days)
    print(f"{days} {total_gold}")
```





## 06901: 话题焦点人物

http://cs101.openjudge.cn/practice/06901/

微博提供了一种便捷的交流平台。一条微博中，可以提及其它用户。例如Lee发出一条微博为：“期末考试顺利 @Kim @Neo”，则Lee提及了Kim和Neo两位用户。

我们收集了N(1 < N < 10000)条微博，并已将其中的用户名提取出来，用小于等于100的正整数表示。

通过分析这些数据，我们希望发现大家的话题焦点人物，即被提及最多的人（题目保证这样的人有且只有一个），并找出那些提及它的人。

**输入**

输入共两部分：
第一部分是微博数量N，1 < N < 10000。
第二部分是N条微博，每条微博占一行，表示为：
发送者序号a，提及人数k(0 < = k < = 20)，然后是k个被提及者序号b1,b2...bk；
其中a和b1,b2...bk均为大于0小于等于100的整数。相邻两个整数之间用单个空格分隔。

**输出**

输出分两行：
第一行是被提及最多的人的序号；
第二行是提及它的人的序号，从小到大输出，相邻两个数之间用单个空格分隔。同一个序号只输出一次。

样例输入

```
5
1 2 3 4
1 0
90 3 1 2 4
4 2 3 2
2 1 3
```

样例输出

```
3
1 2 4
```

来源

医学部计算概论2011年期末考试（谢佳亮）



```python
def find_topic_center_and_mentioners():
    n = int(input())
    mention_count = {}  # 记录每个人被提及的次数
    mention_relations = {}  # 记录提及关系，key为提及的人，value为提及的人的集合
    
    for _ in range(n):
        tweet = input().split()
        sender, k = int(tweet[0]), int(tweet[1])
        if k > 0:
            mentioned = list(map(int, tweet[2:]))
            for person in mentioned:
                if person not in mention_count:
                    mention_count[person] = 1
                    mention_relations[person] = set([sender])
                else:
                    mention_count[person] += 1
                    mention_relations[person].add(sender)
    
    # 找到被提及最多的人
    topic_center = max(mention_count, key=mention_count.get)
    
    # 输出结果
    print(topic_center)
    print(' '.join(map(str, sorted(mention_relations[topic_center]))))

# 调用函数处理输入数据
find_topic_center_and_mentioners()
```



## 02787: 算24

http://cs101.openjudge.cn/practice/02787/

给出4个小于10个正整数，你可以使用加减乘除4种运算以及括号把这4个数连接起来得到一个表达式。现在的问题是，是否存在一种方式使得得到的表达式的结果等于24。

这里加减乘除以及括号的运算结果和运算的优先级跟我们平常的定义一致（这里的除法定义是实数除法）。

比如，对于5，5，5，1，我们知道5 * (5 – 1 / 5) = 24，因此可以得到24。又比如，对于1，1，4，2，我们怎么都不能得到24。

输入

输入数据包括多行，每行给出一组测试数据，包括4个小于10个正整数。最后一组测试数据中包括4个0，表示输入的结束，这组数据不用处理。

输出

对于每一组测试数据，输出一行，如果可以得到24，输出“YES”；否则，输出“NO”。

样例输入

```
5 5 5 1
1 1 4 2
0 0 0 0
```

样例输出

```
YES
NO
```





```python
'''
在这个优化的代码中，我们使用了递归和剪枝策略。首先按照题目的要求，输入的4个数字保持不变，
不进行排序。在每一次运算中，我们首先尝试加法和乘法，因为它们的运算结果更少受到数字大小的影响。
然后，我们根据数字的大小关系尝试减法和除法，只进行必要的组合运算，避免重复运算。

值得注意的是，这种优化策略可以减少冗余计算，但对于某些输入情况仍需要遍历所有可能的组合。
因此，在最坏情况下仍然可能需要较长的计算时间。
'''

from functools import lru_cache 

@lru_cache(maxsize = None)
def find(nums):
    if len(nums) == 1:
        return abs(nums[0] - 24) <= 0.000001

    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            a = nums[i]
            b = nums[j]
            remaining_nums = []

            for k in range(len(nums)):
                if k != i and k != j:
                    remaining_nums.append(nums[k])

            # 尝试加法和乘法运算
            if find(tuple(remaining_nums + [a + b])) or find(tuple(remaining_nums + [a * b])):
                return True

            # 尝试减法运算
            if a > b and find(tuple(remaining_nums + [a - b])):
                return True
            if b > a and find(tuple(remaining_nums + [b - a])):
                return True

            # 尝试除法运算
            if b != 0 and find(tuple(remaining_nums + [a / b])):
                return True
            if a != 0 and find(tuple(remaining_nums + [b / a])):
                return True

    return False

while True:
    card = [int(x) for x in input().split()]
    if sum(card) == 0:
        break

    print("YES" if find(tuple(card)) else "NO")
```





## 04977: 怪盗基德的滑翔翼

http://cs101.openjudge.cn/practice/04977/

怪盗基德是一个充满传奇色彩的怪盗，专门以珠宝为目标的超级盗窃犯。而他最为突出的地方，就是他每次都能逃脱中村警部的重重围堵，而这也很大程度上是多亏了他随身携带的便于操作的滑翔翼。

有一天，怪盗基德像往常一样偷走了一颗珍贵的钻石，不料却被柯南小朋友识破了伪装，而他的滑翔翼的动力装置也被柯南踢出的足球破坏了。不得已，怪盗基德只能操作受损的滑翔翼逃脱。

![img](http://media.openjudge.cn/images/upload/1340073200.jpg)

假设城市中一共有N幢建筑排成一条线，每幢建筑的高度各不相同。初始时，怪盗基德可以在任何一幢建筑的顶端。他可以选择一个方向逃跑，但是不能中途改变方向（因为中森警部会在后面追击）。因为滑翔翼动力装置受损，他只能往下滑行（即：只能从较高的建筑滑翔到较低的建筑）。他希望尽可能多地经过不同建筑的顶部，这样可以减缓下降时的冲击力，减少受伤的可能性。请问，他最多可以经过多少幢不同建筑的顶部（包含初始时的建筑）？



输入

输入数据第一行是一个整数K（K < 100），代表有K组测试数据。
每组测试数据包含两行：第一行是一个整数N(N < 100)，代表有N幢建筑。第二行包含N个不同的整数，每一个对应一幢建筑的高度h（0 < h < 10000），按照建筑的排列顺序给出。

输出

对于每一组测试数据，输出一行，包含一个整数，代表怪盗基德最多可以经过的建筑数量。

样例输入

```
3
8
300 207 155 299 298 170 158 65
8
65 158 170 298 299 155 207 300
10
2 1 3 4 5 6 7 8 9 10
```

样例输出

```
6
6
9
```





```python
def max_increasing_subsequence(a):
    n = len(a)
    dpu = [1] * n
    for i in range(1, n):
        for j in range(i):
            if a[i] > a[j]:
                dpu[i] = max(dpu[i], dpu[j] + 1)
    return max(dpu)

def max_decreasing_subsequence(a):
    n = len(a)
    dpd = [1] * n
    for i in range(1, n):
        for j in range(i):
            if a[i] < a[j]:
                dpd[i] = max(dpd[i], dpd[j] + 1)
    return max(dpd)

def main():
    k = int(input())
    while k:
        k -= 1
        n = int(input())
        a = list(map(int, input().split()))
        mxu = max_increasing_subsequence(a)
        mxd = max_decreasing_subsequence(a)
        print(max(mxu, mxd))

if __name__ == "__main__":
    main()
```





## 28496: 合并编码

给定一个合并编码后的字符串，返回原始字符串。

编码规则为: k[string]，表示其中方括号内部的string重复 k 次，k为正整数且0 < k < 20。
可以认为输入数据中所有的数字只表示重复的次数k，且方括号总是满足要求的，例如不会出现像 3a 、 2[b 或 2[4] 的输入。

输入保证符合格式，且不包含空格。

输入

一行，一个长度为n(1 ≤ n ≤ 500)的字符串，代表合并编码后的字符串。

输出

一行，代表原始字符串。长度m满足0 ≤ m ≤ 1500。

样例输入

```
样例1：
3[abc]1[o]2[n]

样例2：
3[a2[c]]
```

样例输出

```
样例1：
abcabcabconn

样例2：
accaccacc
```



```python
def decode_string(s):
    stack = []
    current_num = 0
    current_str = ''
    result = ''

    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            stack.append(current_str)
            stack.append(current_num)
            current_str = ''
            current_num = 0
        elif char == ']':
            num = stack.pop()
            prev_str = stack.pop()
            current_str = prev_str + num * current_str
        else:
            current_str += char

    return current_str

# 示例
input_str1 = "3[abc]1[o]2[n]"
input_str2 = "3[a2[c]]"

print(decode_string(input_str1))  # 应输出: abcabcabconn
print(decode_string(input_str2))  # 应输出: accaccacc
```





## 01094: Sorting It All Out

http://cs101.openjudge.cn/practice/01094/

An ascending sorted sequence of distinct values is one in which some form of a less-than operator is used to order the elements from smallest to largest. For example, the sorted sequence A, B, C, D implies that A < B, B < C and C < D. in this problem, we will give you a set of relations of the form A < B and ask you to determine whether a sorted order has been specified or not. 

输入

Input consists of multiple problem instances. Each instance starts with a line containing two positive integers n and m. the first value indicated the number of objects to sort, where 2 <= n <= 26. The objects to be sorted will be the first n characters of the uppercase alphabet. The second value m indicates the number of relations of the form A < B which will be given in this problem instance. Next will be m lines, each containing one such relation consisting of three characters: an uppercase letter, the character "<" and a second uppercase letter. No letter will be outside the range of the first n letters of the alphabet. Values of n = m = 0 indicate end of input.

输出

For each problem instance, output consists of one line. This line should be one of the following three:

Sorted sequence determined after xxx relations: yyy...y.
Sorted sequence cannot be determined.
Inconsistency found after xxx relations.

where xxx is the number of relations processed at the time either a sorted sequence is determined or an inconsistency is found, whichever comes first, and yyy...y is the sorted, ascending sequence.

样例输入

```
4 6
A<B
A<C
B<C
C<D
B<D
A<B
3 2
A<B
B<A
26 1
A<Z
0 0
```

样例输出

```
Sorted sequence determined after 4 relations: ABCD.
Inconsistency found after 2 relations.
Sorted sequence cannot be determined.
```

来源

East Central North America 2001





```python
#23n2310307206胡景博
from collections import deque
def topo_sort(graph):
    in_degree = {u:0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    q = deque([u for u in in_degree if in_degree[u] == 0])
    topo_order = [];flag = True
    while q:
        if len(q) > 1:
            flag = False#topo_sort不唯一确定
        u = q.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    if len(topo_order) != len(graph): return 0
    return topo_order if flag else None
while True:
    n,m = map(int,input().split())
    if n == 0: break
    graph = {chr(x+65):[] for x in range(n)}
    edges = [tuple(input().split('<')) for _ in range(m)]
    for i in range(m):
        a,b = edges[i]
        graph[a].append(b)
        t = topo_sort(graph)
        if t:
            s = ''.join(t)
            print("Sorted sequence determined after {} relations: {}.".format(i+1,s))
            break
        elif t == 0:
            print("Inconsistency found after {} relations.".format(i+1))
            break
    else:
        print("Sorted sequence cannot be determined.")
```





## 02492: A Bug's Life

http://cs101.openjudge.cn/practice/02492/

**Background**
Professor Hopper is researching the sexual behavior of a rare species of bugs. He assumes that they feature two different genders and that they only interact with bugs of the opposite gender. In his experiment, individual bugs and their interactions were easy to identify, because numbers were printed on their backs.
**Problem**
Given a list of bug interactions, decide whether the experiment supports his assumption of two genders with no homosexual bugs or if it contains some bug interactions that falsify it.

输入

The first line of the input contains the number of scenarios. Each scenario starts with one line giving the number of bugs (at least one, and up to 2000) and the number of interactions (up to 1000000) separated by a single space. In the following lines, each interaction is given in the form of two distinct bug numbers separated by a single space. Bugs are numbered consecutively starting from one.

输出

The output for every scenario is a line containing "Scenario #i:", where i is the number of the scenario starting at 1, followed by one line saying either "No suspicious bugs found!" if the experiment is consistent with his assumption about the bugs' sexual behavior, or "Suspicious bugs found!" if Professor Hopper's assumption is definitely wrong.

样例输入

```
2
3 3
1 2
2 3
1 3
4 2
1 2
3 4
```

样例输出

```
Scenario #1:
Suspicious bugs found!

Scenario #2:
No suspicious bugs found!
```

提示

Huge input,scanf is recommended.

来源

TUD Programming Contest 2005, Darmstadt, Germany





```python
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)


def solve_bug_life(scenarios):
    for i in range(1, scenarios + 1):
        n, m = map(int, input().split())
        uf = UnionFind(2 * n + 1)  # 为每个虫子创建两个节点表示其可能的两种性别
        suspicious = False
        for _ in range(m):
            u, v = map(int, input().split())
            if suspicious:
                continue

            if uf.is_connected(u, v):
                suspicious = True
            uf.union(u, v + n)  # 将u的一种性别与v的另一种性别关联
            uf.union(u + n, v)  # 同理


        print(f'Scenario #{i}:')
        print('Suspicious bugs found!' if suspicious else 'No suspicious bugs found!')
        print()


# 读取场景数量并解决问题
scenarios = int(input())
solve_bug_life(scenarios)
```





## 01084: 正方形破坏者

http://cs101.openjudge.cn/practice/01084/

- 左图展示了一个由 24 根火柴棍组成的 3 * 3 的网格，所有火柴棍的长度都是 1。在这张网格图中有很多的正方形：边长为 1 的有 9 个，边长为 2 的有 4 个，边长为 3 的有 1 个。

  每一根火柴棍都被编上了一个号码，编码的方式是从上到下：横着的第一行，竖着的第一行，横着的第二行一直到横着的最后一行。在同一行内部，编码的方式是从左到右。 其中对 3 * 3 的火柴网格编码的结果已经标在左图上了。

  右图展示了一个不完整的 3 * 3 的网格，它被删去了编号为 12,17,23 的火柴棍。删去这些火柴棍后被摧毁了 5 个大小为 1 的正方形，3 个大小为 2 的正方形和 1 个大小为 3 的正方形。（一个正方形被摧毁当且仅当它的边界上有至少一个火柴棍被移走了）

  

  ![img](http://media.openjudge.cn/images/g86/1084.gif)

  

  可以把上述概念推广到 n * n 的火柴棍网格。在完整的 n * n 的网格中，使用了 2n(n+1) 根火柴棍，其中边长为 i(i ∈ [1,n]) 的正方形有 (n-i+1)2个。

  现在给出一个 n * n 的火柴棍网格，最开始它被移走了 k 根火柴棍。问最少再移走多少根火柴棍，可以让所有的正方形都被摧毁。

  输入

  输入包含多组数据，第一行一个整数 T 表示数据组数。

  对于每组数据，第一行输入一个整数 n 表示网格的大小( n <= 5)。第二行输入若干个空格隔开的整数，第一个整数 k 表示被移走的火柴棍个数，接下来 k 个整数表示被移走的火柴棍编号。

  输出

  对于每组数据，输出一行一个整数表示最少删除多少根火柴棍才能摧毁所有的正方形。

  样例输入

  ```
  2
  2
  0
  3
  3 12 17 23
  ```

  样例输出

  ```
  3
  3
  ```



Deng_Leo

https://blog.csdn.net/2301_79402523/article/details/137194237

```python
import copy
import sys
sys.setrecursionlimit(1 << 30)
found = False
 
def check1(x, tmp):
    for y in graph[x]:
        if tmp[y]:
            return False
    return True
 
def check2(x):
    for y in graph[x]:
        if judge[y]:
            return False
    return True
 
def estimate():
    cnt = 0
    tmp = copy.deepcopy(judge)
    for x in range(1, total+1):
        if check1(x, tmp):
            cnt += 1
            for u in graph[x]:
                tmp[u] = True
    return cnt
 
def dfs(t):
    global found
    if t + estimate() > limit:
        return
    for x in range(1, total+1):
        if check2(x):
            for y in graph[x]:
                judge[y] = True
                dfs(t+1)
                judge[y] = False
                if found:
                    return
            return
    found = True
 
for _ in range(int(input())):
    n = int(input())
    lst = list(map(int, input().split()))
    d, m, nums, total = 2*n+1, lst[0], lst[1:], 0
    graph = {}
    for i in range(n):
        for j in range(n):
            for k in range(1, n+1):
                if i+k <= n and j+k <= n:
                    total += 1
                    graph[total] = []
                    for p in range(1, k+1):
                        graph[total] += [d*i+j+p, d*(i+p)+j-n, d*(i+p)+j-n+k, d*(i+k)+j+p]
    judge = [False for _ in range(2*n*(n+1)+1)]
    for num in nums:
        judge[num] = True
    limit = estimate()
    found = False
    while True:
        dfs(0)
        if found:
            print(limit)
            break
        limit += 1
```





# 2024北京大学智能学院优秀大学生暑期夏令营机试

http://bailian.openjudge.cn/xly2024062701/

2024-06-27 14:00~16:00



| 题目                           | tags           |
| ------------------------------ | -------------- |
| 画矩形                         | implementation |
| 花生采摘                       | Implementation |
| 棋盘问题                       | dfs            |
| 最大上升子序列和               | dp             |
| 文件结构“图”                   | tree           |
| Stockbroker                    | floyd_warshall |
| 由中根序列和后根序列重建二叉树 | tree           |
| 超级备忘录                     | splay tree     |



## 08183:画矩形

http://cs101.openjudge.cn/practice/08183/

根据参数，画出矩形。

输入

输入一行，包括四个参数：前两个参数为整数，依次代表矩形的高和宽（高不少于3行不多于10行，宽不少于5列不多于10列）；第三个参数是一个字符，表示用来画图的矩形符号；第四个参数为1或0，0代表空心，1代表实心。

输出

输出画出的图形。

样例输入

```
7 7 @ 0
```

样例输出

```
@@@@@@@
@     @
@     @
@     @
@     @
@     @
@@@@@@@
```





```python
def draw_rectangle(height, width, char, is_filled):
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height - 1 or j == 0 or j == width - 1 or is_filled:
                print(char, end="")
            else:
                print(" ", end="")
        print()

# Test the function
h,w,c,f = input().split()
draw_rectangle(int(h), int(w), c, int(f))

```



## 07902:花生采摘

http://cs101.openjudge.cn/practice/07902/

鲁宾逊先生有一只宠物猴，名叫多多。这天，他们两个正沿着乡间小路散步，突然发现路边的告示牌上贴着一张小小的纸条：“欢迎免费品尝我种的花生！——熊字”。

鲁宾逊先生和多多都很开心，因为花生正是他们的最爱。在告示牌背后，路边真的有一块花生田，花生植株整齐地排列成矩形网格（如图1）。有经验的多多一眼就能看出，每棵花生植株下的花生有多少。为了训练多多的算术，鲁宾逊先生说：“你先找出花生最多的植株，去采摘它的花生；然后再找出剩下的植株里花生最多的，去采摘它的花生；依此类推，不过你一定要在我限定的时间内回到路边。”

我们假定多多在每个单位时间内，可以做下列四件事情中的一件：

1) 从路边跳到最靠近路边（即第一行）的某棵花生植株；
2) 从一棵植株跳到前后左右与之相邻的另一棵植株；
3) 采摘一棵植株下的花生；
4) 从最靠近路边（即第一行）的某棵花生植株跳回路边。

![img](http://media.openjudge.cn/images/upload/1446616081.jpg)

现在给定一块花生田的大小和花生的分布，请问在限定时间内，多多最多可以采到多少个花生？注意可能只有部分植株下面长有花生，假设这些植株下的花生个数各不相同。

例如在图2所示的花生田里，只有位于(2, 5), (3, 7), (4, 2), (5, 4)的植株下长有花生，个数分别为13, 7, 15, 9。沿着图示的路线，多多在21个单位时间内，最多可以采到37个花生。

输入

第一行包括三个整数，M, N和K，用空格隔开；表示花生田的大小为M * N（1 <= M, N <= 20），多多采花生的限定时间为K（0 <= K <= 1000）个单位时间。接下来的M行，每行包括N个非负整数，也用空格隔开；第i + 1行的第j个整数Pij（0 <= Pij <= 500）表示花生田里植株(i, j)下花生的数目，0表示该植株下没有花生。

输出

包括一行，这一行只包含一个整数，即在限定时间内，多多最多可以采到花生的个数。

样例输入

```
样例 #1：
6 7 21
0 0 0 0 0 0 0
0 0 0 0 13 0 0
0 0 0 0 0 0 7
0 15 0 0 0 0 0
0 0 0 9 0 0 0
0 0 0 0 0 0 0

样例 #2：
6 7 20
0 0 0 0 0 0 0
0 0 0 0 13 0 0
0 0 0 0 0 0 7
0 15 0 0 0 0 0
0 0 0 9 0 0 0
0 0 0 0 0 0 0
```

样例输出

```
样例 #1：
37

样例 #2：
28
```

来源

NOIP2004复赛 普及组 第二题



```python
def max_peanuts(M, N, K, field):
    # 提取所有有花生的位置及其数量
    peanuts = []
    for i in range(M):
        for j in range(N):
            if field[i][j] > 0:
                peanuts.append((field[i][j], i, j))
    
    # 按照花生数量从大到小排序
    peanuts.sort(reverse=True, key=lambda x: x[0])
    
    # 初始化当前时间和采摘的花生总数
    current_time = 0
    total_peanuts = 0
    
    # 初始位置设为路边
    current_pos = (-1, 0)
    
    for peanut in peanuts:
        amount, x, y = peanut
        
        # 计算从当前位置到该位置的时间
        if current_pos[0] == -1:  # 从路边跳到第一行
            time_to_reach = x + 1 + abs(current_pos[1] - y)
        else:
            time_to_reach = abs(current_pos[0] - x) + abs(current_pos[1] - y)
        
        if current_pos == (-1, 0):  # 从路边跳到第一行的时间
            current_time += (x + 1)
        else:
            current_time += time_to_reach
        
        # 采摘花生需要1单位时间
        current_time += 1
        
        if current_time + x + 1 <= K:
            total_peanuts += amount
            current_pos = (x, y)
        else:
            break
    
    return total_peanuts

# 读取输入
M, N, K = map(int, input().split())
field = []
for _ in range(M):
    field.append(list(map(int, input().split())))

# 计算并输出结果
result = max_peanuts(M, N, K, field)
print(result)

```



## 01321:棋盘问题

http://cs101.openjudge.cn/practice/01321/

在一个给定形状的棋盘（形状可能是不规则的）上面摆放棋子，棋子没有区别。要求摆放时任意的两个棋子不能放在棋盘中的同一行或者同一列，请编程求解对于给定形状和大小的棋盘，摆放k个棋子的所有可行的摆放方案C。

输入

输入含有多组测试数据。
每组数据的第一行是两个正整数，n k，用一个空格隔开，表示了将在一个n*n的矩阵内描述棋盘，以及摆放棋子的数目。 n <= 8 , k <= n
当为-1 -1时表示输入结束。
随后的n行描述了棋盘的形状：每行有n个字符，其中 # 表示棋盘区域， . 表示空白区域（数据保证不出现多余的空白行或者空白列）。

输出

对于每一组数据，给出一行输出，输出摆放的方案数目C （数据保证C<2^31）。

样例输入

```
2 1
#.
.#
4 4
...#
..#.
.#..
#...
-1 -1
```

样例输出

```
2
1
```

来源

蔡错@pku





```python
# https://www.cnblogs.com/Ayanowww/p/11555193.html
'''
本题知识点：深度优先搜索 + 枚举 + 回溯

题意是要求我们把棋子放在棋盘的'#'上，但不能把两枚棋子放在同一列或者同一行上，问摆好这k枚棋子有多少种情况。

我们可以一行一行地找，当在某一行上找到一个可放入的'#'后，就开始找下一行的'#'，如果下一行没有，就再从下一行找。这样记录哪个'#'已放棋子就更简单了，只需要记录一列上就可以了。
'''
n, k, ans = 0, 0, 0
chess = [['' for _ in range(10)] for _ in range(10)]
take = [False] * 10

def dfs(h, t):
    global ans

    if t == k:
        ans += 1
        return

    if h == n:
        return

    for i in range(h, n):
        for j in range(n):
            if chess[i][j] == '#' and not take[j]:
                take[j] = True
                dfs(i + 1, t + 1)
                take[j] = False

while True:
    n, k = map(int, input().split())
    if n == -1 and k == -1:
        break

    for i in range(n):
        chess[i] = list(input())

    take = [False] * 10
    ans = 0
    dfs(0, 0)
    print(ans)
```





## 03532:最大上升子序列和

http://cs101.openjudge.cn/practice/03532/

一个数的序列bi，当b1 < b2 < ... < bS的时候，我们称这个序列是上升的。对于给定的一个序列(a1, a2, ...,aN)，我们可以得到一些上升的子序列(ai1, ai2, ..., aiK)，这里1 <= i1 < i2 < ... < iK <= N。比如，对于序列(1, 7, 3, 5, 9, 4, 8)，有它的一些上升子序列，如(1, 7), (3, 4, 8)等等。这些子序列中序列和最大为18，为子序列(1, 3, 5, 9)的和.

你的任务，就是对于给定的序列，求出最大上升子序列和。注意，最长的上升子序列的和不一定是最大的，比如序列(100, 1, 2, 3)的最大上升子序列和为100，而最长上升子序列为(1, 2, 3)

输入

输入的第一行是序列的长度N (1 <= N <= 1000)。第二行给出序列中的N个整数，这些整数的取值范围都在0到10000（可能重复）。

输出

最大上升子序列和

样例输入

```
7
1 7 3 5 9 4 8
```

样例输出

```
18
```





```python
input()
a = [int(x) for x in input().split()]

n = len(a)
dp = [0]*n

for i in range(n):
    dp[i] = a[i]
    for j in range(i):
        if a[j]<a[i]:
            dp[i] = max(dp[j]+a[i], dp[i])
    
print(max(dp))
```





## 02775:文件结构“图”

在计算机上看到文件系统的结构通常很有用。Microsoft Windows上面的"explorer"程序就是这样的一个例子。但是在有图形界面之前，没有图形化的表示方法的，那时候最好的方式是把目录和文件的结构显示成一个"图"的样子，而且使用缩排的形式来表示目录的结构。比如：



```
ROOT
|     dir1
|     file1
|     file2
|     file3
|     dir2
|     dir3
|     file1
file1
file2
```

这个图说明：ROOT目录包括三个子目录和两个文件。第一个子目录包含3个文件，第二个子目录是空的，第三个子目录包含一个文件。

输入

你的任务是写一个程序读取一些测试数据。每组测试数据表示一个计算机的文件结构。每组测试数据以'*'结尾，而所有合理的输入数据以'#'结尾。一组测试数据包括一些文件和目录的名字（虽然在输入中我们没有给出，但是我们总假设ROOT目录是最外层的目录）。在输入中,以']'表示一个目录的内容的结束。目录名字的第一个字母是'd'，文件名字的第一个字母是'f'。文件名可能有扩展名也可能没有（比如fmyfile.dat和fmyfile）。文件和目录的名字中都不包括空格,长度都不超过30。一个目录下的子目录个数和文件个数之和不超过30。

输出

在显示一个目录中内容的时候，先显示其中的子目录（如果有的话），然后再显示文件（如果有的话）。文件要求按照名字的字母表的顺序显示（目录不用按照名字的字母表顺序显示，只需要按照目录出现的先后显示）。对每一组测试数据，我们要先输出"DATA SET x:"，这里x是测试数据的编号（从1开始）。在两组测试数据之间要输出一个空行来隔开。

你需要注意的是，我们使用一个'|'和5个空格来表示出缩排的层次。

样例输入

```
file1
file2
dir3
dir2
file1
file2
]
]
file4
dir1
]
file3
*
file2
file1
*
#
```

样例输出

```
DATA SET 1:
ROOT
|     dir3
|     |     dir2
|     |     file1
|     |     file2
|     dir1
file1
file2
file3
file4

DATA SET 2:
ROOT
file1
file2
```

提示

一个目录和它的子目录处于不同的层次
一个目录和它的里面的文件处于同一层次

来源

翻译自 Pacific Northwest 1998 的试题



```python
class Node:
    def __init__(self, name):
        self.name = name
        self.dirs = []
        self.files = []

def print_structure(node, indent=0):
    prefix = '|     ' * indent
    print(prefix + node.name)
    for dir in node.dirs:
        print_structure(dir, indent + 1)
    for file in sorted(node.files):
        print(prefix + file)

dataset = 1
datas = []
temp = []
while True:
    line = input()
    if line == '#':
        break
    if line == '*':
        datas.append(temp)
        temp = []
    else:
        temp.append(line)

for data in datas:
    print(f'DATA SET {dataset}:')
    root = Node('ROOT')
    stack = [root]
    for line in data:
        if line[0] == 'd':
            dir = Node(line)
            stack[-1].dirs.append(dir)
            stack.append(dir)
        elif line[0] == 'f':
            stack[-1].files.append(line)
        elif line == ']':
            stack.pop()
    print_structure(root)
    if dataset < len(datas):
        print()
    dataset += 1
```



## 01125:Stockbroker Grapevine

http://cs101.openjudge.cn/practice/01125/

Stockbrokers are known to overreact to rumours. You have been contracted to develop a method of spreading disinformation amongst the stockbrokers to give your employer the tactical edge in the stock market. For maximum effect, you have to spread the rumours in the fastest possible way.

Unfortunately for you, stockbrokers only trust information coming from their "Trusted sources" This means you have to take into account the structure of their contacts when starting a rumour. It takes a certain amount of time for a specific stockbroker to pass the rumour on to each of his colleagues. Your task will be to write a program that tells you which stockbroker to choose as your starting point for the rumour, as well as the time it will take for the rumour to spread throughout the stockbroker community. This duration is measured as the time needed for the last person to receive the information.

输入

Your program will input data for different sets of stockbrokers. Each set starts with a line with the number of stockbrokers. Following this is a line for each stockbroker which contains the number of people who they have contact with, who these people are, and the time taken for them to pass the message to each person. The format of each stockbroker line is as follows: The line starts with the number of contacts (n), followed by n pairs of integers, one pair for each contact. Each pair lists first a number referring to the contact (e.g. a '1' means person number one in the set), followed by the time in minutes taken to pass a message to that person. There are no special punctuation symbols or spacing rules.

Each person is numbered 1 through to the number of stockbrokers. The time taken to pass the message on will be between 1 and 10 minutes (inclusive), and the number of contacts will range between 0 and one less than the number of stockbrokers. The number of stockbrokers will range from 1 to 100. The input is terminated by a set of stockbrokers containing 0 (zero) people.



输出

For each set of data, your program must output a single line containing the person who results in the fastest message transmission, and how long before the last person will receive any given message after you give it to this person, measured in integer minutes.
It is possible that your program will receive a network of connections that excludes some persons, i.e. some people may be unreachable. If your program detects such a broken network, simply output the message "disjoint". Note that the time taken to pass the message from person A to person B is not necessarily the same as the time taken to pass it from B to A, if such transmission is possible at all. 

样例输入

```
3
2 2 4 3 5
2 1 2 3 6
2 1 2 2 2
5
3 4 4 2 8 5 3
1 5 8
4 1 6 4 10 2 7 5 2
0
2 2 5 1 5
0
```

样例输出

```
3 2
3 10
```

来源

Southern African 2001





```python
# 23n2300011072(蒋子轩)
def floyd_warshall(graph):
    """
    实现Floyd-Warshall算法，找到所有顶点对之间的最短路径。
    """
    n = len(graph)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if i != j and j != k and i != k:
                    # 通过顶点i更新顶点j和k之间的最短路径
                    graph[j][k] = min(graph[j][k], graph[j][i] + graph[i][k])
    return graph

def find_best_broker(graph):
    """
    找到最佳经纪人开始传播谣言，以使其以最短时间到达所有其他人。
    """
    n = len(graph)
    mmin = float('inf')
    best_broker = -1
    for i in range(n):
        # 查找从经纪人i向所有其他人传播信息所需的最大时间
        mmax = max(graph[i])
        if mmin > mmax:
            mmin = mmax
            best_broker = i
    return best_broker, mmin


while True:
    # 读取经纪人的数量
    n = int(input())
    if n == 0:
        break

    # 用'inf'初始化图（代表无直接连接）
    graph = [[float('inf') for _ in range(n)] for _ in range(n)]
    for i in range(n):
        graph[i][i] = 0  # 经纪人将消息传递给自己的时间为0
        data = list(map(int, input().split()))
        for j in range(1, len(data), 2):
            # 用直接连接和传递消息所需的时间更新图
            graph[i][data[j] - 1] = data[j + 1]
    
    # 计算所有经纪人对之间的最短路径
    graph = floyd_warshall(graph)
    # 查找开始传播谣言的最佳经纪人
    broker, time = find_best_broker(graph)
    
    # 打印结果
    if time == float('inf'):
        print("disjoint")
    else:
        print(broker + 1, time)
```



## 05414:由中根序列和后根序列重建二叉树

http://cs101.openjudge.cn/practice/05414/

我们知道如何按照三种深度优先次序来周游一棵二叉树，来得到中根序列、前根序列和后根序列。反过来，如果给定二叉树的中根序列和后根序列，或者给定中根序列和前根序列，可以重建一二叉树。本题输入一棵二叉树的中根序列和后根序列，要求在内存中重建二叉树，最后输出这棵二叉树的前根序列。

用不同的整数来唯一标识二叉树的每一个结点，下面的二叉树

![img](http://media.openjudge.cn/images/upload/1351670567.png)

中根序列是9 5 32 67

后根序列9 32 67 5

前根序列5 9 67 32

输入

两行。第一行是二叉树的中根序列，第二行是后根序列。每个数字表示的结点之间用空格隔开。结点数字范围0～65535。暂不必考虑不合理的输入数据。

输出

一行。由输入中的中根序列和后根序列重建的二叉树的前根序列。每个数字表示的结点之间用空格隔开。

样例输入

```
9 5 32 67
9 32 67 5
```

样例输出

```
5 9 67 32
```





```python
def buildTree(inorder, postorder):
    if not inorder or not postorder:
        return []
    root = postorder[-1]
    rootIndex = inorder.index(root)
    leftInorder = inorder[:rootIndex]
    rightInorder = inorder[rootIndex+1:]
    leftPostorder = postorder[:len(leftInorder)]
    rightPostorder = postorder[len(leftInorder):-1]
    return [root] + buildTree(leftInorder, leftPostorder) + buildTree(rightInorder, rightPostorder)

inorder = list(map(int, input().split()))
postorder = list(map(int, input().split()))
preorder = buildTree(inorder, postorder)
print(' '.join(map(str, preorder)))
```





## 04090:超级备忘录

http://cs101.openjudge.cn/practice/04090/

你的朋友Jackson被邀请参加一个叫做“超级备忘录”的电视节目。在这个节目中，参与者需要玩一个记忆游戏。在一开始，主持人会告诉所有参与者一个数列，{A1, A2, ..., An}。接下来，主持人会在数列上做一些操作，操作包括以下几种：

1. ADD x y D：给子序列{Ax, ..., Ay}统一加上一个数D。例如，在{1, 2, 3, 4, 5}上进行操作"ADD 2 4 1"会得到{1, 3, 4, 5, 5}。
2. REVERSE x y：将子序列{Ax, ..., Ay}逆序排布。例如，在{1, 2, 3, 4, 5}上进行操作"REVERSE 2 4"会得到{1, 4, 3, 2, 5}。
3. REVOLVE x y T：将子序列{Ax, ..., Ay}轮换T次。例如，在{1, 2, 3, 4, 5}上进行操作"REVOLVE 2 4 2"会得到{1, 3, 4, 2, 5}。
4. INSERT x P：在Ax后面插入P。例如，在{1, 2, 3, 4, 5}上进行操作"INSERT 2 4"会得到{1, 2, 4, 3, 4, 5}。
5. DELETE x：删除Ax。在Ax后面插入P。例如，在{1, 2, 3, 4, 5}上进行操作"DELETE 2"会得到{1, 3, 4, 5}。
6. MIN x y：查询子序列{Ax, ..., Ay}中的最小值。例如，{1, 2, 3, 4, 5}上执行"MIN 2 4"的正确答案为2。

为了使得节目更加好看，每个参赛人都有机会在觉得困难时打电话请求场外观众的帮助。你的任务是看这个电视节目，然后写一个程序对于每一个询问计算出结果，这样可以使得Jackson在任何时候打电话求助你的时候，你都可以给出正确答案。

输入

第一行包含一个数n (n ≤ 100000)。
接下来n行给出了序列中的数。
接下来一行包含一个数M (M ≤ 100000)，描述操作和询问的数量。
接下来M行给出了所有的操作和询问。

输出

对于每一个"MIN"询问，输出正确答案。

样例输入

```
5
1 
2 
3 
4 
5
2
ADD 2 4 1
MIN 4 5
```

样例输出

```
5
```







# 2024数算B模拟考试1@gw calss

Updated 1424 GMT+8 Jun 5, 2024

2024 spring, Complied by Hongfei Yan



2024-05-23 19:40:00 ～ 2024-05-23 21:30:00







| 题目                       | tags              |
| -------------------------- | ----------------- |
| 最长上升子串               | implementation    |
| 快速排序填空               | sortings          |
| 检测括号嵌套               | stack,dict        |
| 移动办公                   | Dp                |
| 我想完成数算作业：沉淀     | Linked List       |
| 单词序列                   | bfs               |
| 我想成为数算高手：重生     | binary tree       |
| 我想成为数算高手：穿越     | tree, dfs         |
| 团结不用排序就是力量       | mst 最小生成树    |
| 判断是否是深度优先遍历序列 | graph, stack, set |
| 二叉搜索树的遍历           | bst 二叉搜索树    |





## 001/26588: 最长上升子串

implementation, http://dsbpython.openjudge.cn/2024moni1/001/

一个由数字‘0’到‘9’构成的字符串，求其中最长上升子串的长度。

一个子串，如果其中每个字符都不大于其右边的字符，则该子串为上升子串

**输入**

第一行是整数n, 1 < n < 100，表示有n个由数字‘0’到‘9’构成的字符串
接下来n行，每行一个字符串，字符串长度不超过100

**输出**

对每个字符串，输出其中最长上升子串的长度

样例输入

```
4
112300125239
1
111
1235111111
```

样例输出

```
5
1
3
6
```



```python
def longestIncreasingSubstring(s):
    max_length = 1
    current_length = 1

    for i in range(1, len(s)):
        if s[i] >= s[i - 1]:
            current_length += 1
        else:
            max_length = max(max_length, current_length)
            current_length = 1

    return max(max_length, current_length)

if __name__ == "__main__":
    n = int(input())
    for _ in range(n):
        string = input()
        print(longestIncreasingSubstring(string))
```



```python
#蒋子轩 23工学院
for _ in range(int(input())):
    s=list(map(int,input()))
    n=len(s);ans=temp=1
    for i in range(1,n):
        if s[i]>=s[i-1]:
            temp+=1
        else:
            ans=max(ans,temp)
            temp=1
    ans=max(ans,temp)
    print(ans)
```



## 002/26589: 快速排序填空

sortings, http://dsbpython.openjudge.cn/2024moni1/002/

输入n ( 1 <= n <= 100000) 个整数，每个整数绝对值不超过100000，将这些整数用快速排序算法排序后输出

请填空

```python
sort = None
def quickSort(a,s,e): 
    if s >= e:
        return
    i,j = s,e
    while i != j:
        while i < j and a[i] <= a[j]:
            j -= 1
        a[i],a[j] = a[j],a[i]
// 在此处补充你的代码
#-----
    quickSort(a,s,i-1)
    quickSort(a, i+1,e)
n = int(input())
a = []
for i in range(n):
    a.append(int(input()))
quickSort(a,0,len(a)-1)
for i in a:
    print(i)
```

输入

第一行是整数n
接下来n行，每行1个整数

输出

n行，输入中n个整数排序后的结果

样例输入

```
3
6
12
3
```

样例输出

```
3
6
12
```


```python
sort = None
def quickSort(a,s,e): 
    if s >= e:
        return
    i,j = s,e
    while i != j:
        while i < j and a[i] <= a[j]:
            j -= 1
        a[i],a[j] = a[j],a[i]
        while i < j and a[i] <= a[j]:
            i += 1
        a[i],a[j] = a[j],a[i]
#-----
    quickSort(a,s,i-1)
    quickSort(a, i+1,e)
n = int(input())
a = []
for i in range(n):
    a.append(int(input()))
quickSort(a,0,len(a)-1)
for i in a:
    print(i)
```





## 003/26590: 检测括号嵌套

stack, dict, http://dsbpython.openjudge.cn/2024moni1/003/

字符串中可能有3种成对的括号，"( )"、"[ ]"、"{}"。请判断字符串的括号是否都正确配对以及有无括号嵌套。无括号也算正确配对。括号交叉算不正确配对，例如"1234[78)ab]"就不算正确配对。一对括号被包含在另一对括号里面，例如"12(ab[8])"就算括号嵌套。括号嵌套不影响配对的正确性。 给定一个字符串: 如果括号没有正确配对，则输出 "ERROR" 如果正确配对了，且有括号嵌套现象，则输出"YES" 如果正确配对了，但是没有括号嵌套现象，则输出"NO"   

输入

一个字符串，长度不超过5000,仅由 ( ) [ ] { } 和小写英文字母以及数字构成

输出

根据实际情况输出 ERROR, YES 或NO

样例输入

```
样例1:
[](){}
样例2:
[(a)]bv[]
样例3:
[[(])]{}
```

样例输出

```
样例1:
NO
样例2:
YES
样例3:
ERROR
```



```python
def check_brackets(s):
    stack = []
    nested = False
    pairs = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in pairs.values():
            stack.append(ch)
        elif ch in pairs.keys():
            if not stack or stack.pop() != pairs[ch]:
                return "ERROR"
            if stack:
                nested = True
    if stack:
        return "ERROR"
    return "YES" if nested else "NO"

s = input()
print(check_brackets(s))
```





```python
#蒋子轩 23工学院
a=[];s='([{'
dic={')':'(',']':'[','}':'{'}
flag=0
for i in input():
    if i in s:
        a.append(i)
    elif i in dic:
        if a[-1]!=dic[i]:
            print('ERROR');exit()
        a.pop()
        if a: flag=1
print('YES' if flag else 'NO')
```





## 004/19164: 移动办公

dp, http://dsbpython.openjudge.cn/2024moni1/004/

假设你经营着一家公司，公司在北京和南京各有一个办公地点。公司只有你一个人，所以你只能每月选择在一个城市办公。在第i个月，如果你在北京办公，你能获得Pi的营业额，如果你在南京办公，你能获得Ni的营业额。但是，如果你某个月在一个城市办公，下个月在另一个城市办公，你需要支付M的交通费。那么，该怎样规划你的行程（可在任何一个城市开始），才能使得总收入（总营业额减去总交通费）最大？

输入

输入的第一行有两个整数T（1 <= T <= 100）和M（1 <= M <= 100），T代表总共的月数，M代表交通费。接下来的T行每行包括两个在1到100之间（包括1和100）的整数，分别表示某个月在北京和在南京办公获得的营业额。

输出

输出只包括一行，这一行只包含一个整数，表示可以获得的最大总收入。

样例输入

```
4 3
10 9
2 8
9 5
8 2
```

样例输出

```
31
```





```python
n,m=map(int,input().split())
dp1=[0]*(n+1)
dp2=[0]*(n+1)
for i in range(1,n+1):
    a,b=map(int,input().split())
    dp1[i]=max(dp1[i-1],dp2[i-1]-m)+a
    dp2[i]=max(dp1[i-1]-m,dp2[i-1])+b
print(max(dp1[n],dp2[n]))
```





## 005/26570: 我想完成数算作业：沉淀

Linked list, http://dsbpython.openjudge.cn/2024moni1/005/

小A选了数算课之后从来没去过，有一天他和小B聊起来才知道原来每周都有上机作业！于是小A连忙问了小B题目序号，保险起见他又问了小C作业的题号，不幸的是两人的回答并不一样...

无奈之下小A决定把两个人记的题目都做完，为此他希望先合并两人的题号，并去除其中的重复题目，请你帮他完成下面的程序填空。

```python
class Node:
    def __init__(self, data, next=None):
        self.data, self.next = data, next
class LinkedList:
    def __init__(self):
        self.head = None
    def create(self, data):
        self.head = Node(0)
        cur = self.head
        for i in range(len(data)):
            node = Node(data[i])
            cur.next = node
            cur = cur.next
def printList(head):
    cur = head.next
    while cur:
        print(cur.data, end=' ')
        cur = cur.next

def mergeTwoLists( l1, l2):
    head = cur = Node(0)
    while l1 and l2:
        if l1.data > l2.data:
            cur.next = l2
            l2 = l2.next
        else:
            cur.next = l1
            l1 = l1.next
        cur = cur.next
    cur.next = l1 or l2
    return head
def deleteDuplicates(head):
// 在此处补充你的代码
data1 = list(map(int, input().split()))
data2 = list(map(int, input().split()))
list1 = LinkedList()
list2 = LinkedList()
list1.create(data1)
list2.create(data2)
head = mergeTwoLists(list1.head.next, list2.head.next)
deleteDuplicates(head)
printList(head)
```

输入

输入包括两行，分别代表小B和小C记下的排好序的作业题号

输出

按题号顺序输出一行小A应该完成的题目序号。

样例

```
sample1 input：
1 3
1 2

sample1 output：
1 2 3
```

样例

```
sample2 input：
1 2 4
1 3 4

sample2 output：
1 2 3 4
```

提示

程序中处理的链表是带空闲头结点的



```python
class Node:
    def __init__(self, data, next=None):
        self.data, self.next = data, next
class LinkedList:
    def __init__(self):
        self.head = None
    def create(self, data):
        self.head = Node(0)
        cur = self.head
        for i in range(len(data)):
            node = Node(data[i])
            cur.next = node
            cur = cur.next
def printList(head):
    cur = head.next
    while cur:
        print(cur.data, end=' ')
        cur = cur.next

def mergeTwoLists( l1, l2):
    head = cur = Node(0)
    while l1 and l2:
        if l1.data > l2.data:
            cur.next = l2
            l2 = l2.next
        else:
            cur.next = l1
            l1 = l1.next
        cur = cur.next
    cur.next = l1 or l2
    return head
def deleteDuplicates(head):
    cur = head
    while cur and cur.next:
        if cur.data == cur.next.data:
            cur.next = cur.next.next
        else:
            cur = cur.next
data1 = list(map(int, input().split()))
data2 = list(map(int, input().split()))
list1 = LinkedList()
list2 = LinkedList()
list1.create(data1)
list2.create(data2)
head = mergeTwoLists(list1.head.next, list2.head.next)
deleteDuplicates(head)
printList(head)
```





## 006/26571: 我想完成数算作业：代码

disjoint set, http://dsbpython.openjudge.cn/2024moni1/006/

当卷王小D睡前意识到室友们每天熬夜吐槽的是自己也选了的课时，他距离早八随堂交的ddl只剩下了不到4小时。已经debug一晚上无果的小D有心要分无力做题，于是决定直接抄一份室友的作业完事。万万没想到，他们作业里完全一致的错误，引发了一场全面的作业查重……

假设a和b作业雷同，b和c作业雷同，则a和c作业雷同。所有抄袭现象都会被发现，且雷同的作业只有一份独立完成的原版，请输出独立完成作业的人数

输入

第一行输入两个正整数表示班上的人数n与总比对数m，接下来m行每行均为两个1-n中的整数i和j，表明第i个同学与第j个同学的作业雷同。

输出

独立完成作业的人数

样例输入

```
样例1：
3 2
1 2
1 3
样例2：
4 2
2 4
1 3
```

样例输出

```
样例1：
1
样例2:
2
```



```python
def find(parent, i):
    if parent[i] != i:
        parent[i] = find(parent, parent[i])
    return parent[i]

def union(parent, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if xroot != yroot:
        parent[xroot] = yroot

n, m = map(int, input().split())
parent = list(range(n + 1))
for _ in range(m):
    i, j = map(int, input().split())
    union(parent, i, j)

count = sum(i == parent[i] for i in range(1, n + 1))
print(count)
```





```python
#蒋子轩 23工学院
class UnionFind:
    def __init__(self,n):
        self.p=list(range(n))
        self.h=[0]*n
    def find(self,x):
        if self.p[x]!=x:
            self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self,x,y):
        rootx=self.find(x)
        rooty=self.find(y)
        if rootx!=rooty:
            self.p[rootx]=rooty
n,m=map(int,input().split())
uf=UnionFind(n)
for _ in range(m):
    x,y=map(int,input().split())
    uf.union(x-1,y-1)
print(len(set([uf.find(i) for i in range(n)])))
```



## 007/04128: 单词序列

bfs, http://cs101.openjudge.cn/practice/04128/

给出两个单词（开始单词和结束单词）以及一个词典。找出从开始单词转换到结束单词，所需要的最短转换序列。转换的规则如下：

1、每次只能改变一个字母

2、转换过程中出现的单词(除开始单词和结束单词)必须存在于词典中

例如：

开始单词为：hit

结束单词为：cog

词典为：[hot,dot,dog,lot,log,mot]

那么一种可能的最短变换是： hit -> hot -> dot -> dog -> cog,

所以返回的结果是序列的长度5；

注意：

1、如果不能找到这种变换，则输出0；

2、词典中所有单词长度一样；

3、所有的单词都由小写字母构成；

4、开始单词和结束单词可以不在词典中。

**输入**

共两行，第一行为开始单词和结束单词（两个单词不同），以空格分开。第二行为若干的单词（各不相同），以空格分隔开来，表示词典。单词长度不超过5,单词个数不超过30。

**输出**

输出转换序列的长度。

样例输入

```
hit cog
hot dot dog lot log
```

样例输出

```
5
```







```python
from collections import deque

def is_valid_transition(word1, word2):
    """
    Check if two words differ by exactly one character.
    """
    diff_count = sum(c1 != c2 for c1, c2 in zip(word1, word2))
    return diff_count == 1

def shortest_word_sequence(start, end, dictionary):
    if start == end:
        return 0
    
    if len(start) != len(end):
        return 0
    
    if start not in dictionary:
        dictionary.append(start)
    
    if end not in dictionary:
        dictionary.append(end)

    queue = deque([(start, 1)])
    visited = set([start])

    while queue:
        current_word, steps = queue.popleft()
        if current_word == end:
            return steps
        
        for word in dictionary:
            if is_valid_transition(current_word, word) and word not in visited:
                queue.append((word, steps + 1))
                visited.add(word)
    
    return 0

# Example usage:
start, end = input().split()
dictionary = input().split()

result = shortest_word_sequence(start, end, dictionary)
print(result)
```





```python
#蒋子轩 23工学院
from collections import deque
def check(a,b):
    i=0
    while a[i]==b[i]:
        i+=1
    if a[i+1:]==b[i+1:]:
        return True
    return False
def bfs(x,step):
    q=deque([(x,step)])
    while q:
        x,step=q.popleft()
        if check(x,y):
            return step+1
        for i in range(len(dic)):
            if i not in vis and check(x,dic[i]):
                q.append((dic[i],step+1))
                vis.add(i)
    return 0
x,y=input().split()
dic=list(input().split())
vis=set()
print(bfs(x,1))
```



## 008/26582: 我想成为数算高手：重生

binary tree, http://dsbpython.openjudge.cn/2024moni1/008/

经历了查重风波的小D被取消了当次作业成绩，但他可是我们这次故事的主角。学习难道只是为了这点作业和分数吗？！小D振作了起来，我不能成为作业的奴隶，我要成为数算高手！

成为数算高手，第一步是要会建QTE二叉树。

我们可以把由 0 和 1 组成的字符串分为三类：全 0 串称为 Q串，全 1 串称为 T 串，既含 0 又含 1 的串则称为 E 串。

QTE二叉树是一种二叉树，它的结点类型包括 Q 结点，T 结点和 E 结点三种。由一个长度为 n 的 01 串 S 可以构造出一棵 QTE 二叉树 T，递归的构造方法如下：

设T 的根结点为root，则root类型与串 S 的类型相同；
若串 S的长度大于1，将串 S 从中间分开，分为左右子串 S1 和 S2，若串S长度为奇数，则让左子串S1的长度比右子串恰好长1，否则让左右子串长度相等；由左子串 S1 构造root的左子树 T1，由右子串 S2 构造root 的右子树 T2。

 

输入

第一行是一个整数 T, T <= 50,表示01串的数目。
接下来T行，每行是一个01串，对应于一棵QTE二叉树。01串长度至少为1，最多为2048。

输出

输出T行，每行是一棵QTE二叉树的后序遍历序列

样例输入

```
3
111
11010
10001011
```

样例输出

```
TTTTT
TTTQETQEE
TQEQQQETQETTTEE
```






```python
#2300011335	邓锦文
class TreeNode:
    def __init__(self, val, letter):
        self.val = val
        self.letter = letter
        self.left = None
        self.right = None

def build(node):
    if len(node.val) > 1:
        mid = (len(node.val)+1) // 2
        l = node.val[:mid]
        if '0' not in l:
            lc = 'T'
        elif '1' not in l:
            lc = 'Q'
        else:
            lc = 'E'
        node.left = TreeNode(l, lc)
        r = node.val[mid:]
        if '0' not in r:
            rc = 'T'
        elif '1' not in r:
            rc = 'Q'
        else:
            rc = 'E'
        node.right = TreeNode(r, rc)
        build(node.left)
        build(node.right)

def postfix(root):
    if root is None:
        return ''
    return postfix(root.left)+postfix(root.right)+root.letter

for _ in range(int(input())):
    s = input().strip()
    if '0' not in s:
        c = 'T'
    elif '1' not in s:
        c = 'Q'
    else:
        c = 'E'
    root = TreeNode(s, c)
    build(root)
    print(postfix(root))
```





```python
#蒋子轩 23工学院
class Node:
    def __init__(self,val):
        self.val=val
        self.left=None
        self.right=None
def check(x):
    if '0' not in x:return 'T'
    if '1' not in x:return 'Q'
    return 'E'
def build(s):
    root=Node(check(s))
    if len(s)>1:
        n=(len(s)+1)//2
        root.left=build(s[:n])
        root.right=build(s[n:])
    return root
def post(root):
    if not root:return ''
    return post(root.left)+post(root.right)+root.val
for _ in range(int(input())):
    s=input()
    root=build(s)
    print(post(root))
```



## 009/26585: 我想成为数算高手：穿越

tree, dfs, http://dsbpython.openjudge.cn/2024moni1/009/

小D拿着地图走进理一楼，发现地图上记录的还不足这大迷宫的冰山一角……

经过考察，理一楼应当用一个树形结构描述，每一个结点上都是一个办公室或者一个机房，当然也可以是一片空地。在确定结点1 为根结点之后，对于每一个非叶子的结点 i，设以它为根的子树中所有的办公室数量为 office_i，机房数目为 computerRoom_i，都有 office_i=computerRoom_i。

小D心想，这偌大的理一里到底最多能容纳多少个机房呀。

 

输入

输入的第一行是一个正整数n, 1 <= n <= 100000，表示用树形结构描述的理一的结点，结点编号为1 到n。
第 2 至 n 行，每行两个正整数 u，v，表示结点 u 与结点 v 间有一条边。

输出

只有一行，即理一最多能容纳多少个机房。

样例输入

```
5
1 2
2 3
3 4
2 5
```

样例输出

```
2
```



思路：

- **构建树**：将输入的节点和边转化为树结构，表示理一楼的布局。
- **DFS遍历树**：通过DFS遍历计算每个子树的节点总数，确保办公室和机房数量相等。
- **奇偶性判断**：对于非根节点的子树，如果节点总数为奇数，则减1以保证为偶数，从而能够平分为办公室和机房。
- **计算最大机房数**：根节点特殊处理，返回整个树节点总数的一半作为最终结果。

通过以上步骤，能够有效地计算出理一楼最多能容纳的机房数。

```python
#2300011335	邓锦文
class Node:
    def __init__(self, val):
        self.val = val
        self.children = []


def build_tree(n, edges):
    """
    构建树结构
    :param n: 节点数量
    :param edges: 边列表
    :return: 树的根节点
    """
    nodes = [Node(i) for i in range(n + 1)]
    for a, b in edges:
        nodes[a].children.append(nodes[b])
    return nodes[1]  # 返回根节点


def calculate_max_computer_rooms(root):
    """
    计算理一楼最多能容纳的机房数
    :param root: 树的根节点
    :return: 最大机房数
    """

    def dfs(node):
        if not node.children:
            return 1  # 叶子节点

        total_nodes = 1  # 包含当前节点

        for child in node.children:
            total_nodes += dfs(child)

        if node.val == 1:
            # 根节点特殊处理，返回整个树的节点数的一半
            return total_nodes // 2
        else:
            # 非根节点，判断子树的节点数是奇数还是偶数
            return total_nodes - 1 if total_nodes % 2 else total_nodes

    return dfs(root)


n = int(input())
edges = [tuple(map(int, input().split())) for _ in range(n-1)]

# 构建树并计算最大机房数
root = build_tree(n, edges)
print(calculate_max_computer_rooms(root))

```





```python
#蒋子轩 23工学院
n=int(input())
graph=[[] for _ in range(n+1)]
children=[[] for _ in range(n+1)]
for _ in range(n-1):
    u,v=map(int,input().split())
    graph[u].append(v)
    graph[v].append(u)
cnt=[1]*(n+1);vis=[0]*(n+1)
def dfs(u):
    vis[u]=1
    for v in graph[u]:
        if not vis[v]:
            cnt[u]+=dfs(v)
            children[u].append(v)
    return cnt[u]
dfs(1);num=0
for i in range(1,n+1):
    temp=0
    if cnt[i]==1:continue
    for j in children[i]:
        if cnt[j]==1:temp+=1
    if temp%2==0:num+=1
print((n-num)//2)
```



## 010/25612: 团结不用排序就是力量

mst, http://dsbpython.openjudge.cn/2024moni1/010/

有n个人，本来互相都不认识。现在想要让他们团结起来。假设如果a认识b，b认识c，那么最终a总会通过b而认识c。如果最终所有人都互相认识，那么大家就算团结起来了。但是想安排两个人直接见面认识，需要花费一些社交成本。不同的两个人要见面，花费的成本还不一样。一个人可以见多个人，但是，有的两个人就是不肯见面。问要让大家团结起来，最少要花多少钱。请注意，认识是相互的，即若a认识b，则b也认识a。

输入

第一行是整数n和m，表示有n个人，以及m对可以安排见面的人(0 < n <100, 0 < m < 5000) 。n个人编号从0到n-1。

接下来的m行,每行有两个整数s,e和一个浮点数w, 表示花w元钱(0 <= w < 100000)可以安排s和e见面。数据保证每一对见面的花费都不一样。

输出

第一行输出让大家团结起来的最小花销，保留小数点后面2位。接下来输出每一对见面的人。输出一对人的时候，编号小在前面。
如果没法让大家团结起来，则输出"NOT CONNECTED"。

样例输入

```
5 9
0 1 10.0
0 3 7.0
0 4 25.0
1 2 8.0
1 3 9.0
1 4 35.0
2 3 11.0
2 4 50.0
3 4 24.0
```

样例输出

```
48.00
0 3
1 2
1 3
3 4
```

来源

Guo Wei



```python
#2300011335	邓锦文
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0 for _ in range(n)]

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return False
        if self.rank[x] < self.rank[y]:
            self.parent[x] = y
        elif self.rank[x] > self.rank[y]:
            self.parent[y] = x
        else:
            self.parent[y] = x
            self.rank[x] += 1
        return True

def operate(n, nodes):
    nodes.sort(key=lambda x:x[2])
    f = UnionFind(n)
    cost = 0
    result = []
    for s, e, w in nodes:
        if f.union(s, e):
            cost += w
            result.append([s, e])
    if len(result) == n-1:
        return cost, result
    return 'NOT CONNECTED'

n, m = map(int, input().split())
nodes = []
for _ in range(m):
    s, e, w = input().split()
    nodes.append([int(s), int(e), float(w)])
ans = operate(n, nodes)
if isinstance(ans, tuple):
    cost, result = ans
    print(f'{cost:.2f}')
    for row in result:
        print(*row)
else:
    print(ans)
```





```python
#蒋子轩 23工学院
from heapq import heappop,heappush
n,m=map(int,input().split())
graph=[[-1]*n for _ in range(n)]
for _ in range(m):
    u,v,w=map(float,input().split())
    u,v=int(u),int(v)
    graph[u][v]=graph[v][u]=w
vis=[0]*n;q=[(0,0,-1)]
ans=0;pairs=[]
while q:
    w,u,pre=heappop(q)
    if vis[u]:continue
    ans+=w;vis[u]=1
    for v in range(n):
        if not vis[v] and graph[u][v]!=-1:
            heappush(q,(graph[u][v],v,u))
    if u>pre:u,pre=pre,u
    if u!=-1:pairs.append((u,pre))
if len(pairs)==n-1:
    print(f'{ans:.2f}')
    for i in pairs:print(*i)
else:print("NOT CONNECTED")
```





## 011/25156: 判断是否是深度优先遍历序列

http://dsbpython.openjudge.cn/2024moni1/011/

给定一个无向图和若干个序列，判断该序列是否是图的深度优先遍历序列。

**输入**

第一行是整数n( 0 < n <= 16)和m,表示无向图的点数和边数。顶点编号从0到n-1
接下来有m行，每行两个整数a,b，表示顶点a和b之间有边
接下来是整数k(k<50)，表示下面有k个待判断序列
再接下来有k行，每行是一个待判断序列,每个序列都包含n个整数,整数范围[0,n-1]

**输出**

对每个待判断序列，如果是该无向图的深度优先遍历序列，则输出YES，否则输出NO

样例输入

```
9 9
0 1
0 2
3 0
2 1
1 5
1 4
4 5
6 3
8 7
3
0 1 2 4 5 3 6 8 7
0 1 5 4 2 3 6 8 7
0 1 5 4 3 6 2 8 7
```

样例输出

```
YES
YES
NO
```

来源

Guo Wei



判断给定的序列是否为某个无向图的深度优先遍历(DFS)序列。下面我们来详细解读一下这个程序:

1. `def check(s)`:
   - 这个函数接受一个序列 `s` 作为输入,并返回 `True` 或 `False` 来表示这个序列是否为 DFS 序列。
   - 函数内部使用了一个栈 `stack` 来存储当前到达的路径,栈顶元素表示当前已到达的点。
   - 同时使用了一个布尔数组 `vis` 来记录每个点是否已访问过。

2. 主循环:
   - 在每次循环中,程序都会判断当前序列中的一个点"能不能走"。
   - 如果栈为空,说明需要开始一个新的连通分量,此时任何点都是可以走的,所以直接将当前点压入栈,并将其标记为已访问。
   - 如果栈非空,则可以访问的点只有当前点(栈顶)的未访问过的邻居。

3. DFS 策略:
   - 如果当前点还有未访问过的邻居,并且序列中的下一个点是这些邻居之一,那么就可以将这个点压入栈,并标记为已访问。
   - 如果当前点没有未访问过的邻居,那么就需要回溯到路径上的前一个点。
   - 如果当前序列中的下一个点不符合 DFS 要求(即不是当前点的未访问过的邻居),那么这个序列就不合法,直接返回 `False`。

4. 主程序:
   - 首先输入图的信息,包括点数 `n`、边数 `m` 以及每条边的两个端点。
   - 使用邻接表 `graph` 来表示这个无向图。
   - 然后输入 `k` 个待判断的序列,对每个序列都调用 `check` 函数进行判断,并输出结果。

```python
# 22-物院-罗熙佑
def check(s):
    stack = []  # 存当前到达的路径，栈顶是当前已到达的点
    vis = [False] * n

    while s:  # 每次判断一个点"能不能走"
        if not stack:  # 栈空代表需要开始一个新的连通分量，此时任何点都是能走的
            stack.append(s[-1])
            vis[s.pop()] = True
        available = [v for v in graph[stack[-1]] if not vis[v]]  # 栈非空时，能走的点只有"当前点(栈顶)没访问过的的邻居"
        if available:
            if s[-1] in available:  # DFS 要求:当前点还有没访问过的邻居时，走的下一个点必须是这些点之一
                stack.append(s[-1])
                vis[s.pop()] = True
            else:  # 不符合要求则序列不合法
                return False
        else:  # 当前点没有未访问过的邻居: 回溯到路径上前一个点
            stack.pop()
    return True


n, m = map(int, input().split())
graph = [set() for _ in range(n)]
for _ in range(m):
    a, b = map(int, input().split())
    graph[a].add(b)
    graph[b].add(a)

k = int(input())
for _ in range(k):
    print('YES' if check([int(x) for x in input().split()][::-1]) else 'NO')
```





```python
#2300011335	邓锦文
n, m = map(int, input().split())
graph = {i: [] for i in range(n)}
for _ in range(m):
    a, b = map(int, input().split())
    graph[a].append(b)
    graph[b].append(a)
vis = [False for _ in range(n)]
stack = []
def check(nodes):
    node = nodes.pop(0)
    if not nodes:
        return True
    vis[node] = True
    stack.append(node)
    if nodes[0] in graph[node]:
        if not vis[nodes[0]]:
            if check(nodes):
                return True
        return False
    if not all(vis[i] for i in graph[node]):
        return False
    stack.pop()
    while stack:
        if nodes[0] in graph[stack[-1]]:
            if not vis[nodes[0]]:
                if check(nodes):
                    return True
        if not all(vis[i] for i in graph[stack[-1]]):
            return False
        stack.pop()
    if check(nodes):
        return True
    return False
for _ in range(int(input())):
    vis = [False for _ in range(n)]
    stack = []
    nodes = list(map(int, input().split()))
    print('YES' if check(nodes) else 'NO')
```





## 012/22275: 二叉搜索树的遍历

bst, http://dsbpython.openjudge.cn/2024moni1/012/

给出一棵二叉搜索树的前序遍历，求它的后序遍历

输入

第一行一个正整数n（n<=2000）表示这棵二叉搜索树的结点个数
第二行n个正整数，表示这棵二叉搜索树的前序遍历
保证第二行的n个正整数中，1~n的每个值刚好出现一次

输出

一行n个正整数，表示这棵二叉搜索树的后序遍历

样例输入

```
5
4 2 1 3 5
```

样例输出

```
1 3 2 5 4
```

提示

树的形状为
   4  
  / \ 
  2  5 
 / \  
 1  3  



```python
def postorder(preorder):
    if not preorder:
        return []
    root = preorder[0]
    i = 1
    while i < len(preorder):
        if preorder[i] > root:
            break
        i += 1
    left = postorder(preorder[1:i])
    right = postorder(preorder[i:])
    return left + right + [root]

n = int(input())
preorder = list(map(int, input().split()))
postorder = postorder(preorder)
print(' '.join(map(str, postorder)))
```



```python
#蒋子轩 23工学院
class Node:
    def __init__(self,val):
        self.val=val
        self.left=None
        self.right=None
def insert(root,num):
    if not root:return Node(num)
    if num<root.val:
        root.left=insert(root.left,num)
    else:
        root.right=insert(root.right,num)
    return root
def post(root):
    if not root:return []
    return post(root.left)+post(root.right)+[root.val]
n=int(input())
a=list(map(int,input().split()))
root=Node(a[0])
for i in a[1:]:
    insert(root,i)
print(*post(root))
```



# 2024春-数据结构与算法B-2班



2024-05-31 17:50 ~ 20:20

http://sydsb.openjudge.cn/2024exam/

http://sydsb.openjudge.cn/2024jkkf/



8个题目，考了7个计概的。只有22460是数据结构题目。



## 28332: 收集金币达成成就

http://cs101.openjudge.cn/practice/28332/

小明和他的伙伴们正在玩一个游戏。游戏中有26种不同的金币和成就，金币用小写字母'a'到'z'表示，成就用大写字母'A'到'Z'表示。每种成就需要收集指定数量的金币才能达成：成就'A'需要26个'a'金币，成就'B'需要25个'b'金币，依此类推，成就'Z'需要1个'z'金币。玩家每分钟可以收集一枚金币。每个玩家的金币收集和成就独立于其他玩家。

游戏结束后，你拿到了每个玩家的金币收集记录——由**小写字母和空格**组成的字符串，表示依次收集的金币，空格表示该分钟内没有收集到金币。如果玩家收集的某种金币总数达到了达成对应成就所需的数量，他就获得该成就。获得成就后，玩家可能还会继续收集这种金币。

现在给出小明和伙伴们的金币收集记录，计算每个玩家在游戏期间达成了多少种成就，并输出依次达成的成就。

输入

若干行字符串表示若干个玩家，每一行包含一个长度为n的字符串 (1≤n≤1000)，字符串仅由小写字母和空格组成，表示一个玩家的金币收集记录。

输出

对于每个玩家输出一行结果，首先包含一个整数，表示该玩家达成的成就数量；如果成就数量不为0，再空一格，输出一个仅由大写字母组成的字符串，表示玩家依次达成的成就，否则仅输出成就数量0。

样例输入

```
xyy xxz
zz z
swsw sweet ttuu sswwwtt ttt
a bccba
```

样例输出

```
3 YXZ
1 Z
2 WT
0
```



```python
def achievements(records):
    results = []
    for record in records:
        coins = [0] * 26
        achieved = []
        for coin in record.replace(' ', ''):
            coins[ord(coin) - ord('a')] += 1
            if coins[ord(coin) - ord('a')] == 26 - (ord(coin) - ord('a')):
                achieved.append(chr(ord('A') + (ord(coin) - ord('a'))))
        if achieved:
            results.append(f"{len(achieved)} {''.join(achieved)}")
        else:
            results.append("0")
    return results

# 读取输入
records = []
while True:
    try:
        records.append(input().strip())
    except EOFError:
        break

# 计算并输出结果
results = achievements(records)
for result in results:
    print(result)
```





## 28231: 卡牌游戏

http://sydsb.openjudge.cn/2024jkkf/2//

小K同学最近迷上了一款名为《束狻十三：惧与罚》的卡牌游戏，在该游戏中角色受到攻击时可以选取背包中的**连续一列卡牌**进行防守。防守规则如下：选取的每张卡牌都有其对应的正整数能力值，选取一列卡牌后生成护盾至多能抵挡的伤害值为其所有卡牌能力值之积。现在给出其背包中N（N<=1000000）个卡牌分别的能力值，请计算共有多少种选取方式使得生成的护盾能够抵挡伤害为K（K在int表示范围内）的攻击。（这个数字可能很大，需要定义long long 类型，请输出其关于233333的余数，即ans%233333）

输入

输入共两行，第一行为两个正整数N,K.
第二行共N个正整数，表示背包中每个位置卡牌对应的能力值。

输出

一个数字，表示符合要求的卡牌选取方式数ans关于233333的余数。

样例输入

```
5 8
1 2 5 3 4
```

样例输出

```
9
```

提示

题目中K在int表示范围内，N<=1000000.(1后6个0）.如果定义数组元素>1000000的数组，建议定义为全局变量，比如int card[1000005];

样例中，共有如下9个连续子列符合要求：
(1,2,5) (1,2,5,3) (1,2,5,3,4)
(2,5) (2,5,3) (2,5,3,4)
(5,3) (5,3,4)
(3,4)

中间运算结果可能比较大，需要定义long long 类型，一个long long 变量占8个字节,比如 long long ans;
long long 类型的输出方式为 printf("%lld",ans);



Memory Limit Exceeded

```python
MOD = 233333


def count_valid_subarrays(N, K, prefix):

    for i in range(N):
        prefix[i+1] *= prefix[i]

    ans = 0
    for i in range(N):
        for j in range(i + 1, N+1):
            product = prefix[j] // prefix[i]
            if product >= K:
                ans = (ans + N - j + 1) % MOD
                break

    return ans

N, K = map(int, input().split())
cards = [1] + list(map(int, input().split()))

result = count_valid_subarrays(N, K, cards)
print(result)
```





## 28332: 小明的加密算法

http://cs101.openjudge.cn/practice/28322/

小明设计了一个加密算法，用于加密一个字母字符串 s。

假设 s 中只有小写字母，且a-z的值分别对应正整数1-26。

小明想利用栈这个结构来加密字符串 s：

1. 从左到右遍历字符串 s，对于每个字符 c，先将其压入栈中。
2. 如果此时栈顶元素的值为偶数，则将所有元素出栈。如果此时栈顶元素的值为奇数，则继续遍历下一个字符。
3. 如果所有字符都遍历完毕，但栈中仍有元素，将0压入栈中。此时，栈顶元素为偶数0，将所有元素出栈。

最终，栈中的元素按出栈顺序组成一个新的字符串 t。小明将字符串 t 作为加密后的字符串。

现在你需要完成该算法的加解密算法。加密算法和解密算法的定义如下：

加密算法：给定一个字符串 s，返回按照上述算法加密后的字符串 t。

解密算法：给定一个字符串 t，返回解密后的字符串 s，即，s可以通过上述算法加密得到 t。

输入

每个测试包含多个测试用例。第一行包含测试用例的数量 t（1 ≤ t ≤ 100）。每个测试用例的描述如下。

每个测试用例的第一行为"encrypt"或"decrypt"，表示加密或解密操作。

第二行包含一个待加密或解密的字符串 s 或 t，长度不超过 100。

输出

对于每个测试用例，输出一个字符串作为答案，每个答案占一行。

样例输入

```
2
encrypt
abcde
decrypt
badc
```

样例输出

```
badc0e
abcd
```



```python
def encrypt(s):
    stack = []
    result = []

    for c in s:
        stack.append(c)
        if ord(c) % 2 == 0:
            while stack:
                result.append(stack.pop())

    if stack:
        stack.append('0')
        while stack:
            result.append(stack.pop())

    return ''.join(result)

def decrypt(t):
    result = []
    temp = []

    for c in t[::-1]:
        if c == '0':
            result += temp
            temp.clear()
        elif ord(c)  % 2 == 0:
            result = temp + [c] + result
            temp.clear()
        else:
            temp.append(c)

    return ''.join(result)


n = int(input())

results = []

for _ in range(n):
    operation = input()
    string = input()

    if operation == "encrypt":
        results.append(encrypt(string))
    elif operation == "decrypt":
        results.append(decrypt(string))

for result in results:
    print(result)

```



## 28307: 老鼠和奶酪

http://cs101.openjudge.cn/practice/28307/

有两只老鼠和 n 块不同类型的奶酪，每块奶酪都只能被其中一只老鼠吃掉。
下标为 i 处的奶酪被吃掉的得分为：
如果被第一只老鼠吃掉，则得分为reward1[i] 。 
如果被第二只老鼠吃掉，则得分为reward2[i]。 
给你一个正整数数组 reward1 ，一个正整数数组 reward2，和一个非负整数 k 。
请你返回全部奶酪被吃完，但第一只老鼠恰好吃掉 k 块奶酪的情况下，总的最大得分为多少。

输入

第一行是一个整数n，代表奶酪个数。1<=n<=10000。
第二行是n个整数，代表上述reward1，分别对应n个奶酪被第一只老鼠吃掉的得分。行内使用空格分割。
第三行是n个整数，代表上述reward2，分别对应n个奶酪被第二只老鼠吃掉的得分。行内使用空格分割。
第一只或第二只老鼠吃到任意奶酪的得分1<=reward1[i],reward2[i]<=1000
第四行是一个整数k，代表第一只老鼠吃掉的奶酪块数。 0<=k<=n。

输出

一个整数，代表最大得分。

样例输入

```
4
1 1 3 4
4 4 1 1
2
```

样例输出

```
15
```

提示

贪心法



```python
def max_cheese_score(n, reward1, reward2, k):
    # 计算每块奶酪的得分差值
    differences = [(reward1[i] - reward2[i], i) for i in range(n)]
    
    # 按得分差值从大到小排序
    differences.sort(reverse=True, key=lambda x: x[0])
    
    # 初始化总得分
    total_score = 0
    
    # 选择前k块奶酪给第一只老鼠
    for i in range(k):
        index = differences[i][1]
        total_score += reward1[index]
    
    # 选择剩余的奶酪给第二只老鼠
    for i in range(k, n):
        index = differences[i][1]
        total_score += reward2[index]
    
    return total_score

# 输入处理
n = int(input())
reward1 = list(map(int, input().split()))
reward2 = list(map(int, input().split()))
k = int(input())

# 计算并输出结果
print(max_cheese_score(n, reward1, reward2, k))
```





## 28321: 电影排片

http://cs101.openjudge.cn/practice/28321/

一个电影排片系统现在要安排n部电影的上映，第i部电影的评分要求不小于bi。

现在已经有一个初步的上映电影的列表，长度为n，第i部电影的评分为ai。

最初，数组 {ai} 和 {bi} 均按升序排列。

由于某些电影的评分可能低于要求，即存在 ai < bi，因此我们需要添加评分更高的电影（假设我们有无数多的任何评分的电影供选择）。

当我们添加一个评分为 0 <= w <= 100 的新电影进入排片系统时，系统将删除评分最低的电影，并按升序排列电影的评分。

换句话说，在每次操作中，你选择一个整数 w，将其插入数组 {ai} 中，然后将数组 {ai} 按升序排列，并移除第一个元素。

求需要添加的最少新电影数量，使得对于所有的 i，满足 ai >= bi。

输入

每个测试包含多个测试用例。第一行包含测试用例的数量 t（1 ≤ t ≤ 10000）。每个测试用例的描述如下。

每个测试用例的第一行包含一个正整数 n（1 ≤ n ≤ 100），表示电影数量。

第二行包含一个长度为 n 的数组 a（0 ≤ a1 ≤ a2 ≤ ... ≤ an ≤ 100）。

第三行包含一个长度为 n 的数组 b（0 ≤ b1 ≤ b2 ≤ ... ≤ bn ≤ 100）。

输出

对于每个测试用例，输出一个整数作为答案，每个答案占一行。

样例输入

```
2
6
10 20 30 40 50 60
8 21 35 35 55 65

6
9 9 9 30 40 50
10 20 30 40 50 60
```

样例输出

```
1
3
```

提示

在第一个测试用例中：
\- 添加一个评分为 w = 70 的电影，数组 a 变为 [20, 30, 40, 50, 60, 70]。

在第二个测试用例中：
\- 添加一个评分为 w = 60 的电影，数组 a 变为 [9, 9, 30, 40, 50, 60]。
\- 添加一个评分为 w = 20 的电影，数组 a 变为 [9, 20, 30, 40, 50, 60]。
\- 添加一个评分为 w = 10 的电影，数组 a 变为 [10, 20, 30, 40, 50, 60]。





```python
from collections import deque

t = int(input())
for _ in range(t):
    n = int(input())
    a = deque(map(int, input().split()))
    b = deque(map(int, input().split()))

    cnt = 0
    i = 0
    while i < n:
        if a[i] < b[i]:
            cnt += 1
            a.popleft()
            a.append(b[-1] + 1)
            i = 0
            continue

        i += 1

    print(cnt)

```





## 28274: 细胞计数

http://cs101.openjudge.cn/practice/28274/

小北是一名勤奋的医学生，某天，他们实验室中新来了一台显微镜，这台显微镜拥有先进的图像分析功能，能够通过特殊的算法识别和分析细胞结构。显微镜能识别每个图像点的物质密度，并用1到9加以标识，对于空白区域则用数字0标识。同时我们规定，所有上、下、左、右联通，且数字不为0的图像点都属于同一细胞，请你帮助小北求出给定图像内的细胞个数。

输入

第一行两个整数代表矩阵大小n和m。
接下来n行，每行一个长度为m的只含字符 0 到 9 的字符串，代表这个nxm的矩阵。

输出

输出细胞个数

样例输入

```
4 10
0234500067
1034560500
2045600671
0000000089
```

样例输出

```
4
```



```python
def count_cells(n, m, matrix):
    visited = [[False]*m for _ in range(n)]
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]

    def dfs(x, y):
        visited[x][y] = True
        for i in range(4):
            nx, ny = x + dx[i], y + dy[i]
            if 0 <= nx < n and 0 <= ny < m and not visited[nx][ny] and matrix[nx][ny] != '0':
                dfs(nx, ny)

    count = 0
    for i in range(n):
        for j in range(m):
            if not visited[i][j] and matrix[i][j] != '0':
                dfs(i, j)
                count += 1

    return count

n, m = map(int, input().split())
matrix = [list(input().strip()) for _ in range(n)]
print(count_cells(n, m, matrix))
```



## 22460: 火星车勘探

http://cs101.openjudge.cn/practice/22460/

火星这颗自古以来寄托了中国人无限遐思的红色星球，如今第一次留下了中国人的印迹。2021年5月15日，“天问一号”探测器成功降落在火星预选着陆区。这标志着中国首次火星探测着陆任务取得成功，同时也使中国成为继美国之后第二个实现探测器着陆火星的国家。

假设火星车需要对形如二叉树的地形进行遍历勘察。火星车初始处于二叉树地形的根节点，对二叉树进行前序遍历。当火星车遇到非空节点时，则采样一定量的泥土样本，记录下样本数量；当火星车遇到空节点，使用一个标记值`#`进行记录。

![img](http://media.openjudge.cn/images/upload/3861/1621603311.png)

对上面的二叉树地形可以前序遍历得到 `9 3 4 # # 1 # # 2 # 6 # #`，其中 `#` 代表一个空节点，整数表示在该节点采样的泥土样本数量。

我们的任务是，给定一串以空格分隔的序列，验证它是否是火星车对于二叉树地形正确的前序遍历结果。



输入

每组输入包含多个测试数据，每个测试数据由两行构成。
每个测试数据的第一行：1个正整数N，表示遍历结果中的元素个数。
每个测试数据的第二行：N个以空格分开的元素，每个元素可以是#，也可以是小于100的正整数。(1<=N<=200000)
输入的最后一行为0，表示输入结束。

输出

对于每个测试数据，输出一行判断结果。
输入的序列如果是对某个二叉树的正确的前序遍历结果，则输出“T”，否则输出“F”。

样例输入

```
13
9 3 4 # # 1 # # 2 # 6 # #
4
9 # # 1
2
# 99
0
```

样例输出

```
T
F
F
```



验证给定的前序遍历结果是否能构成一个有效的二叉树。需要通过维护节点的出入度来检查遍历是否有效。

在前序遍历中，每个非空节点会贡献2个出度（左右子节点），而每个节点（包括空节点）会消耗1个入度。因此，我们可以通过维护一个计数器来记录当前剩余的可用入度。

```python
def is_valid_preorder(n, sequence):
    out_degree = 1  # Initial out_degree for the root

    for value in sequence:
        out_degree -= 1  # Every node uses one out_degree

        if out_degree < 0:
            return "F"

        if value != '#':
            out_degree += 2  # Non-null nodes provide 2 out_degrees

    return "T" if out_degree == 0 else "F"

# 读取输入
import sys
input = sys.stdin.read

data = input().strip().split('\n')
index = 0

while index < len(data):
    n = int(data[index])
    if n == 0:
        break
    index += 1
    sequence = data[index].split()
    index += 1
    print(is_valid_preorder(n, sequence))

```



输出 F 有两种可能, 要么节点多了, 要么少了; 多了就是建完树s还有剩余, 少了就是s空了还会pop, 就会raise一个IndexError

```python
# 22n2200011610 罗熙佑
class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


def build_tree(s):
    val = s.pop()
    if val == '#':
        return None
    
    root = Node(val)
    root.left = build_tree(s)
    root.right = build_tree(s)
    return root


while True:
    try:
        n = int(input())
        if n == 0:
            break
        s = input().split()[::-1]
        _ = build_tree(s)
        print('F' if len(s) else 'T')
    except IndexError:
        print('F')
```





## 28389: 跳高

http://cs101.openjudge.cn/practice/28389/

体育老师组织学生进行跳高训练，查看其相对于上一次训练中跳高的成绩是否有所进步。为此，他组织同学们按学号排成一列进行测试。本次测验使用的老式测试仪，只能判断同学跳高成绩是否高于某一预设值，且由于测试仪器构造的问题，其横杠只能向上移动。由于老师只关心同学是否取得进步，因此老师只将跳高的横杠放在该同学上次跳高成绩的位置，查看同学是否顺利跃过即可。为了方便进行上次成绩的读取，同学们需按照顺序进行测验，因此对于某个同学，当现有的跳高测试仪高度均高于上次该同学成绩时，体育老师需搬出一个新的测试仪进行测验。已知同学们上次测验的成绩，请问体育老师至少需要使用多少台测试仪进行测验？

由于采用的仪器精确度很高，因此测试数据以毫米为单位，同学们的成绩为正整数，最终测试数据可能很大，但不超过10000，且可能存在某同学上次成绩为0。

输入

输入共两行，第一行为一个数字N，N<=100000，表示同学的数量。第二行为N个数字，表示同学上次测验的成绩（从1号到N号排列）。

输出

一个正整数，表示体育老师最少需要的测试仪数量。

样例输入

```
5
1 7 3 5 2
```

样例输出

```
3
```

提示

1.50%的数据中，N<=5000.100%的数据中，N<=100000.
2.可通过观察规律将题目转化为学过的问题进行求解。
3.对于10w的数据，朴素算法可能会超时，可以采用二分法进行搜索上的优化。



这个题目挺好的，维护多个非递减队列。

```python
from bisect import bisect_right


def min_instruments_needed(scores):
    instruments = []  # 用于存储当前的各个递增序列的最后一个元素

    for score in scores:
        if instruments:
            # 找到第一个大于或等于当前成绩的位置
            pos = bisect_right(instruments, score)
            if pos == 0:
                instruments.insert(0, score)
            else:
                instruments[pos - 1] = score

        else:
            instruments.append(score)

    return len(instruments)


N = int(input())
scores = list(map(int, input().split()))

result = min_instruments_needed(scores)
print(result)

"""
5
1 2 1 3 2

2
"""
```



```python
"""
Dilworth定理:
Dilworth定理表明，任何一个有限偏序集的最长反链(即最长下降子序列)的长度，
等于将该偏序集划分为尽量少的链(即上升子序列)的最小数量。
因此，计算序列的最长下降子序列长度，即可得出最少需要多少台测试仪。
"""

from bisect import bisect_left

def min_testers_needed(scores):
    scores.reverse()  # 反转序列以找到最长下降子序列的长度
    lis = []  # 用于存储最长上升子序列

    for score in scores:
        pos = bisect_left(lis, score)
        if pos < len(lis):
            lis[pos] = score
        else:
            lis.append(score)

    return len(lis)


N = int(input())
scores = list(map(int, input().split()))

result = min_testers_needed(scores)
print(result)
```



# 2024春-数据结构与算法B-9班

Updated 1417 GMT+8 Jun 5, 2024
2024 spring, Complied by Hongfei Yan



2024-05-31 13:00 ~ 15:00

http://dsaex.openjudge.cn/2024final/ 



题目挺友好的，6个题目，gpt能做5个。



## 28348: 单链表

http://dsaex.openjudge.cn/2024final/A/

实现一个单链表，链表初始为空，支持三种操作：

1. 向链表头插入一个数；
2. 删除第 k 个插入的数后面的一个数；
3. 在第 k 个插入的数后插入一个数。

现在要对该链表进行 M 次操作，进行完所有操作后，从头到尾输出整个链表。

输入

第一行为整数M，表示操作次数。
接下来M行，每行包含一个操作命令，操作命令可能为以下几种：

1. H x，表示向链表头插入一个数x。
2. D k，表示删除第k个插入的数后面的数（当k为0时，表示删除头节点）。
3. I k x，表示在第k个插入的数后面插入一个数x（此操作k均大于0）。

输出

共一行，将整个链表从头到尾输出。

样例输入

```
10
H 9
I 1 1
D 1
D 0
H 6
I 3 6
I 4 5
I 4 5
I 3 4
D 6
```

样例输出

```
6 4 6 5
```

提示

1≤M≤100000
插入的数x为非负整数且均不超过100
所有操作保证合法。

来源

acwing



```python
#蒋子轩 23工学院
class Node:
    def __init__(self,num,val):
        self.val=val
        self.num=num
        self.next=None
def find(x):
    cur=head
    while cur.num!=x:
        cur=cur.next
    return cur
m=int(input());head=None;cnt=0
for _ in range(m):
    a=input().split()
    if a[0]=='H':
        cnt+=1
        node=Node(cnt,int(a[1]))
        node.next=head
        head=node
    elif a[0]=='D':
        if a[1]=='0':
            head=head.next;continue
        cur=find(int(a[1]))
        cur.next=cur.next.next
    else:
        cnt+=1;cur=find(int(a[1]))
        node=Node(cnt,int(a[2]))
        node.next=cur.next
        cur.next=node
while head:
    print(head.val,end=' ')
    head=head.next
```



```c
#include <stdio.h>
#include <stdlib.h>

typedef struct ListNode {
    int val;
    struct ListNode *next;
} ListNode;

typedef struct {
    ListNode *head;
    ListNode **inserts;
    int insertCount;
} LinkedList;

LinkedList* createLinkedList(int maxSize) {
    LinkedList *list = (LinkedList*) malloc(sizeof(LinkedList));
    list->head = NULL;
    list->inserts = (ListNode**) malloc(sizeof(ListNode*) * maxSize);
    list->insertCount = 0;
    return list;
}

void insertAtHead(LinkedList *list, int x) {
    ListNode *newNode = (ListNode*) malloc(sizeof(ListNode));
    newNode->val = x;
    newNode->next = list->head;
    list->head = newNode;
    list->inserts[list->insertCount++] = newNode;
}

void deleteAfterK(LinkedList *list, int k) {
    if (k == 0 && list->head != NULL) {
        ListNode *toDelete = list->head;
        list->head = list->head->next;
        free(toDelete);
    } else if (k > 0 && k <= list->insertCount) {
        ListNode *target = list->inserts[k - 1];
        if (target->next != NULL) {
            ListNode *toDelete = target->next;
            target->next = toDelete->next;
            free(toDelete);
        }
    }
}

void insertAfterK(LinkedList *list, int k, int x) {
    if (k > 0 && k <= list->insertCount) {
        ListNode *newNode = (ListNode*) malloc(sizeof(ListNode));
        newNode->val = x;
        ListNode *target = list->inserts[k - 1];
        newNode->next = target->next;
        target->next = newNode;
        list->inserts[list->insertCount++] = newNode;
    }
}

void printList(LinkedList *list) {
    ListNode *curr = list->head;
    while (curr != NULL) {
        printf("%d ", curr->val);
        curr = curr->next;
    }
    printf("\n");
}

void freeList(LinkedList *list) {
    ListNode *curr = list->head;
    while (curr != NULL) {
        ListNode *next = curr->next;
        free(curr);
        curr = next;
    }
    free(list->inserts);
    free(list);
}

int main() {
    int M;
    scanf("%d", &M);
    LinkedList *list = createLinkedList(M);

    for (int i = 0; i < M; ++i) {
        char op;
        int x, k;
        scanf(" %c", &op);
        if (op == 'H') {
            scanf("%d", &x);
            insertAtHead(list, x);
        } else if (op == 'D') {
            scanf("%d", &k);
            deleteAfterK(list, k);
        } else if (op == 'I') {
            scanf("%d %d", &k, &x);
            insertAfterK(list, k, x);
        }
    }

    printList(list);
    freeList(list);

    return 0;
}

```





## 28374: 机器翻译

http://dsaex.openjudge.cn/2024final/B/

小晨的电脑上安装了一个机器翻译软件，他经常用这个软件来翻译英语文章。

这个翻译软件的原理很简单，它只是从头到尾，依次将每个英文单词用对应的中文含义来替换。

对于每个英文单词，软件会先在内存中查找这个单词的中文含义，如果内存中有，软件就会用它进行翻译；如果内存中没有，软件就会在外存中的词典内查找，查出单词的中文含义然后翻译，并将这个单词和译义放入内存，以备后续的查找和翻译。 

假设内存中有 M 个单元，每单元能存放一个单词和译义。

每当软件将一个新单词存入内存前，如果当前内存中已存入的单词数不超过 M−1，软件会将新单词存入一个未使用的内存单元；若内存中已存入 M 个单词，软件会清空最早进入内存的那个单词，腾出单元来，存放新单词。 

假设一篇英语文章的长度为 N 个单词。

给定这篇待译文章，翻译软件需要去外存查找多少次词典？

假设在翻译开始前，内存中没有任何单词。

输入

输入文件共2行，每行中两个数之间用一个空格隔开。 
第一行为两个正整数M和N，代表内存容量和文章的长度。 
第二行为N个非负整数，按照文章的顺序，每个数（大小不超过1000）代表一个英文单词。
文章中两个单词是同一个单词，当且仅当它们对应的非负整数相同。

输出

输出文件共1行，包含一个整数，为软件需要查词典的次数。

样例输入

```
3 7
1 2 1 5 4 4 1
```

样例输出

```
5
```

提示

1≤M≤100
1≤N≤1000

来源

acwing





```python
#蒋子轩 23工学院
from collections import deque
m,n=map(int,input().split())
cnt=0;a=deque()
for i in list(map(int,input().split())):
    if i not in a:
        cnt+=1
        a.append(i)
        if len(a)>m:
            a.popleft()
print(cnt)
```





```c++
#include <iostream>
#include <unordered_set>
#include <queue>
#include <vector>

using namespace std;

int main() {
    int M, N;
    cin >> M >> N;
    
    vector<int> words(N);
    for (int i = 0; i < N; ++i) {
        cin >> words[i];
    }
    
    unordered_set<int> memory; // To store the words in memory
    queue<int> order;          // To maintain the order of insertion in memory
    int lookup_count = 0;      // Number of dictionary lookups
    
    for (int word : words) {
        if (memory.find(word) == memory.end()) { // Word not in memory
            lookup_count++;
            if (memory.size() >= M) {
                int oldest = order.front();
                order.pop();
                memory.erase(oldest);
            }
            memory.insert(word);
            order.push(word);
        }
    }
    
    cout << lookup_count << endl;
    return 0;
}

```





## 28375: 信息加密

http://dsaex.openjudge.cn/2024final/C/

在传输信息的过程中，为了保证信息的安全，我们需要对原信息进行加密处理，形成加密信息，从而使得信息内容不会被监听者窃取。

现在给定一个字符串，对其进行加密处理。

加密的规则如下：

1. 字符串中的小写字母，a">a 加密为 b">b，b">b 加密为 c">c，…，y">y 加密为 z">z，z">z 加密为 a">a。
2. 字符串中的大写字母，A">A 加密为 B">B，B">B 加密为 C">C，…，Y">Y 加密为 Z">Z，Z">Z 加密为 A">A。
3. 字符串中的其他字符，不作处理。

请你输出加密后的字符串。

输入

共一行，包含一个字符串。注意字符串中可能包含空格。

输出

输出加密后的字符串。

样例输入

```
Hello! How are you!
```

样例输出

```
Ifmmp! Ipx bsf zpv!
```

提示

输入字符串的长度不超过100000。

来源

acwing





```python
#蒋子轩 23工学院
s=input()
a=''
for i in s:
    t=ord(i)
    if ord('a')<=t<=ord('z'):
        t=(t-ord('a')+1)%26+ord('a')
    elif ord('A')<=t<=ord('Z'):
        t=(t-ord('A')+1)%26+ord('A')
    a+=chr(t)
print(a)
```





```c++
#include <iostream>
#include <string>

using namespace std;

string encrypt(const string& input) {
    string result;
    for (char c : input) {
        if (c >= 'a' && c <= 'z') {
            result += (c == 'z') ? 'a' : c + 1;
        } else if (c >= 'A' && c <= 'Z') {
            result += (c == 'Z') ? 'A' : c + 1;
        } else {
            result += c;
        }
    }
    return result;
}

int main() {
    string input;
    getline(cin, input);

    string encrypted_string = encrypt(input);
    cout << encrypted_string << endl;

    return 0;
}

```



## 28360: 艾尔文的探险

http://dsaex.openjudge.cn/2024final/D/

在遥远的Teyvat大陆上，有一位聪明但有些古怪的学者，名叫艾尔文。他酷爱研究各种复杂的谜题和数学问题。一天，他听闻了一个神秘的传说，说在遥远的森林深处有一座神秘的神殿，守护着一卷古老的卷轴。据说，这卷卷轴蕴含着巨大的财富，但要解开其中的秘密，需要解决一个复杂的问题。

传说中，神殿里的卷轴上写满了由‘（’和‘）’两种符号组成的文字，隐藏着一个巨大的谜题。谜题的核心是寻找其中最长的格式正确的括号子串。这项任务看似简单，但实际上极为艰巨，因为括号的数量之多令人望而生畏。

艾尔文听闻这个传说，兴奋不已，决心前往挑战。然而，当他终于找到神殿，展开卷轴时，眼前的景象让他大吃一惊。卷轴上密密麻麻的括号让他眼花缭乱，算力不足，他无法一眼看清其中的奥秘。

无奈之下，艾尔文只能无功而返，但他留下了这个问题，希望有更有智慧的人能够解开这个古老的谜题。

现在，你挑战这个问题吧：给出一个仅包含‘（’和‘）’的字符串，计算出其中最长的格式正确的括号子串的长度。

输入

一个仅包含‘（’和‘）’的字符串。

输出

计算出其中最长的格式正确的括号子串的长度。

样例输入

```
(())(()
```

样例输出

```
4
```

提示

字符串的长度最大为2*10^6





```python
#蒋子轩 23工学院
s=input();a=[-1];ans=0
for i,c in enumerate(s):
    if c=='(':
        a.append(i)
    else:
        a.pop()
        if a:
            ans=max(ans,i-a[-1])
        else:
            a.append(i)
print(ans)
```





```python
#include<bits/stdc++.h>
using namespace std;

int main() {
    string s;
    cin >> s;
    stack<int> stk;
    stk.push(-1);  // 初始化栈，压入-1作为哨兵
    int maxLen = 0;
    for(int i = 0; i < s.size(); i++) {
        if(s[i] == '(') {
            stk.push(i);  // 左括号，索引入栈
        } else {
            stk.pop();  // 右括号，弹出栈顶元素
            if(stk.empty()) {
                stk.push(i);  // 如果栈为空，将当前位置入栈
            } else {
                maxLen = max(maxLen, i - stk.top());  // 计算最大长度
            }
        }
    }
    cout << maxLen << endl;
    return 0;
}
```





## 28361: 能量果实

tree dp, http://dsaex.openjudge.cn/2024final/E/

在一个宁静的小镇边缘，有一座神秘的古老庄园。这座庄园已经存在了数百年，传说中它的每一块砖石都蕴含着无数的故事。庄园的主人是一位老园丁，他精心呵护着一棵与众不同的古树。这棵古树庞大无比，树上的每一个分枝都是老园丁用心栽培的结果。

这棵古树有着复杂的结构，每个节点都代表着树上的一个分枝，而每个分枝上都结着果实。这些果实不仅味美多汁，每一个果实还有一个特殊的数值，这些数值代表着果实所蕴含的神秘能量。老园丁知道，这些能量可以用来帮助小镇上的人们解决各种难题。然而，果实的采摘并不是那么简单，因为古树有一个古老的禁忌：任何时候采摘果实时，不能同时摘取具有父子关系的两个果实，否则树的魔力将会消失。

为了从这棵古树上获得最大的能量，老园丁决定在小镇上举办一个竞赛，邀请各方高手前来参与。参与者需要设计一个方案，选择一些果实，使得它们的能量总和最大，同时遵守禁忌。庄园门前贴出了这样一份通知：

**欢迎来到古老庄园的能量果实采摘竞赛！**
**竞赛要求**

1. 树的结构：古树的根节点编号为1。树的总结点数为n，每个节点都有一个独特的编号（范围在1~n之间）和一个非负整数值，代表着该节点上的果实所蕴含的能量。
2. 古树的特点：这是一棵 二叉树，每个节点最多有两个孩子。
3. 禁忌规则：为了保护古树的魔力，采摘的果实不能同时来自父子关系的两个节点。
4. 目标：设计一个算法，找出一个最佳的果实采摘方案，使得采摘的果实能量总和最大。

现在，你是这场果实采摘大赛的参与者，请求出果实能量总和的最大值。

输入

第一行：一个整数 n，表示树上的节点个数。

接下来的 n 行，每行三个整数 vi, li, ri，分别表示第 i 号节点的果实能量值、左孩子节点编号和右孩子节点编号。如果某个孩子节点不存在，则用0表示。

输出

一个整数，表示最大能量总和。

样例输入

```
5
6 2 3
2 4 5
3 0 0
4 0 0
5 0 0
```

样例输出

```
15
```

提示

1 ≤ n ≤ 2*10^5, 0 ≤ vi ≤ 5000



典型的树形动态规划(Tree DP)问题。在这个问题中,我们有一棵二叉树,每个节点都有一个值。我们需要找到一种方案,使得我们选择的节点值的总和最大,并且不能选择相邻的节点。

这个问题可以使用动态规划来解决,具体思路如下:

定义 `dp[i][0]` 表示不选择节点 i 的最大值,`dp[i][1]` 表示选择节点 i 的最大值。
使用深度优先搜索(DFS)遍历整棵树,并计算每个节点的 `dp[i][0] `和 `dp[i][1]`。
对于每个节点 i:
`dp[i][0] `等于它的左子树和右子树不选择时的最大值,即 `max(dp[left][0], dp[left][1]) + max(dp[right][0], dp[right][1])`;
`dp[i][1] `等于它的值加上它的左子树和右子树都不选择时的最大值,即 `tree[i].value + dp[left][0] + dp[right][0]`。
最终,答案就是 `max(dp[1][0], dp[1][1])`,其中 1 表示根节点。
这个解决方案的时间复杂度是 O(n),其中 n 是树的节点数。空间复杂度是 O(n),因为我们需要存储每个节点的 dp 值。

这个问题是一个经典的树形 DP 问题,通常会出现在面试和算法竞赛中。掌握这种解决方法对于理解和解决更复杂的树形 DP 问题很有帮助。





```python
#蒋子轩 23工学院
def dfs(x,t):
    global a
    if x==0:return 0
    l=a[x][1];r=a[x][2]
    if not t: return dfs(l,1)+dfs(r,1)
    return max(dfs(l,0)+dfs(r,0)+a[x][0],dfs(l,1)+dfs(r,1))
n=int(input());a={}
for i in range(1,n+1):
    a[i]=tuple(map(int,input().split()))
print(dfs(1,1))
```





```c++
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

struct TreeNode {
    int value;
    int left;
    int right;
};

void dfs(int node, const vector<TreeNode>& tree, vector<vector<int>>& dp) {
    if (node == 0) return;

    int left = tree[node].left;
    int right = tree[node].right;

    dfs(left, tree, dp);
    dfs(right, tree, dp);

    dp[node][0] = max(dp[left][0], dp[left][1]) + max(dp[right][0], dp[right][1]);
    dp[node][1] = tree[node].value + dp[left][0] + dp[right][0];
}

int main() {
    int n;
    cin >> n;

    vector<TreeNode> tree(n + 1);
    for (int i = 1; i <= n; ++i) {
        cin >> tree[i].value >> tree[i].left >> tree[i].right;
    }

    vector<vector<int>> dp(n + 1, vector<int>(2, 0));
    dfs(1, tree, dp);

    cout << max(dp[1][0], dp[1][1]) << endl;

    return 0;
}

```





## 28362: 魔法森林

http://dsaex.openjudge.cn/2024final/F/

在古老的幻境国度，有一片传说中的幽秘森林。这片森林里生长着无数的魔法树，每一棵树上都栖息着爱搞恶作剧的精灵。整片森林被神秘的单向魔法路径缠绕，形成了一张扑朔迷离的网络。这些路径只能单向通行，无法逆行。

传说中，森林里的魔法树共有 N 棵，它们被奇异的编号标记，从 1 到 N。森林中存在 M 条单向魔法路径，每条路径将两棵树连接起来。

某天，森林里的精灵们突发奇想，决定玩一个令人费解的游戏：每个精灵都要找到从自己栖息的魔法树出发，沿着单向的魔法路径，最终能够到达的编号最小的魔法树。传说在这棵编号最小的魔法树上，隐藏着一个无尽智慧的宝藏。

精灵们的聪明才智似乎到达了极限，于是它们向外界发布了一个挑战，希望有智慧的冒险者们能够解开这个谜题。你，作为一个足智多谋的冒险者，决定接受这个挑战。

你的任务是：对于每一棵魔法树 v，找出从它出发，沿着单向的魔法路径，能够到达的编号最小的魔法树。你需要编写一个程序来实现这一点，并帮助精灵们找到它们的梦想之树。



输入

第一行包含两个整数 N 和 M，分别表示魔法树的数量和魔法路径的数量。

接下来的 M 行，每行包含两个整数 u 和 v，表示存在一条从魔法树 u 到魔法树 v 的单向魔法路径。

输出

用空格隔开的 N 个数，第 i 个数表示从第 i 棵魔法树出发，能够到达的编号最小的魔法树。

样例输入

```
5 4
1 3
3 4
4 5
4 2
```

样例输出

```
1 2 2 2 5
```

提示

1 ≤ N, M ≤ 10^5



反向建图。

访问过的节点不需要再走一次了，但是应该需要防止成环的情况。

从小到大遍历的时候就不用判断了。

```python
# http://dsaex.openjudge.cn/2024final/F/
# 夏天明
def dfs(i, label):
    if ans[i] is None:
        ans[i] = label
        for nex in graph[i]:
            dfs(nex, label)

N, M = map(int, input().split())
graph = [[] for i in range(N)]
ans = [None] * N

for _ in range(M):
    u, v = map(int, input().split())
    graph[v - 1].append(u - 1)

for i in range(N):
    if ans[i] is None:
        dfs(i, i + 1)

print(*ans)
```







```python
#蒋子轩 23工学院
def dfs(u,mini):
    if ans[u]==-1:
        ans[u]=mini
        for v in a[u]:
            dfs(v,mini)
n,m=map(int,input().split())
a={i:[] for i in range(1,n+1)}
for _ in range(m):
    u,v=map(int,input().split())
    a[v].append(u)
ans=[-1]*(n+1)
for i in range(1,n+1):
    dfs(i,i)
print(*ans[1:])
```



# 2024春季数算B5班机考

Updated 2117 GMT+8 Jun 7, 2024

2024 spring, Complied by Hongfei Yan



数算xzm班

http://xzmdsa.openjudge.cn/2024final/



| 题目                         | tags                 |
| ---------------------------- | -------------------- |
| 坏掉的键盘                   | string               |
| 餐厅订单页面设计             | sortings             |
| 抓住那头牛                   | bfs                  |
| 正方形                       | math, implementation |
| Taki的乐队梦想               | 拓扑排序变形         |
| 斗地主大师                   | 二分查找dfs          |
| 小明的刷题计划               | 二分查找             |
| 银河贸易问题                 | bfs                  |
| Taki的乐队梦想（数据强化版） | 拓扑排序变形         |





## 27378: 坏掉的键盘

http://cs101.openjudge.cn/practice/27378/

小张在机房找到一个坏掉的键盘，26个字母只有一个键坏掉了，当遇到这个字母时，小张会用'.'（英文句号）替代。请你将小张的文字还原。

**输入**

首先输入一个字符c，表示坏掉的键。

第二行有一串只包括剩余25个字母，空格以及英文句点的字符串，请你帮忙还原

**输出**

还原后的字符串

样例输入

```
i
look . f.nd a d.rty keyboard.
```

样例输出

```
look i find a dirty keyboardi
```





```python
t = input()
print(input().replace('.', t))
```







## 28404: 餐厅订单页面设计

http://xzmdsa.openjudge.cn/2024final/2/

你是餐厅订单系统的设计师，你需要设计一个页面展示客户的点菜订单。

给你一个数组orders，表示客户在餐厅中完成的订单（订单编号记为i），orders[i]=[customerName[i], tableNumber[i], foodItem[i]]，其中customerName[i]是客户的姓名，tableNumber[i]是客户所在餐桌的桌号，而foodItem[i]是客户点的餐品名称。

请你设计该餐厅的订单页面，用一张表表示。在这张表中，第一行为标题，其第一列为餐桌桌号"Table",后面每一列都是按字母顺序排列的餐品名称。接下来的每一行中的项则表示每张餐桌订购的相应餐品数量，第一列应当填对应的桌号，后面依次填写下单的餐品数量。

注意：客户姓名不是点菜展示表的一部分。此外，表中的数据行应按照餐桌桌号升序排列。

**输入**

第1行是orders的个数N
第1至N+1行是多行orders，每行orders由三个元素组成：customerName[i], tableNumber[i], foodItem[i]，其中customerName[i]是客户的姓名，tableNumber[i]是客户所在餐桌的桌号，而foodItem[i]是客户点的餐品名称。每个元素由','分隔。

**输出**

一张订单展示表。在这张表中，第一行为标题，其第一列为餐桌桌号"Table",后面每一列都是按字母顺序排列的餐品名称。
接下来的每一行中的项则表示每张餐桌订购的相应餐品数量，第一列应当填对应的桌号，后面依次填写下单的餐品数量。
注意:为了订单展示表的美观，每行数据有空行分隔，每行数据中的每个元素由制表符分割('\t')

样例输入

```
6
David,3,Ceviche
Corina,10,Beef-Burrito
David,3,Fried-Chicken
Carla,5,Water
Carla,5,Ceviche
Rous,3,Ceviche
```

样例输出

```
Table   Beef-Burrito    Ceviche Fried-Chicken   Water

3       0       2       1       0

5       0       1       0       1

10      1       0       0       0
```

提示

1.1<=orders.length<=5*10^4;
2.orders[i].length==3;
3.1<=customerName[i].length,foodItem[i].length<=20;
4.customerName[i]和foodItem[i]有大小写英文字母，连字符'-'及空格' '组成；
5.tableNumber[i]是1到500范围内的整数。





```python
# 23物院 宋昕杰
tables = {}
dishes = {}
for _ in range(int(input())):
    name, table, dish = input().split(',')
    table = int(table)
    if table not in tables:
        tables[table] = True
    if dish not in dishes:
        dishes[dish] = {}
    if table not in dishes[dish]:
        dishes[dish][table] = 0
    dishes[dish][table] += 1

keys = sorted(list(dishes.keys()))
tables = sorted(list(tables.keys()))
print('\t'.join(['Table'] + keys))
print()
for table in tables:
    ls = [str(table)]
    for dish in keys:
        if table not in dishes[dish]:
            ls.append('0')
        else:
            ls.append(str(dishes[dish][table]))
    print('\t'.join(ls))
    print()
```



## 04001: 抓住那头牛

http://cs101.openjudge.cn/practice/04001/

农夫知道一头牛的位置，想要抓住它。农夫和牛都位于数轴上，农夫起始位于点N(0<=N<=100000)，牛位于点K(0<=K<=100000)。农夫有两种移动方式：

1、从X移动到X-1或X+1，每次移动花费一分钟

2、从X移动到2*X，每次移动花费一分钟

假设牛没有意识到农夫的行动，站在原地不动。农夫最少要花多少时间才能抓住牛？



**输入**

两个整数，N和K

**输出**

一个整数，农夫抓到牛所要花费的最小分钟数

样例输入

```
5 17
```

样例输出

```
4
```



bfs+剪枝

```python
# 23物院 宋昕杰
from queue import Queue

n, k = map(int, input().split())
q = Queue()
vis = {}

q.put((n, 0))
while True:
    x, t = q.get()
    if x == k:
        print(t)
        exit()

    if x in vis:
        continue
    vis[x] = True

    t += 1
    if x < k:
        q.put((2*x, t))
    q.put((x + 1, t))
    if x > 0:
        q.put((x - 1, t))
```



## 02002: 正方形

http://cs101.openjudge.cn/practice/02002/

给定直角坐标系中的若干整点，请寻找可以由这些点组成的正方形，并统计它们的个数。

**输入**

包括多组数据，每组数据的第一行是整点的个数n(1<=n<=1000)，其后n行每行由两个整数组成，表示一个点的x、y坐标。输入保证一组数据中不会出现相同的点，且坐标的绝对值小于等于20000。输入以一组n=0的数据结尾。

**输出**

对于每组输入数据，输出一个数，表示这组数据中的点可以组成的正方形的数量。

样例输入

```
4
1 0
0 1
1 1
0 0
9
0 0
1 0
2 0
0 2
1 2
2 2
0 1
1 1
2 1
4
-2 5
3 7
0 0
5 2
0
```

样例输出

```
1
6
1
```



For each pair of points, it checks the two possible orientations of squares (clockwise and counterclockwise) by calculating the required third and fourth points.

```python
def count_squares(points):
    point_set = set(points)
    count = 0

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1 = points[i]
            p2 = points[j]
            
            if p1 == p2:
                continue
            
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            
            p3 = (p1[0] + dy, p1[1] - dx)
            p4 = (p2[0] + dy, p2[1] - dx)
            
            if p3 in point_set and p4 in point_set:
                count += 1
            
            p3 = (p1[0] - dy, p1[1] + dx)
            p4 = (p2[0] - dy, p2[1] + dx)
            
            if p3 in point_set and p4 in point_set:
                count += 1

    return count // 4

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    while True:
        n = int(data[index])
        index += 1
        if n == 0:
            break
        
        points = []
        for _ in range(n):
            x = int(data[index])
            y = int(data[index + 1])
            points.append((x, y))
            index += 2
        
        print(count_squares(points))

if __name__ == "__main__":
    main()
```

Explanation

1. **Function `count_squares(points)`**:
   - It takes a list of points and returns the number of squares that can be formed by these points.
   - It uses a set to quickly check if a point exists.
   - For each pair of points, it checks the two possible orientations of squares (clockwise and counterclockwise) by calculating the required third and fourth points.
   - It counts these squares and divides by 4 at the end because each square is counted four times (once for each pair of its points).
2. **Function `main()`**:
   - Reads input from standard input (usually used in competitive programming).
   - Processes multiple sets of data until a set with `n = 0` is encountered.
   - For each set, it collects the points and uses `count_squares` to find the number of squares.
   - Prints the result for each set.





```python
# 23物院 宋昕杰
import sys

input = lambda : sys.stdin.readline().strip()

while n := int(input()):

    ls = []
    dic = {}
    vis = {}

    for _ in range(n):
        x, y = map(int, input().split())
        ls.append((x, y))
        dic[(x, y)] = True

    ls.sort()
    cnt = 0
    for i in range(n - 1):
        x1, y1 = ls[i]
        for j in range(i + 1, n):
            x2, y2 = ls[j]
            dx, dy = x2 - x1, y2 - y1

            x3, y3 = x1 - dy, y1 + dx
            x4, y4 = x2 - dy, y2 + dx

            key = tuple(sorted([(x1, y1), (x2, y2), (x3, y3), (x4, y4)]))
            if (x3, y3) in dic and (x4, y4) in dic and key not in vis:
                cnt += 1
                vis[key] = True
    print(cnt)
```



## 28413: Taki的乐队梦想

http://xzmdsa.openjudge.cn/2024final/5/

Taki有一个组乐队的梦想。她和她的伙伴们一起组建了一支乐队，但是这个过程并不是很顺利……

好不容易办完了第一次演出，乐队有了雏形，但是很快打击接踵而至，成员们又各自分离。

为了挽救乐队，Taki试图有所行动。她开始逐个联系乐队成员，希望她们能够回来组乐队。

她的队友们具有一些怪异的个性：每个乐队成员都有一个单方面对之“相爱相杀”的队友清单。也有的队员性格平和，那么她的清单就是空的。
如果“相爱相杀”的队友都已归队，Taki一联系这名队员，她就会马上归队；但遗憾的是，她归队的同时又会把“相爱相杀”的队友都气走。

幸好，在不愿透露姓名的工作人员S的帮助下，Taki对乐队成员进行了一次排序编号。保证每个队员对之“相爱相杀”的对象编号都严格小于队员自己。在这种情况下，无疑是可以让所有乐队成员都归队的。

请告诉Taki，为了让所有乐队成员归队，她最少需要联系多少次乐队成员，并且给出字典序最小的一种联系方案。

（这里的字典序是指乐队成员排序的顺序，而非名字顺序或者别的什么顺序）

**输入**

本题有多组数据。第一行是一个正整数T，表示数据组数。
每组数据的第一行是一个正整数n，表示乐队成员数量。
接下来n行，每行表示一名乐队成员的信息：开头是一个字符串，表示乐队成员的姓名（由字母和数字组成，没有空格），后面跟着零个或者若干个数字，表示她的相关人员的序号（序号从1开始计数）

数据保证一定存在让所有人都归队的联系方式。

**输出**

对于每组数据，第一行输出一个数字m，表示最小联系次数。
第二行输出m个姓名，中间用空格分割，表示字典序最小的联系方案下，每一次联系的人的姓名。

样例输入

```
3
3
A
B 1
C 2 1
4
A
B 1
C 2
D 3
5
Takamatsu
Kaname
Shiina 2
Chihaya 1
Nagasaki 1 4
```

样例输出

```
7
A B A C A B A
10
A B A C B A D C B A
10
Takamatsu Kaname Shiina Kaname Chihaya Takamatsu Nagasaki Takamatsu Chihaya Takamatsu
```



拓扑排序变形（9也可以过）

```python
# 23物院 宋昕杰
from heapq import *
import sys

input = lambda: sys.stdin.readline().strip()

for _ in range(int(input())):
    n = int(input())

    names = []
    edges = {i: {} for i in range(n)}
    hated = {i: {} for i in range(n)}
    in_degree = [0]*n
    cnt = 0

    for i in range(n):
        ls = input().split()
        names.append(ls[0])
        for t in ls[1:]:
            t = int(t) - 1
            edges[t][i] = hated[i][t] = True
            in_degree[i] += 1

    heap = []
    for i in range(n):
        if in_degree[i] == 0:
            heap.append(i)
    heapify(heap)
    ls = []
    inside = [False]*n
    while cnt < n:
        idx = heappop(heap)
        ls.append(names[idx])
        cnt += 1
        for i in hated[idx]:
            cnt -= 1
            for j in edges[i]:
                in_degree[j] += 1
            inside[i] = False
        for i in edges[idx]:
            in_degree[i] -= 1

        for i in hated[idx]:
            if in_degree[i] == 0 and not inside[i]:
                inside[i] = True
                heappush(heap, i)
        for i in edges[idx]:
            if in_degree[i] == 0 and not inside[i]:
                inside[i] = True
                heappush(heap, i)
    print(len(ls))
    print(*ls)
```





## 24837: 斗地主大师

http://cs101.openjudge.cn/practice/24837/

斗地主大师今天有P个欢乐豆，他夜观天象，算出了一个幸运数字Q，如果他能有恰好Q个欢乐豆，就可以轻松完成程设大作业了。

斗地主大师显然是斗地主大师，可以在斗地主的时候轻松操控游戏的输赢。

1.他可以轻松赢一把，让自己的欢乐豆变成原来的Y倍

2.他也可以故意输一把，损失X个欢乐豆(注意欢乐豆显然不能变成负数，所以如果手里没有X个豆就不能用这个能力)

而斗地主大师还有一种怪癖，扑克除去大小王只有52张，所以他一天之内最多只会打52把斗地主。

斗地主大师希望你能告诉他，为了把P个欢乐豆变成Q个，他至少要打多少把斗地主？

**输入**

第一行4个正整数 P,Q,X,Y
0< P,X,Q <= 2^31, 1< Y <= 225

**输出**

输出一个数表示斗地主大师至少要用多少次能力
如果打了52次斗地主也不能把P个欢乐豆变成Q个，请输出一行 "Failed"

样例输入

```
输入样例1：
2 2333 666 8
输入样例2：
1264574 285855522 26746122 3
```

样例输出

```
输出样例1：
Failed
输出样例2：
33
```

提示

可以考虑深搜
要用long long



二分查找dfs

```python
# 23物院 宋昕杰
p, q, x, y = map(int, input().split())
dp = {q: 0}
mid = 0


def dfs(p, cnt):
    if cnt > mid:
        return False
    if p > q and (p - q)//x + cnt > mid:
        return False
    elif p > q and (p - q)//x + cnt <= mid and (p - q) % x == 0:
        return True

    if p == q:
        return True

    cnt += 1
    if p > x and dfs(p - x, cnt):
        return True
    if dfs(p*y, cnt):
        return True
    return False


mid = 52
if not dfs(p, 0):
    print('Failed')
    exit()

l, r = 0, 52
while l < r:
    mid = (l + r)//2

    if dfs(p, 0):
        r = mid
    else:
        l = max(mid, l + 1)
print(r)
```



## 28405: 小明的刷题计划

http://xzmdsa.openjudge.cn/2024final/7/

为了提高自己的代码能力，小明制定了OpenJudge的刷题计划。他选中OpenJudge中的n道题，编号0~n-1，并计划在m天内按照题目编号顺序刷完所有题目。提醒：小明不能用多天完成同一题。

在小明刷题计划中，需要用time[i]的时间完成编号i的题目。同时，小明可以使用场外求助功能，通过询问编程高手小红可以省去该题的做题时间。为了防止小明过度依赖场外求助，小明每天只能使用一次求助机会。

我们定义m天中做题时间最多的一天耗时为T（小红完成的题目不计入做题总时间）。

请你帮助小明指定做题计划，求出最小的T是多少。

**输入**

第一行小明需要完成的题目个数N；
第二至N+1行小明完成每个题目的时间time；
最后一行小明计划完成刷题任务的时间m。

**输出**

做题最多一天的耗时T。

样例输入

```
4
1
2
3
3
2
```

样例输出

```
3
```

提示

1.1<=time.length<=10^5;
2.1<=time[i]<10000;
3.1<=m<=1000



二分查找

```python
# 23物院 宋昕杰
n = int(input())
times = [int(input()) for _ in range(n)]
m = int(input())

l, r = 0, sum(times) + 1


def judge(t):
    i = 0
    cnt = 0
    while i < n:
        total = 0
        max_today = 0
        while i < n and total <= t:
            total += times[i]
            max_today = max(max_today, times[i])
            i += 1

        total -= max_today

        while i < n:
            if total + times[i] <= t:
                total += times[i]
                i += 1
            else:
                break
        cnt += 1

    return cnt


while l < r:
    mid = (l + r) // 2
    cnt = judge(mid)

    if cnt <= m:
        r = mid
    else:
        l = max(mid, l + 1)

print(r)
```



## 03447: 银河贸易问题

http://cs101.openjudge.cn/practice/03447/

随着一种称为“￥”的超时空宇宙飞船的发明，一种叫做“￥￥”的地球与遥远的银河之间的商品进出口活动应运而生。￥￥希望从PluralZ星团中的一些银河进口商品，这些银河中的行星盛产昂贵的商品和原材料。初步的报告显示：
（1） 每个银河都包含至少一个和最多26个行星，在一个银河的每个行星用A~Z中的一个字母给以唯一的标识。
（2） 每个行星都专门生产和出口一种商品，在同一银河的不同行星出口不同的商品。
（3） 一些行星之间有超时空货运航线连接。如果行星A和B相连，则它们可以自由贸易；如果行星C与B相连而不与A相连，则A和C之间仍可通过B进行贸易，不过B要扣留5%的货物作为通行费。一般来说，只要两个行星之间可以通过一组货运航线连接，他们就可以进行贸易，不过每个中转站都要扣留5%的货物作为通行费。
（4） 在每个银河至少有一个行星开放一条通往地球的￥航线。对商业来说，￥航线和其他星际航线一样。
￥￥已经对每个行星的主要出口商品定价（不超过10的正实数），数值越高，商品的价值越高。在本地市场，越值钱的商品获利也越高。问题是要确定如果要考虑通行费是，哪个行星的商品价值最高。

输入

输入包含若干银河的描述。每个银河的描述开始的第1行是一个整数n，表示银河的行星数。接下来的n行每行包括一个行星的描述，即：
（1） 一行用以代表行星的字母；
（2） 一个空格；
（3） 以d.dd的形式给出该行星的出口商品的价值；
（4） 一个空格；
（5） 一个包含字母和（或）字符“*”的字符串；字母表示一条通往该行星的货运航线；“*”表示该行星向地球开放￥货运航线。

输出

对每个银河的描述，输出一个字母P表示在考虑通行费的前提下，行星P具有最高价值的出口商品。如果用有最高价值的出口商品的行星多于一个，只需输出字母序最小的那个行星。

样例输入

```
5
E 0.01 *A
D 0.01 A*
C 0.01 *A
A 1.00 EDCB
B 0.01 A*
```

样例输出

```
A
```





和这道爆了。”输入包含若干银河的描述“以为是多组输入，价值理解反了，被硬控30分钟。

”输入包含若干银河的描述“应该改成”输入包含银河的若干描述“

```python
# 23物院 宋昕杰
from queue import Queue

n = int(input())

values = []
names = []
ways = []
name_to_idx = {}
for i in range(n):
    t = input().split()
    if len(t) == 2:
        t.append('')
    name, value, way = t

    names.append(name)
    values.append(int(value[0] + value[2:]))
    ways.append(way)
    name_to_idx[name] = i

real_values = []
for i in range(n):
    q = Queue()
    q.put((i, 0))
    found = False
    vis = {}
    while not found and not q.empty():
        idx, distance = q.get()
        if idx in vis:
            continue
        vis[idx] = True

        for name in ways[idx]:
            if name == '*':
                found = True
                real_values.append((values[i]*0.95**distance, names[i]))
                break
            q.put((name_to_idx[name], distance + 1))

    if not found:
        real_values.append((0, names[i]))

real_values.sort(key=lambda t: (-t[0], t[1]))
print(real_values[0][1])
```



```python
# 肖添天
from collections import defaultdict, deque

n = int(input())
graph = defaultdict(set)
to_earth = set()
price = {}
for i in range(n):
    a, b, c = input().split()
    b = float(b)
    price[a] = b if a not in price else max(price[a], b)
    for x in c:
        if x == "*":
            to_earth.add(a)
        else:
            graph[a].add(x)
            graph[x].add(a)

def bfs(start):
    Q = deque([start])
    visited = set()
    visited.add(start)
    cnt = 0
    while Q:
        l = len(Q)
        for _ in range(l):
            f = Q.popleft()
            if f in to_earth:
                return price[start] * (0.95 ** cnt)
            for x in graph[f]:
                if x not in visited:
                    Q.append(x)
                    visited.add(x)
        cnt += 1
    return 0


ans = []
for planet in price.keys():
    ans.append((bfs(planet), planet))

ans.sort(key=lambda x: [-x[0], x[1]])
print(ans[0][1])
```



## 28416: Taki的乐队梦想（数据强化版）

http://xzmdsa.openjudge.cn/2024final/9/

题目描述全部同上，唯一的区别是增强了数据的强度。

现在T=20，n<=3000，保证输出总长度不超过500kB。

输入

同上

输出

同上

样例输入

```
同上
```

样例输出

```
同上
```

提示

如果涉及到大量的列表合并需求的话，使用list1.extend(list2)会比list1+=list2快很多。基本可以认为extend操作是O(1)的。
你的代码应该具有O(输入规模+输出规模)的时间复杂度。
（当然，如果你能靠优化常数通过这道题也是一种能力的体现）



同 28413



# Xzm2023期末机考

02774: 木材加工

http://cs101.openjudge.cn/practice/02774/

02766: 最大子矩阵

http://cs101.openjudge.cn/practice/02766/

26573: 康托集的图像表示

http://cs101.openjudge.cn/practice/26573/

26572: 多余的括号

http://cs101.openjudge.cn/practice/26572/

06364: 牛的选举

http://cs101.openjudge.cn/practice/06364

03720: 文本二叉树

http://cs101.openjudge.cn/practice/03720/

05907: 二叉树的操作

http://cs101.openjudge.cn/practice/05907/



# Xzm2022期末机考

24684: 直播计票

http://cs101.openjudge.cn/practice/24684/

24677: 安全位置

http://cs101.openjudge.cn/practice/24677/

24676: 共同富裕

http://cs101.openjudge.cn/practice/24676/

24678: 任性买房

http://cs101.openjudge.cn/practice/24678/

24686: 树的重量

http://cs101.openjudge.cn/dsapre/24686/

24687: 封锁管控

http://cs101.openjudge.cn/practice/24687/



# xzm2020期末机考

20742: 泰波拿契數

http://cs101.openjudge.cn/practice/20742/

20743: 整人的提词本

http://cs101.openjudge.cn/practice/20743/

20741: 两座孤岛最短距离

http://cs101.openjudge.cn/practice/20741

20746: 满足合法工时的最少人数

http://cs101.openjudge.cn/practice/20746/

20626: 对子数列做XOR运算

http://cs101.openjudge.cn/practice/20626/

20744: 土豪购物

http://cs101.openjudge.cn/practice/20744/



# Xzm2020模拟上机

20449: 是否被5整除

http://cs101.openjudge.cn/practice/20449/

20453: 和为k的子数组个数

http://cs101.openjudge.cn/practice/20453/

20456: 统计封闭岛屿的数目

http://cs101.openjudge.cn/practice/20456/

20472: 死循环的机器人

http://cs101.openjudge.cn/practice/20472/

20625: 1跟0数量相等的子字串

http://cs101.openjudge.cn/practice/20625/

20644: 统计全为 1 的正方形子矩阵

http://cs101.openjudge.cn/practice/20644/
