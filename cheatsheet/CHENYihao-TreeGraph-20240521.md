# 数算期末复习 (feat. Chen)

## 🌲

### 树的基本题型

#### 1. 树的建立

##### class法

```python
class TreeNode:
	def __init__(self, value):
        # 二叉树（binary tree)
        self.value = value
		self.left = None
		self.right = None
        
        # 多叉树（N-nary tree)
        self.value = value
        self.children = []
        
        # 左儿子右兄弟树（First child / Next sibling representation）
		self.value = value
		self.firstChild = None
		self.nextSibling = None
		# 这玩意像个链表，有时候会很好用

n = int(input())
# 一般而言会有一个存Nodes的dict或是list
nodes = [TreeNode() for i in range(n)]
# 甚至会让你找root，这也可以用于记录森林的树量
has_parents = [False] * n

for i in range(n):
    opt = map(int, input().spilt())
    if opt[0] != -1:
        nodes[i].left = nodes[opt[0]]
        has_parent[opt[0]] = True
    if opt[1] != -1:
        nodes[i].right = nodes[opt[1]]
        has_parent[opt[1]] = True
# 这里完成了树的建立

root = has_parent.index(False) # 对于一棵树而言, root可以被方便的确定
```



##### list法&dict法

```python
ans, l, r = 1, [-1], [-1]
# 这里其实就是 列表索引对应
def dfs(n, count):
    global ans, l, r
    if l[n] != -1:
        dfs(l[n], count + 1)
    if r[n] != -1:
        dfs(r[n], count + 1)
    ans = max(ans, count)

n = int(input())
for i in range(n):
    a, b = map(int, input().split())
    l.append(a)
    r.append(b)
dfs(1, 1)
print(ans)
```

```python
# 字典存储value和children，模拟TreeNode的存储结构
def parse_tree(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():  # 如果是字母，创建新节点
            node = {'value': char, 'children': []}
            if stack:  # 如果栈不为空，把节点作为子节点加入到栈顶节点的子节点列表中
                stack[-1]['children'].append(node)
        elif char == '(':  # 遇到左括号，当前节点可能会有子节点
            if node:
                stack.append(node)  # 把当前节点推入栈中
                node = None
        elif char == ')':  # 遇到右括号，子节点列表结束
            if stack:
                node = stack.pop()  # 弹出当前节点
    return node  # 根节点
```

#### 2. 高度&深度

$$
高度+1 = 深度
$$

对二叉树或多叉树而言有

```python
def tree_depth(node):
    if node is None:
        return 0
    
    left_depth = tree_depth(node.left)
    right_depth = tree_depth(node.right)
    
    return max(left_depth, right_depth) + 1 # 可以被修改为多叉树
```



#### 3. 数叶子数量

```python
def count_leaves(node):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)
```



#### 4. node之间的关系

##### 二叉树的性质

1)第$i$层最个多$2^i$个结点
2)高为h的二叉树结点总数最多$2^{h+1}-1$
3)结点数为$n$的树，边的数目为$n-1$
4)$n$个结点的非空二叉树至少有$[log(n+1)]$层结点，即高度至少为 $\left \lceil log₂(n+1) \right \rceil - 1$
5)在任意一棵二叉树中，若叶子结点的个数为$n_0$，度为2的结点个数为$n_2$，则
$$
n_0 - n_2 = TreeNumber(1\ here)
$$
6)非空满二叉树叶结点数目等于分支结点数目加1。

>   ​	在满二叉树中，我们把有子节点的节点称为分支节点。每个分支节点都会产生两个新的叶节点。  
>   ​	但是，当我们添加一个新的分支节点时，原来的一个叶节点会变成分支节点。所以，实际上只增加了一个叶节点。 



1)完全二叉树中的1度结点数目为0个或1个

2)有n个结点的完全二叉树有$\left \lfloor (n+1)/2 \right \rfloor$个叶结点。

3)有n个叶结点的完全二叉树有2n或2n-1个结点(两种都可以构建)

4)有n个结点的非空完全二叉树的高度为$\left \lceil log_{2}{(n+1)} \right \rceil - 1$



#### 5. 有关临接表&visited表

临接表内存储了遍历的方向

无向图中
$$
A-B \Rightarrow A\rightarrow B \ \&\ B \rightarrow A
$$
visited表保证了BFS和DFS遍历的线性性



*visited还可用于判断是否成环*



#### 6. 遍历问题

前序（Pre Order)

```python
def pre_Order(root):
    if root is None:
		return []
    
    return root.value + pre_Order(root.left) + pre_Order(root.right)
```

中序（Mid Order）

```python
def mid_Order(root):
    if root is None:
		return []
    
    return mid_Order(root.left) + root.value + mid_Order(root.right)
```

后序（Post Order）

```python
def post_Order(root):
    if root is None:
		return []
    
    return post_Order(root.left) + post_Order(root.right) + root.value
```

层级遍历（Level Order Traversal）# 利用BFS（deque）

```python
import deque

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

def level_Order(root):
    queue = deque()
    queue.append(root)
    
    while (len(queue) != 0): # 注意这里是一个特殊的BFS,以层为单位
        
        n = len(queue)
        
        while (n > 0): #一层层的输出结果
            point = queue.popleft()
            print(point.value, end=" ") # 这里的输出是一行
            
            queue.extend(point.children)
            n -= 1
            
        print()
        
```

#### 7. 🌲的转化

$$
括号嵌套树 \Rightarrow 正常的多叉树
$$

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []
        
def build_Tree(string):
    node = None
    stack = [] # 及时处理
    for chr in string:
    	if chr.isalpha(): # 这个是一个判断函数，多见于buffer
            node = TreeNode(chr)
            if stack:
                stack[-1].children.append(node)
        elif chr == "(":
            stack.append(node)
            node = None # 及时更新
        elif chr == ")":
            node = stack.pop() # 最后返回树根
        else:
            continue
    return node
# stack在这里的运用非常符合栈的定义和特征

def preorder(root):
    output = [root.val]
    for i in root.children: # 这里的输出不一样，因为孩子不止一个
        output.extend(preorder(i))
    return "".join(output)
```

$$
括号嵌套树 \Leftarrow 正常的多叉树
$$

```python
def convert_to_bracket_tree(node):
    # 两个终止条件
    if not node:
        return ""
    if not node.children:
        return node.val
    
    result = node.val + "("
    for i, child in enumerate(node.children):
        result += convert_to_bracket_tree(child)
        if i != len(node.children) - 1:
            result += "," # 核心是“，”的加入，这里选择在一层结束前加入
    result += ")"
    
    return result
```

$$
文件转化树
$$

```python
class Dir:
    def __init__(self, file_name):
        self.name = file_name
        self.dirs = []
        self.files = []

    def show(self, dir_name, layers = 0): # 这里把layer作为遍历的“线”
        layer = layers
        result = ["|     " * layer + dir_name.name]
        dir_name.files.sort()

        for dir in dir_name.dirs:
            result.extend(self.show(dir, layer + 1))
        for file in dir_name.files:
            result.extend(["|     " * layer + file]) # extend(str)会把字符串拆开
        return result


n = 0
while True:
    n += 1
    stack = [Dir("ROOT")] # 这的输入比较难，其实也是采用的是栈的思路——及时处理，及时退出
    while (s := input()) != "*":
        if s == "#":
            exit()
        if s[0] == "f":
            stack[-1].files.append(s)
        elif s[0] == "d":
            stack.append(Dir(s))
            stack[-2].dirs.append(stack[-1])
        else:
            stack.pop()

    print(f"DATA SET {n}:")
    print(*stack[0].show(stack[0]), sep="\n") # result是个列表，存储字符串
    print() # 分割线
```

$$
建立起表达式树，按层次遍历表达式树的结果前后颠倒就得到队列表达式
$$

#### 8.🌲的分类

##### 1. 解析树 ParseTree & 抽象语法树，AST

重点如下：
	❏ 如何根据完全括号表达式==构建==解析树；
	❏ 如何==计算解析==树中的表达式；
	❏ 如何将解析树==还原==成最初的数学表达式。

构建解析树的第一步是将表达式字符串拆分成标记列表。需要考虑4种标记：==左括号、右括号、运算符和操作数==。

>   左括号代表新表达式的起点，所以应该创建一棵对应该表达式的新树。
>
>   反之，遇到右括号则意味着到达该表达式的终点。
>
>   我们也知道，操作数既是叶子节点，也是其运算符的子节点。
>
>   此外，每个运算符都有左右子节点。

有了上述信息，便可以定义以下4条规则：

>   (1) 如果当前标记是(，就为当前节点添加一个左子节点，并下沉至该子节点；
>   (2) 如果当前标记在列表`['+', '-', '/', '＊']`中，就将当前节点的值设为当前	标记对应的运算符；为当前节点添加一个右子节点，并下沉至该子节点；
>   (3) 如果当前标记是数字，就将当前节点的值设为这个数并返回至父节点；
>   (4) 如果当前标记是)，就跳到当前节点的父节点。

###### 建立

```python
class BinaryTree:
    def __init__(self, rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self, newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:  # 已经存在左子节点。此时，插入一个节点，并将已有的左子节点降一层。
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self, newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t

    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self, obj):
        self.key = obj

    def getRootVal(self):
        return self.key

    def traversal(self, method="preorder"):
        if method == "preorder":
            print(self.key, end=" ")
        if self.leftChild != None:
            self.leftChild.traversal(method)
        if method == "inorder":
            print(self.key, end=" ")
        if self.rightChild != None:
            self.rightChild.traversal(method)
        if method == "postorder":
            print(self.key, end=" ")


def buildParseTree(fpexp):
    fplist = fpexp.split()
    pStack = [] # 其实就是stack
    eTree = BinaryTree('')
    pStack.push(eTree)
    currentTree = eTree

    for i in fplist:
        if i == '(': 
        # (1) 如果当前标记是(，就为当前节点添加一个左子节点，并下沉至该子节点；
            currentTree.insertLeft('') # 未命名的树结点
            pStack.append(currentTree) # 压回parent结点
            currentTree = currentTree.getLeftChild() # 指针传到左结点
        elif i not in '+-*/)':
        # (3) 如果当前标记是数字，就将当前节点的值设为这个数并返回至父节点；
            currentTree.setRootVal(int(i)) # 命名叶子结点的值
            parent = pStack.pop() # 回到运算符
            currentTree = parent # 指针回溯
        elif i in '+-*/':
        # (2) 如果当前标记在列表`['+', '-', '/', '＊']`中，就将当前节点的值设为当前	标记对应的运算符；为当前节点添加一个右子节点，并下沉至该子节点；
            currentTree.setRootVal(i) # 命名结点的运算符
            currentTree.insertRight('') # 开右子结点的空间
            pStack.append(currentTree) # 压回parent
            currentTree = currentTree.getRightChild() # 指针传到右结点
        elif i == ')':
        # (4) 如果当前标记是)，就跳到当前节点的父节点。
            currentTree = pStack.pop() # 该结点处理结束，弹出
        else:
            raise ValueError("Unknown Operator: " + i)
    return eTree
```

###### 计算器

```python
import operator

def evaluate(parseTree):
    opers = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv}

    leftC = parseTree.getLeftChild()
    rightC = parseTree.getRightChild()

    if leftC and rightC:
        fn = opers[parseTree.getRootVal()]
        return fn(evaluate(leftC),evaluate(rightC))
    else:
        return parseTree.getRootVal()

print(evaluate(pt))
```



```python
#代码清单6-14 后序求值
def postordereval(tree):
    opers = {'+':operator.add, '-':operator.sub,
             '*':operator.mul, '/':operator.truediv}
    res1 = None
    res2 = None
    if tree:
        res1 = postordereval(tree.getLeftChild())
        res2 = postordereval(tree.getRightChild())
        if res1 and res2:
            return opers[tree.getRootVal()](res1,res2)
        else:
            return tree.getRootVal()

print(postordereval(pt))
# 30

#代码清单6-16 中序还原完全括号表达式
def printexp(tree):
    sVal = ""
    if tree:
        sVal = '(' + printexp(tree.getLeftChild())
        sVal = sVal + str(tree.getRootVal())
        sVal = sVal + printexp(tree.getRightChild()) + ')'
    return sVal

print(printexp(pt))
```



###### 20576: printExp（逆波兰表达式建树）

http://cs101.openjudge.cn/dsapre/20576/

输出中缀表达式（去除不必要的括号）

**输入**

一个字串

**输出**

一个字串

样例输入

```
( not ( True or False ) ) and ( False or True and True )
```

样例输出

```
not ( True or False ) and ( False or True and True )
```



这三个操作符：`not`：优先级最高，`and`：其次，`or`：优先级最低。

```python
class BinaryTree:
    def __init__(self, root, left=None, right=None):
        self.root = root
        self.leftChild = left
        self.rightChild = right

    def getrightchild(self):
        return self.rightChild

    def getleftchild(self):
        return self.leftChild

    def getroot(self):
        return self.root

def postorder(string):    #中缀改后缀 Shunting yard algorightm
    opStack = []
    postList = []
    inList = string.split()
    prec = { '(': 0, 'or': 1,'and': 2,'not': 3}

    for word in inList:
        if word == '(':
            opStack.append(word)
        elif word == ')':
            topWord = opStack.pop()
            while topWord != '(':
                postList.append(topWord)
                topWord = opStack.pop()
        elif word == 'True' or word == 'False':
            postList.append(word)
        else:
            while opStack and prec[word] <= prec[opStack[-1]]:
                postList.append(opStack.pop())
            opStack.append(word)
    while opStack:
        postList.append(opStack.pop())
    return postList

def buildParseTree(infix):       #以后缀表达式为基础建树
    postList = postorder(infix)
    stack = []
    for word in postList:
        if word == 'not':  
            newTree = BinaryTree(word)
            newTree.leftChild = stack.pop()
            stack.append(newTree)
        elif word == 'True' or word == 'False':
            stack.append(BinaryTree(word))
        else:
            right = stack.pop()
            left = stack.pop()
            newTree = BinaryTree(word)
            newTree.leftChild = left
            newTree.rightChild = right
            stack.append(newTree)
    currentTree = stack[-1]
    return currentTree

# 表达可以不看
def printTree(parsetree: BinaryTree):
    if parsetree.getroot() == 'or':
        return printTree(parsetree.getleftchild()) + ['or'] + printTree(parsetree.getrightchild())
    elif parsetree.getroot() == 'not':
        return ['not'] + (['('] + printTree(parsetree.getleftchild()) + [')'] if parsetree.leftChild.getroot() not in ['True', 'False'] else printTree(parsetree.getleftchild()))
    elif parsetree.getroot() == 'and':
        leftpart = ['('] + printTree(parsetree.getleftchild()) + [')'] if parsetree.leftChild.getroot() == 'or' else printTree(parsetree.getleftchild())
        rightpart = ['('] + printTree(parsetree.getrightchild()) + [')'] if parsetree.rightChild.getroot() == 'or' else printTree(parsetree.getrightchild())
        return leftpart + ['and'] + rightpart
    else:
        return [str(parsetree.getroot())]

def main():
    infix = input()
    Tree = buildParseTree(infix)
    print(' '.join(printTree(Tree)))

main()
```



`printTree`函数是一个递归函数，接收一个`BinaryTree`对象作为参数，然后根据树的结构和节点的值生成一个字符串列表。

函数的工作方式如下：

1. 首先，检查树的根节点的值。根据值的不同，函数会执行不同的操作。

2. 如果根节点的值为"or"，函数会递归地调用自身来处理左子树和右子树，然后将结果合并，并在两个结果之间插入"or"。

3. 如果根节点的值为"not"，函数会递归地调用自身来处理左子树。如果左子树的根节点的值不是"True"或"False"，则会在左子树的结果周围添加括号。

4. 如果根节点的值为"and"，函数会递归地调用自身来处理左子树和右子树。如果左子树或右子树的根节点的值为"or"，则会在相应子树的结果周围添加括号。

5. 如果根节点的值为"True"或"False"，函数会直接返回一个包含该值的列表。

6. 最后，函数会将生成的字符串列表合并为一个字符串，并返回。



###### 27637: 括号嵌套二叉树

http://cs101.openjudge.cn/practice/27637/

可以用括号嵌套的方式来表示一棵二叉树。

方法如下：`*`表示空的二叉树。

如果一棵二叉树只有一个结点，则该树就用一个非`*`字符表示，代表其根结点。

如果一棵二叉左右子树都非空，则用`树根(左子树,右子树)`的形式表示。树根是一个非`*`字符，左右子树之间用逗号隔开，没有空格。左右子树都用括号嵌套法表示。

如果左子树非空而右子树为空，则用`树根(左子树,*)`形式表示；如果左子树为空而右子树非空，则用`树根(*,右子树)`形式表示。

给出一棵树的括号嵌套表示形式，请输出其前序遍历序列、中序遍历序列、后序遍历序列。例如，`A(B(*,C),D(E))`表示的二叉树如图所示

![img](http://media.openjudge.cn/images/upload/1636/1707558029.jpg)

**输入**

第一行是整数n表示有n棵二叉树(n<100) 接下来有n行，每行是1棵二叉树的括号嵌套表示形式

**输出**

对每棵二叉树，输出其前序遍历序列和中序遍历序列

样例输入

```
2
A
A(B(*,C),D(E))
```

样例输出

```
A
A
ABCDE
BCAED
```

来源

http://dsbpython.openjudge.cn/dspythonbook/P0680/
$$
括号树问题变种
$$

```python
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def build_Tree(string):
    node = None
    stack = []  # 及时处理
    for chr in string:
        if chr.isalpha() or chr == "*":  # 这个是一个判断函数，多见于buffer
            node = TreeNode(chr)
            if chr == "*":
                node = False
            if stack:
                if stack[-1].left is None:
                    stack[-1].left = node
                    continue
                if stack[-1].right is None:
                    stack[-1].right = node
                    if stack[-1].left is False:
                        stack[-1].left = None
                    if stack[-1].right is False:
                        stack[-1].right = None
        elif chr == "(":
            stack.append(node)
            node = None  # 及时更新
        elif chr == ")":
            node = stack.pop()  # 最后返回树根
        else:
            continue
    return node


def pre_Order(root):
    if root is None:
        return []

    return [root.value] + pre_Order(root.left) + pre_Order(root.right)


def mid_Order(root):
    if root is None:
        return []

    return mid_Order(root.left) + [root.value] + mid_Order(root.right)


n = int(input())
for i in range(n):
    string = input()
    root = build_Tree(string)
    print("".join(pre_Order(root)))
    print("".join(mid_Order(root)))
```

##### 2. Huffman 算法

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20240309111247017.png" alt="image-20240309111247017" style="zoom: 50%;" />
$$
可以看到每一个Huffman真正存储的数据都在叶子结点
$$
每次都小的组合

Huffman的顶端可以用heapq法很轻松的求得。
$$
这里也给出了带权外部路径长度的计算方法
$$

```python
# 同样的 以depth作为递归深度的线
def external_path_length(node, depth=0):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return depth * node.freq
    return (external_path_length(node.left, depth + 1) +
            external_path_length(node.right, depth + 1))
```

要构建一个最优的哈夫曼编码树，首先需要对给定的字符及其权值进行排序。然后，通过重复合并权值最小的两个节点（或子树），直到所有节点都合并为一棵树为止。

下面是用 Python 实现的代码：

```python
import heapq

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(char_freq):
    heap = [Node(char, freq) for char, freq in char_freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq) # note: 合并之后 char 字典是空
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

# 同样的 以depth作为递归深度的线
def external_path_length(node, depth=0):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return depth * node.freq
    return (external_path_length(node.left, depth + 1) +
            external_path_length(node.right, depth + 1))

def main():
    char_freq = {'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 8, 'f': 9, 'g': 11, 'h': 12}
    huffman_tree = huffman_encoding(char_freq)
    external_length = external_path_length(huffman_tree)
    print("The weighted external path length of the Huffman tree is:", external_length)

if __name__ == "__main__":
    main()

# Output:
# The weighted external path length of the Huffman tree is: 169 
```



###### 编程题目

22161: 哈夫曼编码树

http://cs101.openjudge.cn/practice/22161/

根据字符使用频率(权值)生成一棵唯一的哈夫曼编码树。生成树时需要遵循以下规则以确保唯一性：

选取最小的两个节点合并时，节点比大小的规则是:

1) 权值小的节点算小。权值相同的两个节点，字符集里最小字符小的，算小。

例如 （{'c','k'},12) 和 ({'b','z'},12)，后者小。

2) 合并两个节点时，小的节点必须作为左子节点
3) 连接左子节点的边代表0,连接右子节点的边代表1

然后对输入的串进行编码或解码



**输入**

第一行是整数n，表示字符集有n个字符。
接下来n行，每行是一个字符及其使用频率（权重）。字符都是英文字母。
再接下来是若干行，有的是字母串，有的是01编码串。

**输出**

对输入中的字母串，输出该字符串的编码
对输入中的01串,将其解码，输出原始字符串

样例输入

```
3
g 4
d 8
c 10
dc
110
```

样例输出

```
110
dc
```

提示: 数据规模很小，不用在乎效率

来源: 郭炜



建树：主要利用最小堆，每次取出weight最小的两个节点，weight相加后创建节点，连接左右孩子，再入堆，直至堆中只剩一个节点.

编码：跟踪每一步走的是左还是右，用0和1表示，直至遇到有char值的节点，说明到了叶子节点，将01字串添加进字典.

解码：根据01字串决定走左还是右，直至遇到有char值的节点，将char值取出.

```python
import heapq

class Node:
    def __init__(self, weight, char=None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None

    def __lt__(self, other):
        if self.weight == other.weight:
            return self.char < other.char
        return self.weight < other.weight

def build_huffman_tree(characters):
    heap = []
    for char, weight in characters.items():
        heapq.heappush(heap, Node(weight, char))

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        #merged = Node(left.weight + right.weight) #note: 合并后，char 字段默认值是空
        merged = Node(left.weight + right.weight, min(left.char, right.char))
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

#以下把char 和密码对应上了
def encode_huffman_tree(root):
    codes = {}

    def traverse(node, code):
        #if node.char:
        if node.left is None and node.right is None:
            codes[node.char] = code
        else:
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')

    traverse(root, '')
    return codes

def huffman_encoding(codes, string):
    encoded = ''
    for char in string:
        encoded += codes[char]
    return encoded

# 找到第一个字母为止
def huffman_decoding(root, encoded_string):
    decoded = ''
    node = root
    for bit in encoded_string:
        if bit == '0':
            node = node.left
        else:
            node = node.right

        #if node.char:
        if node.left is None and node.right is None:
            decoded += node.char
            node = root
    return decoded

# 读取输入
n = int(input())
characters = {}
for _ in range(n):
    char, weight = input().split()
    characters[char] = int(weight)

#string = input().strip()
#encoded_string = input().strip()

# 构建哈夫曼编码树
huffman_tree = build_huffman_tree(characters)

# 编码和解码
codes = encode_huffman_tree(huffman_tree)

strings = []
while True:
    try:
        line = input()
        strings.append(line)

    except EOFError:
        break

results = []
#print(strings)
for string in strings:
    if string[0] in ('0','1'):
        results.append(huffman_decoding(huffman_tree, string))
    else:
        results.append(huffman_encoding(codes, string))

for result in results:
    print(result)
```

##### 3. BinHeap

```python
class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0

    def percUp(self, i):
        while i // 2 > 0:
            if self.heapList[i] < self.heapList[i // 2]:
                self.heapList[i // 2], self.heapList[i] = self.heapList[i], self.heapList[i // 2]
            i = i // 2

    def insert(self, k):
        self.heapList.append(k)
        self.currentSize = self.currentSize + 1
        self.percUp(self.currentSize)

    def percDown(self, i):
        while (i * 2) <= self.currentSize:
            mc = self.minChild(i)
            if self.heapList[i] > self.heapList[mc]:
                self.heapList[i], self.heapList[mc] = self.heapList[mc], self.heapList[i]
            i = mc


    def minChild(self, i):
        if i * 2 + 1 > self.currentSize:
            return i * 2
        else:
            if self.heapList[i * 2] < self.heapList[i * 2 + 1]:
                return i * 2
            else:
                return i * 2 + 1


    def delMin(self):
        retval = self.heapList[1]
        self.heapList[1] = self.heapList[self.currentSize]
        self.currentSize -= 1
        self.heapList.pop()
        self.percDown(1)

        return retval


    def buildHeap(self, alist):
        i = len(alist) // 2  # 超过中点的节点都是叶子节点
        self.currentSize = len(alist)
        self.heapList = [0] + alist[:]
        while (i > 0):  # 这里挺暴力的，但只洗了一半
            self.percDown(i)
        i -= 1


    def show(self):
        for _ in range(self.currentSize):
            print(self.delMin(), end=" ")


n = int(input())
a = BinHeap()
for i in range(n):
    opt = list(map(int, input().split()))
    if opt[0] == 1:
        a.insert(opt[1])
    else:
        print(a.delMin())
        
```

Binheap是天生的层级遍历，因为heapList就是层级。

##### 4. 二叉搜索树（Binary Search Tree, BST)

二叉搜索树（Binary Search Tree，BST），它是映射的另一种实现。我们感兴趣的不是元素在树中的确切位置，而是如何利用二叉树结构提供高效的搜索。

二叉搜索树依赖于这样一个性质：小于父节点的键都在左子树中，大于父节点的键则都在右子树中。我们称这个性质为二叉搜索性。



###### 22275: 二叉搜索树的遍历

http://cs101.openjudge.cn/practice/22275/

给出一棵二叉搜索树的前序遍历，求它的后序遍历

**输入**

第一行一个正整数n（n<=2000）表示这棵二叉搜索树的结点个数
第二行n个正整数，表示这棵二叉搜索树的前序遍历
保证第二行的n个正整数中，1~n的每个值刚好出现一次

**输出**

一行n个正整数，表示这棵二叉搜索树的后序遍历

```python
class TreeNode:
    def __init__(self, value):
        self.val = value
        self.left = None
        self.right = None


def insert(root, value):
    if root is None:
        return TreeNode(value)
    elif value > root.val:
        root.right = insert(root.right, value)  # 这里是一个递归计算
    else:
        root.left = insert(root.left, value)
    return root


def postOrder(root):
    if root is None:
        return []
    return postOrder(root.left) + postOrder(root.right) + [root.val]

def inorder_traversal(root, result):
    if root:
        inorder_traversal(root.left, result)
        result.append(root.val)
        inorder_traversal(root.right, result)
        
N = int(input())
arr = list(map(int, input().split()))
root = None
for value in arr:
    root = insert(root, value)
print(*postOrder(root))
```



##### 5. 平衡二叉搜索树（AVL 平衡树）

当二叉搜索树不平衡时，get和put等操作的性能可能降到O(n)。本节将介绍一种特殊的二叉搜索树，它能自动维持平衡。这种树叫作 AVL树，以其发明者G. M. Adelson-Velskii和E. M. Landis的姓氏命名。

AVL树实现映射抽象数据类型的方式与普通的二叉搜索树一样，唯一的差别就是性能。实现AVL树时，要记录每个节点的平衡因子。我们通过查看每个节点左右子树的高度来实现这一点。更正式地说，我们将平衡因子定义为左右子树的高度之差。

$$
balance Factor = height (left SubTree) - height(right SubTree)
$$
根据上述定义，如果平衡因子大于零，我们称之为左倾；如果平衡因子小于零，就是右倾；如果平衡因子等于零，那么树就是完全平衡的。为了实现AVL树并利用平衡树的优势，我们将平衡因子为-1、0和1的树都定义为**平衡树**。一旦某个节点的平衡因子超出这个范围，我们就需要通过一个过程让树恢复平衡。图1展示了一棵右倾树及其中每个节点的平衡因子。



设`N(h)`表示高度为`h`的AVL树的最少节点数，那么有如下递归关系：
$$
N(h) = N(h-1) + N(h-2) + 1
$$

$$
\begin{split}F_0 = 0 \\
F_1 = 1 \\
F_i = F_{i-1} + F_{i-2}  \text{ for all } i \ge 2\end{split}
$$



AVL 平衡🌲

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

class AVL:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self.root = self._insert(value, self.root)

    def _insert(self, value, node):
        if not node:
            return Node(value)
        elif value < node.value:
            node.left = self._insert(value, node.left)
        else:
            node.right = self._insert(value, node.right)

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

        balance = self._get_balance(node)

        if balance > 1:
            if value < node.left.value:	# 树形是 LL
                return self._rotate_right(node)
            else:	# 树形是 LR
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)

        if balance < -1:
            if value > node.right.value:	# 树形是 RR
                return self._rotate_left(node)
            else:	# 树形是 RL
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)

        return node

    def _get_height(self, node):
        if not node:
            return 0
        return node.height

    def _get_balance(self, node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x

    def preorder(self):
        return self._preorder(self.root)

    def _preorder(self, node):
        if not node:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)

n = int(input().strip())
sequence = list(map(int, input().strip().split()))

avl = AVL()
for value in sequence:
    avl.insert(value)

print(' '.join(map(str, avl.preorder())))
```

要实现从AVL树中删除节点，需要添加一个删除方法，并确保在删除节点后重新平衡树。

下面是更新后的代码，包括删除方法 `_delete`：

```python
class AVL:
    # Existing code...
    
    def delete(self, value):
        self.root = self._delete(value, self.root)

    def _delete(self, value, node):
        if not node:
            return node

        if value < node.value:
            node.left = self._delete(value, node.left)
        elif value > node.value:
            node.right = self._delete(value, node.right)
        else:
            if not node.left:
                temp = node.right
                node = None
                return temp
            elif not node.right:
                temp = node.left
                node = None
                return temp

            temp = self._min_value_node(node.right)
            node.value = temp.value
            node.right = self._delete(temp.value, node.right)

        if not node:
            return node

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

        balance = self._get_balance(node)

        # Rebalance the tree
        if balance > 1:
            if self._get_balance(node.left) >= 0:
                return self._rotate_right(node)
            else:
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)

        if balance < -1:
            if self._get_balance(node.right) <= 0:
                return self._rotate_left(node)
            else:
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)

        return node

    def _min_value_node(self, node):
        current = node
        while current.left:
            current = current.left
        return current

    # Existing code...
```

这段代码中的 `_delete` 方法用于删除节点。它首先检查树中是否存在要删除的节点，然后根据节点的左右子树情况执行相应的操作，以保持AVL树的平衡。



###### 01182: 食物链

并查集, http://cs101.openjudge.cn/practice/01182

动物王国中有三类动物A,B,C，这三类动物的食物链构成了有趣的环形。A吃B， B吃C，C吃A。
现有N个动物，以1－N编号。每个动物都是A,B,C中的一种，但是我们并不知道它到底是哪一种。
有人用两种说法对这N个动物所构成的食物链关系进行描述：
第一种说法是"1 X Y"，表示X和Y是同类。
第二种说法是"2 X Y"，表示X吃Y。
此人对N个动物，用上述两种说法，一句接一句地说出K句话，这K句话有的是真的，有的是假的。当一句话满足下列三条之一时，这句话就是假话，否则就是真话。
1） 当前的话与前面的某些真的话冲突，就是假话；
2） 当前的话中X或Y比N大，就是假话；
3） 当前的话表示X吃X，就是假话。
你的任务是根据给定的N（1 <= N <= 50,000）和K句话（0 <= K <= 100,000），输出假话的总数。

**输入**

第一行是两个整数N和K，以一个空格分隔。
以下K行每行是三个正整数 D，X，Y，两数之间用一个空格隔开，其中D表示说法的种类。
若D=1，则表示X和Y是同类。
若D=2，则表示X吃Y。

**输出**

只有一个整数，表示假话的数目。

样例输入

```
100 7
1 101 1 
2 1 2
2 2 3 
2 3 3 
1 1 3 
2 3 1 
1 5 5
```

样例输出

```
3
```

来源: Noi 01

《挑战程序设计竞赛（第2版）》的2.4.4并查集，也有讲到。

```python
# 并查集，https://zhuanlan.zhihu.com/p/93647900/
'''
我们设[0,n)区间表示同类，[n,2*n)区间表示x吃的动物，[2*n,3*n)表示吃x的动物。

如果是关系1：
　　将y和x合并。将y吃的与x吃的合并。将吃y的和吃x的合并。
如果是关系2：
　　将y和x吃的合并。将吃y的与x合并。将y吃的与吃x的合并。
原文链接：https://blog.csdn.net/qq_34594236/article/details/72587829
'''
# p = [0]*150001

def find(x):	# 并查集查询
    if p[x] == x:
        return x
    else:
        p[x] = find(p[x])	# 父节点设为根节点。目的是路径压缩。
        return p[x]

n,k = map(int, input().split())

p = [0]*(3*n + 1)
for i in range(3*n+1):	#并查集初始化
    p[i] = i

ans = 0
for _ in range(k):
    a,x,y = map(int, input().split())
    if x>n or y>n:
        ans += 1; continue
    
    if a==1:
        if find(x+n)==find(y) or find(y+n)==find(x):
            ans += 1; continue
        
        # 合并
        p[find(x)] = find(y)				
        p[find(x+n)] = find(y+n)
        p[find(x+2*n)] = find(y+2*n)
    else:
        if find(x)==find(y) or find(y+n)==find(x):
            ans += 1; continue
        p[find(x+n)] = find(y)
        p[find(y+2*n)] = find(x)
        p[find(x+2*n)] = find(y+n)

print(ans)
```


$$
既然是一颗树，那么总的节点数=总的边数+1
$$
1.填空完成下列程序：输入一棵二叉树的扩充二叉树的先根周游（前序遍历）序列，构建该二叉树，并输出它的中根周游（中序遍历）序列。这里定义一棵扩充二叉树是指将原二叉树中的所有空引用增加一个表示为@的虚拟叶结点。譬如下图所示的一棵二叉树，
输入样例：
ABD@G@@@CE@@F@@
输出样例：
DGBAECF

笔试中，对于程序阅读理解，要求还是挺高的。因为AC的代码通常有多种写法，如果考出来写的不规范代码，就有点难受。例如：上面程序，递归程序带着全局变量，难受。

较好的写法是：

```python
class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def buildTree(preorder):
    if not preorder:
        return None

    data = preorder.pop(0)
    if data == "@":
        return None

    node = TreeNode(data)
    node.left = buildTree(preorder) # 每个结尾为@@
    node.right = buildTree(preorder)

    return node

def inorderTraversal(node):
    if node is None:
        return []

    result = []
    result.extend(inorderTraversal(node.left))
    result.append(node.data)
    result.extend(inorderTraversal(node.right))

    return result

preorder = input()
tree = buildTree(list(preorder))

inorder = inorderTraversal(tree)
print(''.join(inorder))

"""
sample input:
ABD@G@@@CE@@F@@

sample output:
DGBAECF
"""
```

##### 6. 邻接表&Trie🌲

1. **邻接表**：在图论中，邻接表是一种表示图的常见方式之一。如果你使用字典（`dict`）来表示图的邻接关系，并且将每个顶点的邻居顶点存储为列表（`list`），那么就构成了邻接表。例如：

```python
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}
```

2. **字典树（前缀树，Trie）**：字典树是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。如果你使用嵌套的字典来表示字典树，其中每个字典代表一个节点，键表示路径上的字符，而值表示子节点，那么就构成了字典树。例如：

```python
trie = {
    'a': {
        'p': {
            'p': {
                'l': {
                    'e': {'is_end': True}
                }
            }
        }
    },
    'b': {
        'a': {
            'l': {
                'l': {'is_end': True}
            }
        }
    },
    'c': {
        'a': {
            't': {'is_end': True}
        }
    }
}
```

这样的表示方式使得我们可以非常高效地搜索和插入字符串，特别是在大型数据集上。

```python
# Trie implementation in Python 

class TrieNode:
	def __init__(self):
		# pointer array for child nodes of each node
		self.childNode = [None] * 26
		self.wordCount = 0
		
def insert_key(root, key):
	# Initialize the currentNode pointer with the root node
	currentNode = root

	# Iterate across the length of the string
	for c in key:
		# Check if the node exist for the current character in the Trie.
		if not currentNode.childNode[ord(c) - ord('a')]:
			# If node for current character does not exist
			# then make a new node
			newNode = TrieNode()
			# Keep the reference for the newly created node.
			currentNode.childNode[ord(c) - ord('a')] = newNode
		# Now, move the current node pointer to the newly created node.
		currentNode = currentNode.childNode[ord(c) - ord('a')]
	# Increment the wordEndCount for the last currentNode
	# pointer this implies that there is a string ending at currentNode.
	currentNode.wordCount += 1
	
def search_key(root, key):
	# Initialize the currentNode pointer with the root node
	currentNode = root

	# Iterate across the length of the string
	for c in key:
		# Check if the node exist for the current character in the Trie.
		if not currentNode.childNode[ord(c) - ord('a')]:
			# Given word does not exist in Trie
			return False
		# Move the currentNode pointer to the already existing node for current character.
		currentNode = currentNode.childNode[ord(c) - ord('a')]

	return currentNode.wordCount > 0

def delete_key(root, word):
	currentNode = root
	lastBranchNode = None
	lastBrachChar = 'a'

	for c in word:
		if not currentNode.childNode[ord(c) - ord('a')]:
			return False
		else:
			count = 0
			for i in range(26):
				if currentNode.childNode[i]:
					count += 1
			if count > 1:
				lastBranchNode = currentNode
				lastBrachChar = c
			currentNode = currentNode.childNode[ord(c) - ord('a')]

	count = 0
	for i in range(26):
		if currentNode.childNode[i]:
			count += 1

	# Case 1: The deleted word is a prefix of other words in Trie.
	if count > 0:
		currentNode.wordCount -= 1
		return True

	# Case 2: The deleted word shares a common prefix with other words in Trie.
	if lastBranchNode:
		lastBranchNode.childNode[ord(lastBrachChar) - ord('a')] = None
		return True
	# Case 3: The deleted word does not share any common prefix with other words in Trie.
	else:
		root.childNode[ord(word[0]) - ord('a')] = None
		return True
# Driver Code
if __name__ == '__main__':
	# Make a root node for the Trie
	root = TrieNode()

	# Stores the strings that we want to insert in the Trie
	input_strings = ["and", "ant", "do", "geek", "dad", "ball"]

	# number of insert operations in the Trie
	n = len(input_strings)

	for i in range(n):
		insert_key(root, input_strings[i])

	# Stores the strings that we want to search in the Trie
	search_query_strings = ["do", "geek", "bat"]

	# number of search operations in the Trie
	search_queries = len(search_query_strings)

	for i in range(search_queries):
		print("Query String:", search_query_strings[i])
		if search_key(root, search_query_strings[i]):
			# the queryString is present in the Trie
			print("The query string is present in the Trie")
		else:
			# the queryString is not present in the Trie
			print("The query string is not present in the Trie")

	# stores the strings that we want to delete from the Trie
	delete_query_strings = ["geek", "tea"]

	# number of delete operations from the Trie
	delete_queries = len(delete_query_strings)

	for i in range(delete_queries):
		print("Query String:", delete_query_strings[i])
		if delete_key(root, delete_query_strings[i]):
			# The queryString is successfully deleted from the Trie
			print("The query string is successfully deleted")
		else:
			# The query string is not present in the Trie
			print("The query string is not present in the Trie")

# This code is contributed by Vikram_Shirsat
```

## 🐰

### 邻接表的BFS和DFS运用

**Source Code: BFS in Python**

```python
graph = {'A': ['B', 'C', 'E'],
         'B': ['A','D', 'E'],
         'C': ['A', 'F', 'G'],
         'D': ['B'],
         'E': ['A', 'B','D'],
         'F': ['C'],
         'G': ['C']}
         
         
def bfs(graph, initial):
    visited = []
    queue = [initial]
 
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            neighbours = graph[node]
 
            for neighbour in neighbours:
                queue.append(neighbour)
    return visited
 
print(bfs(graph,'A'))
```

**Source Code: DFS in Python**

```python
import sys

def ret_graph():
    return {
        'A': {'B':5.5, 'C':2, 'D':6},
        'B': {'A':5.5, 'E':3},
        'C': {'A':2, 'F':2.5},
        'D': {'A':6, 'F':1.5},
        'E': {'B':3, 'J':7},
        'F': {'C':2.5, 'D':1.5, 'K':1.5, 'G':3.5},
        'G': {'F':3.5, 'I':4},
        'H': {'J':2},
        'I': {'G':4, 'J':4},
        'J': {'H':2, 'I':4},
        'K': {'F':1.5}
    }

start = 'A'                 
dest = 'J'                  
visited = []                
stack = []                  
graph = ret_graph()
path = []


stack.append(start)                  
visited.append(start)                
while stack:                         
    curr = stack.pop()            
    path.append(curr)
    for neigh in graph[curr]:        
        if neigh not in visited:       
            visited.append(neigh)       
            stack.append(neigh)         
            if neigh == dest :            
                print("FOUND:", neigh)
                print(path)
                sys.exit(0)
print("Not found")
print(path)
```

### 图的概念及表示方法

1. **图的表示**：图可以用不同的==数据结构==来表示，包括==邻接矩阵、邻接表==、~~关联矩阵~~等。这些表示方法影响着对==图进行操作和算法实现==的效率。

2. **图的遍历**：==图的遍历是指从图中的某个顶点出发，访问图中所有顶点且不重复的过程==。常见的图遍历算法包括深度优先搜索（DFS）和广度优先搜索（BFS）。

3. **最短路径**：==最短路径算法用于找出两个顶点之间的最短路径==，例如 ==Dijkstra 算法和 Floyd-Warshall== 算法。这些算法在网络路由、路径规划等领域有广泛的应用。

4. **最小生成树**：==最小生成树算法用于在一个连通加权图中找出一个权值最小的生成树==，常见的算法包括 ==Prim 算法和 Kruskal 算法==。最小生成树在网络设计、电力传输等领域有着重要的应用。

5. ~~**图的匹配**：图的匹配是指在一个图中找出一组边，使得没有两条边有一个公共顶点。匹配算法在任务分配、航线规划等问题中有着广泛的应用。~~

6. **拓扑排序**：==拓扑排序算法用于对有向无环图进行排序==，使得所有的顶点按照一定的顺序排列，并且保证图中的边的方向符合顺序关系。拓扑排序在任务调度、依赖关系分析等领域有重要的应用。

7. **图的连通性**：==图的连通性算法用于判断图中的顶点是否连通==，以及找出图中的连通分量。这对于网络分析、社交网络分析等具有重要意义。

8. ~~**图的颜色着色**：图的着色问题是指给图中的顶点赋予不同的颜色，使得相邻的顶点具有不同的颜色。这在调度问题、地图着色等方面有应用。~~



#### 图的分类与术语

1.   有向图和无向图

2.   **顶点Vertex**
     顶点又称节点，是图的基础部分。它可以有自己的名字，我们称作“键”。顶点也可以带有附加信息，我们称作“有效载荷”。

     

     **边Edge**
     边是图的另一个基础部分。两个顶点通过一条边相连，表示它们之间存在关系。边既可以是单向的，也可以是双向的。如果图中的所有边都是单向的，我们称之为有向图。图1明显是一个有向图，因为必须修完某些课程后才能修后续的课程。

     

     **度Degree**

     顶点的度是指和该顶点相连的边的条数。特别是对于有向图来说，顶点的出边条数称为该顶点的出度，顶点的入边条数称为该顶点的入度。例如图 3 的无向图中，V1的度为 2,V5的度为 4；有向图例子中，V2的出度为 1、入度为 2。

     

     **权值Weight**

     顶点和边都可以有一定属性，而量化的属性称为权值，顶点的权值和边的权值分别称为点权和边权。权值可以根据问题的实际背景设定，例如点权可以是城市中资源的数目，边权可以是两个城市之间来往所需要的时间、花费或距离。



有了上述定义之后，再来正式地定义**图Graph**。图可以用G来表示，并且G = (V, E)。其中，V是一个顶点集合，E是一个边集合。每一条边是一个二元组(v, w)，其中w, v∈V。可以向边的二元组中再添加一个元素，用于表示权重。子图s是一个由边e和顶点v构成的集合，其中e⊂E且v⊂V。

图4 展示了一个简单的带权有向图。我们可以用6个顶点和9条边的两个集合来正式地描述这个图：

$V = \left\{ V0,V1,V2,V3,V4,V5 \right\}$



$\begin{split}E = \left\{ \begin{array}{l}(v0,v1,5), (v1,v2,4), (v2,v3,9), (v3,v4,7), (v4,v0,1), \\
             (v0,v5,2),(v5,v4,8),(v3,v5,3),(v5,v2,1)
             \end{array} \right\}\end{split}$



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/digraph.png" alt="../_images/digraph.png" style="zoom: 67%;" />



<center>图4 简单的带权有向图</center>



图4中的例子还体现了其他两个重要的概念。

**路径Path**
路径是由边连接的顶点组成的序列。路径的正式定义为$w_1, w_2, ···, w_n$，其中对于所有的1≤i≤n-1，有$(w_i, w_{i+1})∈E$。无权重路径的长度是路径上的边数，有权重路径的长度是路径上的边的权重之和。以图4为例，从 V3到 V1的路径是顶点序列(V3, V4, V0, V1)，相应的边是{(v3, v4,7), (v4, v0,1), (v0, v1,5)}。

**环Cycle**
环是有向图中的一条起点和终点为同一个顶点的路径。例如，图4中的路径(V5, V2, V3, V5)就是一个环。没有环的图被称为无环图，没有环的有向图被称为有向无环图，简称为DAG。接下来会看到，DAG能帮助我们解决很多重要的问题。



#### 无向图的度

为了求解每个顶点的度，我们可以创建一个列表来存储每个顶点的度，初始值都为0。然后，对于每条边，我们将边的两个端点的度都加1。

以下是实现这个过程的Python代码：

```python
n, m = map(int, input().split())
degrees = [0] * n
for _ in range(m):
    u, v = map(int, input().split())
    degrees[u] += 1
    degrees[v] += 1

print(' '.join(map(str, degrees)))
```

这段代码首先读取输入，然后创建一个列表来存储每个顶点的度。然后，它遍历每条边，将边的两个端点的度都加1。最后，它输出每个顶点的度。



#### 有向图的度

为了求解每个顶点的入度和出度，我们可以创建两个列表来分别存储每个顶点的入度和出度，初始值都为0。然后，对于每条边，我们将起点的出度加1，终点的入度加1。

以下是实现这个过程的Python代码：

```python
n, m = map(int, input().split())
in_degrees = [0] * n
out_degrees = [0] * n
for _ in range(m):
    u, v = map(int, input().split())
    out_degrees[u] += 1
    in_degrees[v] += 1

for i in range(n):
    print(in_degrees[i], out_degrees[i])
```

这段代码首先读取输入，然后创建两个列表来存储每个顶点的入度和出度。然后，它遍历每条边，将边的起点的出度加1，终点的入度加1。最后，它输出每个顶点的入度和出度。



#### 图的表示方法


图的抽象数据类型由下列方法定义。

❏ Graph() 新建一个空图。
❏ addVertex(vert) 向图中添加一个顶点实例。
❏ addEdge(fromVert, toVert) 向图中添加一条有向边，用于连接顶点fromVert和toVert。
❏ addEdge(fromVert, toVert, weight) 向图中添加一条带权重weight的有向边，用于连接顶点fromVert和toVert。
❏ getVertex(vertKey) 在图中找到名为vertKey的顶点。
❏ getVertices() 以列表形式返回图中所有顶点。
❏ in 通过 vertex in graph 这样的语句，在顶点存在时返回True，否则返回False。

根据图的正式定义，可以通过多种方式在Python中实现图的抽象数据类型（ADT）。在使用不同的表达方式来实现图的抽象数据类型时，需要做很多取舍。有两种非常著名的图实现，它们分别是邻接矩阵 **adjacency matrix** 和邻接表**adjacency list**。本节会解释这两种实现，并且用 Python 类来实现邻接表。



**19943: 图的拉普拉斯矩阵**

http://cs101.openjudge.cn/practice/19943/

在图论中，度数矩阵是一个对角矩阵 ，其中包含的信息为的每一个顶点的度数，也就是说，每个顶点相邻的边数。邻接矩阵是图的一种常用存储方式。如果一个图一共有编号为0,1,2，…n-1的n个节点，那么邻接矩阵A的大小为n*n，对其中任一元素Aij，如果节点i，j直接有边，那么Aij=1；否则Aij=0。

将度数矩阵与邻接矩阵逐位相减，可以求得图的拉普拉斯矩阵。具体可见下图示意。

![img](http://media.openjudge.cn/images/upload/1575881364.jpg)

现给出一个图中的所有边的信息，需要你输出该图的拉普拉斯矩阵。



**输入**

第一行2个整数，代表该图的顶点数n和边数m。
接下m行，每行为空格分隔的2个整数a和b，代表顶点a和顶点b之间有一条无向边相连，a和b均为大小范围在0到n-1之间的整数。输入保证每条无向边仅出现一次（如1 2和2 1是同一条边，并不会在数据中同时出现）。

**输出**

共n行，每行为以空格分隔的n个整数，代表该图的拉普拉斯矩阵。

样例输入

```
4 5
2 1
1 3
2 3
0 1
0 2
```

样例输出

```
2 -1 -1 0
-1 3 -1 -1
-1 -1 3 -1
0 -1 -1 2
```

来源

cs101 2019 Final Exam



```python
class Vertex:	
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self, nbr):
        return self.connectedTo[nbr]

class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self, n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self, n):
        return n in self.vertList

    def addEdge(self, f, t, weight=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], weight)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())

def constructLaplacianMatrix(n, edges):
    graph = Graph()
    for i in range(n):	# 添加顶点
        graph.addVertex(i)
    
    for edge in edges:	# 添加边
        a, b = edge
        graph.addEdge(a, b)
        graph.addEdge(b, a)
    
    laplacianMatrix = []	# 构建拉普拉斯矩阵
    for vertex in graph:
        row = [0] * n
        row[vertex.getId()] = len(vertex.getConnections())
        for neighbor in vertex.getConnections():
            row[neighbor.getId()] = -1
        laplacianMatrix.append(row)

    return laplacianMatrix


n, m = map(int, input().split())	# 解析输入
edges = []
for i in range(m):
    a, b = map(int, input().split())
    edges.append((a, b))

laplacianMatrix = constructLaplacianMatrix(n, edges)	# 构建拉普拉斯矩阵

for row in laplacianMatrix:	# 输出结果
    print(' '.join(map(str, row)))
```

##### 词梯

```python
from collections import defaultdict, deque


def visit_vertex(queue, visited, other_visited, graph):
    word, path = queue.popleft()
    for i in range(len(word)):
        pattern = word[:i] + '_' + word[i + 1:]
        for next_word in graph[pattern]:
            if next_word in other_visited:
                return path + other_visited[next_word][::-1]
            if next_word not in visited:
                visited[next_word] = path + [next_word]
                queue.append((next_word, path + [next_word]))


def word_ladder(words, start, end):
    graph = defaultdict(list)
    for word in words:
        for i in range(len(word)):
            pattern = word[:i] + '_' + word[i + 1:]
            graph[pattern].append(word)

    queue_start = deque([(start, [start])])
    queue_end = deque([(end, [end])])
    visited_start = {start: [start]}
    visited_end = {end: [end]}

    while queue_start and queue_end:
        result = visit_vertex(queue_start, visited_start, visited_end, graph)
        if result:
            return ' '.join(result)
        result = visit_vertex(queue_end, visited_end, visited_start, graph)
        if result:
            return ' '.join(result[::-1])

    return 'NO'


n = int(input())
words = [input() for i in range(n)]
start, end = input().split()
print(word_ladder(words, start, end))
```

##### 骑士周游

```python
from functools import lru_cache

# initializing
size = int(input())
matrix = [[False]*size for i in range(size)]
x, y = map(int, input().split())
dir = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]


def valid(x, y):
    return 0 <= x < size and 0 <= y < size and not matrix[x][y]


def get_degree(x, y):
    count = 0
    for dx, dy in dir:
        nx, ny = x + dx, y + dy
        if valid(nx, ny):
            count += 1
    return count


@lru_cache(maxsize = 1<<30)
def dfs(x, y, count):
    if count == size**2:
        return True

    matrix[x][y] = True

    next_moves = [(dx, dy) for dx, dy in dir if valid(x + dx, y + dy)]
    next_moves.sort(key=lambda move: get_degree(x + move[0], y + move[1]))

    for dx, dy in next_moves:
        if dfs(x + dx, y + dy, count + 1):
            return True

    matrix[x][y] = False
    return False

if dfs(x, y, 1):
    print("success")
else:
    print("fail")
```

一个memory更小的存储办法

```python
import sys

class Graph:
    def __init__(self):
        self.vertices = {}
        self.num_vertices = 0

    def add_vertex(self, key):
        self.num_vertices = self.num_vertices + 1
        new_ertex = Vertex(key)
        self.vertices[key] = new_ertex
        return new_ertex

    def get_vertex(self, n):
        if n in self.vertices:
            return self.vertices[n]
        else:
            return None

    def __len__(self):
        return self.num_vertices

    def __contains__(self, n):
        return n in self.vertices

    def add_edge(self, f, t, cost=0):
        if f not in self.vertices:
            nv = self.add_vertex(f)
        if t not in self.vertices:
            nv = self.add_vertex(t)
        self.vertices[f].add_neighbor(self.vertices[t], cost)
        #self.vertices[t].add_neighbor(self.vertices[f], cost)

    def getVertices(self):
        return list(self.vertices.keys())

    def __iter__(self):
        return iter(self.vertices.values())


class Vertex:
    def __init__(self, num):
        self.key = num
        self.connectedTo = {}
        self.color = 'white'
        self.distance = sys.maxsize
        self.previous = None
        self.disc = 0
        self.fin = 0

    def __lt__(self,o):
        return self.key < o.key

    def add_neighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight


    # def setDiscovery(self, dtime):
    #     self.disc = dtime
    #
    # def setFinish(self, ftime):
    #     self.fin = ftime
    #
    # def getFinish(self):
    #     return self.fin
    #
    # def getDiscovery(self):
    #     return self.disc

    def get_neighbors(self):
        return self.connectedTo.keys()

    # def getWeight(self, nbr):
    #     return self.connectedTo[nbr]

    def __str__(self):
        return str(self.key) + ":color " + self.color + ":disc " + str(self.disc) + ":fin " + str(
            self.fin) + ":dist " + str(self.distance) + ":pred \n\t[" + str(self.previous) + "]\n"



def knight_graph(board_size):
    kt_graph = Graph()
    for row in range(board_size):           #遍历每一行
        for col in range(board_size):       #遍历行上的每一个格子
            node_id = pos_to_node_id(row, col, board_size) #把行、列号转为格子ID
            new_positions = gen_legal_moves(row, col, board_size) #按照 马走日，返回下一步可能位置
            for row2, col2 in new_positions:
                other_node_id = pos_to_node_id(row2, col2, board_size) #下一步的格子ID
                kt_graph.add_edge(node_id, other_node_id) #在骑士周游图中为两个格子加一条边
    return kt_graph

def pos_to_node_id(x, y, bdSize):
    return x * bdSize + y

def gen_legal_moves(row, col, board_size):
    new_moves = []
    move_offsets = [                        # 马走日的8种走法
        (-1, -2),  # left-down-down
        (-1, 2),  # left-up-up
        (-2, -1),  # left-left-down
        (-2, 1),  # left-left-up
        (1, -2),  # right-down-down
        (1, 2),  # right-up-up
        (2, -1),  # right-right-down
        (2, 1),  # right-right-up
    ]
    for r_off, c_off in move_offsets:
        if (                                # #检查，不能走出棋盘
            0 <= row + r_off < board_size
            and 0 <= col + c_off < board_size
        ):
            new_moves.append((row + r_off, col + c_off))
    return new_moves

# def legal_coord(row, col, board_size):
#     return 0 <= row < board_size and 0 <= col < board_size


def knight_tour(n, path, u, limit):
    u.color = "gray"
    path.append(u)              #当前顶点涂色并加入路径
    if n < limit:
        neighbors = ordered_by_avail(u) #对所有的合法移动依次深入
        #neighbors = sorted(list(u.get_neighbors()))
        i = 0

        for nbr in neighbors:
            if nbr.color == "white" and \
                knight_tour(n + 1, path, nbr, limit):   #选择“白色”未经深入的点，层次加一，递归深入
                return True
        else:                       #所有的“下一步”都试了走不通
            path.pop()              #回溯，从路径中删除当前顶点
            u.color = "white"       #当前顶点改回白色
            return False
    else:
        return True

def ordered_by_avail(n):
    res_list = []
    for v in n.get_neighbors():
        if v.color == "white":
            c = 0
            for w in v.get_neighbors():
                if w.color == "white":
                    c += 1
            res_list.append((c,v))
    res_list.sort(key = lambda x: x[0])
    return [y[1] for y in res_list]

# class DFSGraph(Graph):
#     def __init__(self):
#         super().__init__()
#         self.time = 0                   #不是物理世界，而是算法执行步数
#
#     def dfs(self):
#         for vertex in self:
#             vertex.color = "white"      #颜色初始化
#             vertex.previous = -1
#         for vertex in self:             #从每个顶点开始遍历
#             if vertex.color == "white":
#                 self.dfs_visit(vertex)  #第一次运行后还有未包括的顶点
#                                         # 则建立森林
#
#     def dfs_visit(self, start_vertex):
#         start_vertex.color = "gray"
#         self.time = self.time + 1       #记录算法的步骤
#         start_vertex.discovery_time = self.time
#         for next_vertex in start_vertex.get_neighbors():
#             if next_vertex.color == "white":
#                 next_vertex.previous = start_vertex
#                 self.dfs_visit(next_vertex)     #深度优先递归访问
#         start_vertex.color = "black"
#         self.time = self.time + 1
#         start_vertex.closing_time = self.time


def main():
    def NodeToPos(id):
       return ((id//8, id%8))

    bdSize = int(input())  # 棋盘大小
    *start_pos, = map(int, input().split())  # 起始位置
    g = knight_graph(bdSize)
    start_vertex = g.get_vertex(pos_to_node_id(start_pos[0], start_pos[1], bdSize))
    if start_vertex is None:
        print("fail")
        exit(0)

    tour_path = []
    done = knight_tour(0, tour_path, start_vertex, bdSize * bdSize-1)
    if done:
        print("success")
    else:
        print("fail")

    exit(0)

    # 打印路径
    cnt = 0
    for vertex in tour_path:
        cnt += 1
        if cnt % bdSize == 0:
            print()
        else:
            print(vertex.key, end=" ")
            #print(NodeToPos(vertex.key), end=" ")   # 打印坐标

if __name__ == '__main__':
    main()

```

##### sy384: 无向图的顶点层号

https://sunnywhy.com/sfbj/10/3/384

现有一个共n个顶点、m条边的无向连通图（假设顶点编号为从`0`到`n-1`）。我们称从s号顶点出发到达其他顶点经过的最小边数称为各顶点的层号。求图中所有顶点的层号。

**输入**

第一行三个整数n、m、s（$1 \le n \le 100,0 \le m \le \frac{n(n-1)}2, 0 \le s \le n -1$​），分别表示顶点数、边数、起始顶点编号；

接下来m行，每行两个整数u、v（$0 \le u \le n-1,0 \le v \le n-1, u \ne v$），表示一条边的起点和终点的编号。数据保证不会有重边。

**输出**

输出n个整数，分别为编号从`0`到`n-1`的顶点的层号。整数之间用空格隔开，行末不允许有多余的空格。

样例1

输入

```
6 6 0
0 1
0 3
3 5
2 4
1 5
3 4
```

输出

```
0 1 3 1 2 2
```

解释

对应的无向图和顶点层号如下图所示。

![无向图的顶点层号.png](https://raw.githubusercontent.com/GMyhf/img/main/img/a896296d-23e0-4420-9e70-c4e979a78e89.png)





需要找到从给定的起始顶点到图中所有其他顶点的最短路径长度，这也被称为顶点的层号。可以使用广度优先搜索（BFS）来解决这个问题。在BFS中，从起始节点开始，然后访问它的所有邻居，然后再访问这些邻居的邻居，依此类推。我们可以使用一个队列来跟踪待访问的节点，并使用一个距离数组来记录从起始节点到每个节点的最短距离。

以下是解决这个问题的Python代码：

```python
from collections import deque

def bfs(n, m, s, edges):
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    distance = [-1] * n
    distance[s] = 0

    queue = deque([s])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if distance[neighbor] == -1: # -1代表未访问过，也是一种Visited
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)

    return distance

# 接收数据
n, m, s = map(int, input().split())
edges = []
for _ in range(m):
    u, v = map(int, input().split())
    edges.append((u, v))

# 调用函数
distances = bfs(n, m, s, edges)
print(' '.join(map(str, distances)))
```

在这段代码中，我们首先通过`input()`函数接收用户输入的顶点数`n`、边数`m`和起始顶点`s`，然后在一个循环中接收每条边的起点和终点，并将它们添加到`edges`列表中。然后，我们调用`bfs`函数并打印结果。



#### 1. 最短路径算法：

- Dijkstra算法：用于找到两个顶点之间的最短路径。

##### 有权图Dijkstra, 无权图BFS

> Dijkstra算法与BFS（广度优先搜索）有相似之处，但它们有一些关键的区别。
>
> 1. **相似性**：
>     - Dijkstra算法和BFS都是用于图的遍历。
>     - 它们都从一个起始顶点开始，逐步扩展到邻居顶点，并以某种方式记录已经访问过的顶点。
>
> 2. **不同之处**：
>     - BFS是一种无权图的最短路径算法，它以层次遍历的方式遍历图，并找到从起始顶点到所有其他顶点的最短路径。
>     - Dijkstra算法是一种有权图的最短路径算法，它通过==贪心策略==逐步确定从起始顶点到所有其他顶点的最短路径。
>     - BFS使用队列来保存待访问的顶点，并按照顺序进行遍历。==它不考虑权重==，==只关注路径的长度==。
>     - Dijkstra算法使用优先队列（通常是最小堆）来保存待访问的顶点，并按照顶点到起始顶点的距离进行排序。它根据路径长度来决定下一个要访问的顶点，从而保证每次都是选择最短路径的顶点进行访问。
>
> 虽然Dijkstra算法的实现方式和BFS有些相似，但是它们解决的问题和具体实现细节有很大的不同。BFS适用于无权图的最短路径问题，而Dijkstra算法适用于有权图的最短路径问题。

兔子与樱花

```python
import heapq


def dijkstra(graph, start):
    distances = {node: (float('infinity'), []) for node in graph}
    distances[start] = (0, [start])
    queue = [(0, start, [start])]
    visited = set()
    while queue:
        current_distance, current_node, path = heapq.heappop(queue)

        if current_node in visited:  # 湮灭点
            continue
        visited.add(current_node)

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor][0]: # 湮灭点
                distances[neighbor] = (distance, path + [neighbor])
                heapq.heappush(queue, (distance, neighbor, path + [neighbor]))
    return distances


P = int(input())
places = {input(): i for i in range(P)}
graph = {i: {} for i in range(P)}

Q = int(input()) # Graph的建立，邻接表
for _ in range(Q):
    place1, place2, distance = input().split()
    distance = int(distance)
    graph[places[place1]][places[place2]] = distance
    graph[places[place2]][places[place1]] = distance

R = int(input())
for _ in range(R):
    start, end = input().split()
    distances = dijkstra(graph, places[start])
    path = distances[places[end]][1]
    result = ""
    for i in range(len(path) - 1):
        result += f"{list(places.keys())[list(places.values()).index(path[i])]}->({graph[path[i]][path[i + 1]]})->"
    result += list(places.keys())[list(places.values()).index(path[-1])]
    print(result)
```

- Bellman-Ford算法：用于处理带有负权边的图的最短路径问题。
- Floyd-Warshall算法：用于找到图中所有顶点之间的最短路径。

#### 2. 最小生成树算法：

Prim的算法和Kruskal的算法都用于查找连接的加权图的最小生成树（MST）。

- ##### Prim算法：用于找到连接所有顶点的最小生成树。

###### 兔子与星空

```python
import heapq

def prim(graph, start):
    mst = []
    used = set([start])  # 已经使用过的点
    edges = [
        (cost, start, to)
        for to, cost in graph[start].items()
    ]  # (cost, frm, to) 的列表
    heapq.heapify(edges)  # 转换成最小堆

    while edges:  # 当还有边可以选择时
        cost, frm, to = heapq.heappop(edges)    # 弹出最小边
        if to not in used:  # 如果这个点还没被使用过
            used.add(to)  # 标记为已使用
            mst.append((frm, to, cost))  # 加入到最小生成树中
            for to_next, cost2 in graph[to].items():  # 将与这个点相连的边加入到堆中
                if to_next not in used:  # 如果这个点还没被使用过
                    heapq.heappush(edges, (cost2, to, to_next))  # 加入到堆中

    return mst  # 返回最小生成树

n = int(input())
graph = {chr(i+65): {} for i in range(n)}
for i in range(n-1):
    data = input().split()
    node = data[0]
    for j in range(2, len(data), 2):
        graph[node][data[j]] = int(data[j+1])
        graph[data[j]][node] = int(data[j+1])

mst = prim(graph, 'A')  # 从A开始生成最小生成树
print(sum([cost for frm, to, cost in mst]))  # 输出最小生成树的总权值
```

- ##### Kruskal算法 / 并查集：用于找到连接所有顶点的最小生成树，适用于边集合已经给定的情况。

###### sy397: 最小生成树-Kruskal算法 简单

https://sunnywhy.com/sfbj/10/5/397

现有一个共n个顶点、m条边的无向图（假设顶点编号为从`0`到`n-1`），每条边有各自的边权。在图中寻找一棵树，使得这棵树包含图上所有顶点、所有边都是图上的边，且树上所有边的边权之和最小。使用Kruskal算法求出这个边权之和的最小值。

**输入**

第一行两个整数n、m（$1 \le n \le 10^4,0 \le m \le 10^5$），分别表示顶点数、边数；

接下来m行，每行三个整数u、v、w（$0 \le u \le n-1,0 \le v \le n-1, u \ne v, 1 \le w \le 100$），表示一条边的两个端点的编号及边权。数据保证不会有重边。

**输出**

输出一个整数，表示最小的边权之和。如果不存在这样的树，那么输出`-1`。

样例1

输入

```
4 5
0 1 3
0 2 2
0 3 3
2 3 1
1 2 1
```

输出

```
4
```

解释

对应的无向图如下图所示。加粗的部分即为最小生成树，其边权之和为1+1+2=4。

![最小生成树-Prim算法.png](https://raw.githubusercontent.com/GMyhf/img/main/img/202403292321529.png)





Kruskal算法是一种用于寻找最小生成树的算法。它的基本思想是按照边的权值从小到大的顺序选择边，如果这条边连接的两个顶点不在同一连通分量中，则选择这条边，否则放弃这条边。重复这个过程，直到图中所有的顶点都在同一连通分量中。

在实现Kruskal算法时，我们需要使用并查集来维护图中的连通分量，以便于快速判断两个顶点是否在同一连通分量中。

以下是Kruskal算法的Python实现：

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[px] = py
            if self.rank[px] == self.rank[py]:
                self.rank[py] += 1

def kruskal(n, edges):
    uf = UnionFind(n)
    edges.sort(key=lambda x: x[2])
    res = 0
    for u, v, w in edges:
        if uf.find(u) != uf.find(v):
            uf.union(u, v)
            res += w
    if len(set(uf.find(i) for i in range(n))) > 1:
        return -1
    return res

n, m = map(int, input().split())
edges = []
for _ in range(m):
    u, v, w = map(int, input().split())
    edges.append((u, v, w))
print(kruskal(n, edges))
```

这段代码首先定义了一个并查集类`UnionFind`，然后定义了`kruskal`函数来实现Kruskal算法。在`kruskal`函数中，我们首先创建一个并查集实例，然后按照边的权值对边进行排序。然后遍历每一条边，如果这条边连接的两个顶点不在同一连通分量中，就将这两个顶点合并到同一连通分量中，并将这条边的权值加到结果中。最后，如果图中的所有顶点都在同一连通分量中，就返回结果，否则返回-1。



可以看到，kruskal 算法的时间复杂度主要来源于对边进行排序，因此其时间复杂度是O(ElogE)，其中E为图的边数。显然 kruskal 适合顶点数较多、边数较少的情况，这和 prim算法恰好相反。于是可以根据题目所给的数据范围来选择合适的算法，即**如果是稠密图(边多)，则用 prim 算法;如果是稀疏图(边少)，则用 kruskal 算法**。

#### 3. 拓扑排序算法：

- DFS：用于对有向无环图（DAG）进行拓扑排序。
- Karn算法 / BFS ：用于对有向无环图进行拓扑排序。

```python
from collections import deque, defaultdict

def topological_sort(graph):
    indegree = defaultdict(int)
    result = []
    queue = deque()

    # 计算每个顶点的入度
    for u in graph:
        for v in graph[u]:
            indegree[v] += 1

    # 将入度为 0 的顶点加入队列
    for u in graph:
        if indegree[u] == 0:
            queue.append(u)

    # 执行拓扑排序
    while queue:
        u = queue.popleft()
        result.append(u)

        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)

    # 检查是否存在环, 那环内的元素都出不去
    if len(result) == len(graph):
        return result
    else:
        return None

# 示例调用代码，建立字典图
graph = {
    'A': ['B', 'C'],
    'B': ['C', 'D'],
    'C': ['E'],
    'D': ['F'],
    'E': ['F'],
    'F': []
}

sorted_vertices = topological_sort(graph)
if sorted_vertices:
    print("Topological sort order:", sorted_vertices)
else:
    print("The graph contains a cycle.")

# Output:
# Topological sort order: ['A', 'B', 'C', 'D', 'E', 'F']
```



#### 4. 强连通分量算法：

- Kosaraju算法 / 2 DFS：用于找到有向图中的所有强连通分量。

##### Kosaraju / 2 DFS

Kosaraju算法是一种用于在有向图中寻找强连通分量（Strongly Connected Components，SCC）的算法。它基于深度优先搜索（DFS）和图的转置操作。

Kosaraju算法的核心思想就是两次深度优先搜索（DFS）。

1. **第一次DFS**：在第一次DFS中，我们对图进行标准的深度优先搜索，但是在此过程中，我们记录下顶点完成搜索的顺序。这一步的目的是为了找出每个顶点的完成时间（即结束时间）。

2. **反向图**：接下来，我们对原图取反，即将所有的边方向反转，得到反向图。

3. **第二次DFS**：在第二次DFS中，我们按照第一步中记录的顶点完成时间的逆序，对反向图进行DFS。这样，我们将找出反向图中的强连通分量。

Kosaraju算法的关键在于第二次DFS的顺序，它保证了在DFS的过程中，我们能够优先访问到整个图中的强连通分量。因此，Kosaraju算法的时间复杂度为O(V + E)，其中V是顶点数，E是边数。

以下是Kosaraju算法的Python实现：

```python
def dfs1(graph, node, visited, stack):
    visited[node] = True
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs1(graph, neighbor, visited, stack)
    stack.append(node)

def dfs2(graph, node, visited, component):
    visited[node] = True
    component.append(node)
    for neighbor in graph[node]:
        if not visited[neighbor]:
            dfs2(graph, neighbor, visited, component)

def kosaraju(graph):
    # Step 1: Perform first DFS to get finishing times
    stack = []
    visited = [False] * len(graph)
    for node in range(len(graph)):
        if not visited[node]:
            dfs1(graph, node, visited, stack)
    
    # Step 2: Transpose the graph
    transposed_graph = [[] for _ in range(len(graph))]
    for node in range(len(graph)):
        for neighbor in graph[node]:
            transposed_graph[neighbor].append(node)
    
    # Step 3: Perform second DFS on the transposed graph to find SCCs
    visited = [False] * len(graph)
    sccs = []
    while stack:
        node = stack.pop()
        if not visited[node]:
            scc = []
            dfs2(transposed_graph, node, visited, scc)
            sccs.append(scc)
    return sccs

# Example
graph = [[1], [2, 4], [3, 5], [0, 6], [5], [4], [7], [5, 6]]
sccs = kosaraju(graph)
print("Strongly Connected Components:")
for scc in sccs:
    print(scc)

"""
Strongly Connected Components:
[0, 3, 2, 1]
[6, 7]
[5, 4]

"""
```

这段代码首先定义了两个DFS函数，分别用于第一次DFS和第二次DFS。然后，Kosaraju算法包含了三个步骤：

1. 第一次DFS：遍历整个图，记录每个节点的完成时间，并将节点按照完成时间排序后压入栈中。
2. 图的转置：将原图中的边反转，得到转置图。
3. 第二次DFS：按照栈中节点的顺序，对转置图进行DFS，从而找到强连通分量。

最后，输出找到的强连通分量。

- Tarjan算法：用于找到有向图中的所有强连通分量。
