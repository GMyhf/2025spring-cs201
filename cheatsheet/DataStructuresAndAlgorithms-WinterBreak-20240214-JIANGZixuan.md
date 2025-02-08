# 数据结构与算法-寒假

Updated 2112 GMT+8 Feb 14 2024

Feb 14 2024, Complied by 蒋子轩23工学院



## 链表

链表是一种基本的数据结构，广泛用于计算机科学中。它由一系列节点组成，每个节点包含数据部分和指向列表中下一个节点的指针。由于其动态的内存分配特性，链表特别适合于当元素数量经常变化的情况。下面是一些链表相关的经典算法问题：

### 1.**反转链表**：

这是最基本的链表问题之一，要求改变链表中元素的顺序，使其反向。可以通过迭代或递归来实现。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def reverseList(head: ListNode) -> ListNode:
    prev = None
    current = head
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    return prev
```

### 2.**检测环**：

此问题要求判断一个链表是否包含环。快慢指针（Floyd的循环检测算法）是解决这个问题的一个著名方法。

```python
def hasCycle(head: ListNode) -> bool:
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

3.**合并两个排序链表**：给定两个已排序的链表，要求合并它们并保持排序。这个问题可以用迭代或递归方法解决。

```python
def mergeTwoLists(l1: ListNode, l2: ListNode) -> ListNode:
    if not l1 or not l2:
        return l1 or l2
    if l1.val < l2.val:
        l1.next = mergeTwoLists(l1.next, l2)
        return l1
    else:
        l2.next = mergeTwoLists(l1, l2.next)
        return l2
```

### 4.**删除链表的倒数第N个节点**：

要求一次遍历完成删除操作。使用两个指针，第一个指针先前进N步，然后两个指针一起移动，直到第一个指针到达末尾。

```python
def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    dummy = ListNode(0)
    dummy.next = head
    first = dummy
    second = dummy
    for i in range(n+1):
        first = first.next
    while first:
        first = first.next
        second = second.next
    second.next = second.next.next
    return dummy.next
```

### 5.**找到链表的中间节点**：

使用快慢指针方法，快指针每次移动两步，慢指针每次移动一步，当快指针到达末尾时，慢指针就位于中间位置。

```python
def middleNode(head: ListNode) -> ListNode:
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

### 6.**回文链表**：

判断一个链表是否为回文。一种方法是通过找到中点，然后反转后半部分链表，之后比较前半部分和反转后的后半部分。

```python
def isPalindrome(head: ListNode) -> bool:
    fast = slow = head
    # 找到中间节点
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
    # 反转后半部分链表
    prev = None
    while slow:
        temp = slow.next
        slow.next = prev
        prev = slow
        slow = temp
    # 比较前半部分和反转后的后半部分
    left, right = head, prev
    while right:  # 只需比较后半部分
        if left.val != right.val:
            return False
        left = left.next
        right = right.next
    return True
```

### 7.**移除链表元素**：

删除链表中所有等于给定值的节点，要求一次遍历解决。

```python
def removeElements(head: ListNode, val: int) -> ListNode:
    dummy = ListNode(0)
    dummy.next = head
    current = dummy
    while current.next:
        if current.next.val == val:
            current.next = current.next.next
        else:
            current = current.next
    return dummy.next
```

### 8.**分割链表**：

根据给定值x分割链表，使得所有小于x的节点都位于大于或等于x的节点之前，保持原有元素的相对顺序。

```python
def partition(head: ListNode, x: int) -> ListNode:
    before_head = ListNode(0)
    before = before_head
    after_head = ListNode(0)
    after = after_head
    while head:
        if head.val < x:
            before.next = head
            before = before.next
        else:
            after.next = head
            after = after.next
        head = head.next
    after.next = None
    before.next = after_head.next
    return before_head.next
```

### 9.**旋转链表**：

给定一个链表，将链表每个节点向右移动k个位置，其中k是非负数。

```python
def rotateRight(head: ListNode, k: int) -> ListNode:
    if not head or not head.next or k == 0:
        return head
    # 计算链表长度并获取尾节点
    length = 1
    tail = head
    while tail.next:
        tail = tail.next
        length += 1
    # 链接尾节点与头节点，形成环
    tail.next = head
    # 找到新的尾节点：（length - k % length - 1）的位置
    new_tail = head
    for _ in range(length - k % length - 1):
        new_tail = new_tail.next
    # 新的头节点是新尾节点的下一个节点
    new_head = new_tail.next
    # 断开新尾节点与新头节点的链接
    new_tail.next = None
    return new_head
```

## 栈

### 1.**有效的括号**：

给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。有效字符串需满足：左括号必须用相同类型的右括号闭合，左括号必须以正确的顺序闭合。这个问题通常通过使用栈来解决，每次遇到开括号就将其压入栈中，遇到闭括号时检查栈顶元素是否与之匹配，如果匹配则弹出栈顶元素，最后栈空则有效，否则无效。

```python
def isValid(s: str) -> bool:
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)
    return not stack
```

### 2.**逆波兰表达式求值**（后缀表达式求值）：

给定一个逆波兰表达式，使用栈求出其结果。逆波兰表达式是一种后缀记法，其中每个运算符跟在其操作数之后，这种表达式的计算可以通过栈实现，遍历表达式，遇到数字则压入栈中，遇到运算符则从栈中弹出两个元素进行计算，计算结果再压入栈中，最后栈顶元素即为表达式的结果。

```python
def evalRPN(tokens: [str]) -> int:
    stack = []
    for token in tokens:
        if token not in ["+", "-", "*", "/"]:
            stack.append(int(token))
        else:
            right, left = stack.pop(), stack.pop()
            if token == "+":
                stack.append(left + right)
            elif token == "-":
                stack.append(left - right)
            elif token == "*":
                stack.append(left * right)
            elif token == "/":
                stack.append(int(float(left) / right))  # 注意整数除法的处理
    return stack.pop()
```

### 3.**柱状图中最大的矩形**：

给定 n 个非负整数表示柱状图中各柱子的高度，每个柱子彼此相邻，且宽度为 1，求在该柱状图中能够勾勒出来的矩形的最大面积。这个问题可以通过使用栈来解决，栈中存储柱子的索引，确保栈中的柱子高度是单调递增的，以此来计算最大面积。

```python
def largestRectangleArea(heights: [int]) -> int:
    stack = [-1]
    max_area = 0
    for i, height in enumerate(heights):
        while stack[-1] != -1 and heights[stack[-1]] > height:
            h = heights[stack.pop()]
            w = i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)
    while stack[-1] != -1:
        h = heights[stack.pop()]
        w = len(heights) - stack[-1] - 1
        max_area = max(max_area, h * w)
    return max_area
```

## 队列

### 循环队列

```python
class CircularQueue:
    def __init__(self, size):
        self.queue = [None] * size
        self.head = self.tail = -1
        self.size = size
    def enqueue(self, value):
        if (self.tail + 1) % self.size == self.head:
            print("队列已满")
            return False
        elif self.head == -1:
            self.head = 0
        self.tail = (self.tail + 1) % self.size
        self.queue[self.tail] = value
        return True
    def dequeue(self):
        if self.head == -1:
            print("队列为空")
            return False
        elif self.head == self.tail:
            self.head = self.tail = -1
        else:
            self.head = (self.head + 1) % self.size
        return True
    def display(self):
        if self.head == -1:
            print("队列为空")
        elif self.tail >= self.head:
            print("队列:", " ".join([str(self.queue[i]) for i in range(self.head, self.tail + 1)]))
        else:
            print("队列:", " ".join([str(self.queue[i]) for i in range(self.head, self.size)] + [str(self.queue[i]) for i in range(0, self.tail + 1)]))
cq = CircularQueue(5)
cq.enqueue(1)
cq.enqueue(2)
cq.enqueue(3)
cq.display()
cq.dequeue()
cq.display()
```

## 树

### 1.**二叉树遍历**：

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
def preorderTraversal(root):
    if not root:
        return []
    return [root.val] + preorderTraversal(root.left) + preorderTraversal(root.right)
def inorderTraversal(root):
    if not root:
        return []
    return inorderTraversal(root.left) + [root.val] + inorderTraversal(root.right)
def postorderTraversal(root):
    if not root:
        return []
    return postorderTraversal(root.left) + postorderTraversal(root.right) + [root.val]
def levelOrderTraversal(root):
    if not root:
        return []
    result, queue = [], [root]
    while queue:
        level = []
        for i in range(len(queue)):
            node = queue.pop(0)
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

### 2.**二叉搜索树操作**：

**查找**：在二叉搜索树中查找一个特定的值。

```python
def searchBST(root, val):
    if root is None or root.val == val:
        return root
    return searchBST(root.left, val) if val < root.val else searchBST(root.right, val)
```

**插入**：向二叉搜索树中插入一个新的值，同时保持树的有序性。

```python
def insertIntoBST(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insertIntoBST(root.left, val)
    else:
        root.right = insertIntoBST(root.right, val)
    return root
```

**删除**：从二叉搜索树中删除一个值，可能需要进行树的调整，以保持其有序性和平衡性。

```python
def deleteNode(root, key):
    if not root:
        return None
    # 如果key小于root.val，那么它应该在左子树中
    if key < root.val:
        root.left = deleteNode(root.left, key)
    # 如果key大于root.val，那么它应该在右子树中
    elif key > root.val:
        root.right = deleteNode(root.right, key)
    # 找到了要删除的节点
    else:
        # 节点是叶子节点，可以直接删除
        if not root.left and not root.right:
            root = None
        # 节点只有一个右子节点或左子节点
        elif root.right:
            root = root.right
        elif root.left:
            root = root.left
        # 节点有两个子节点，需要找到后继节点（右子树中的最小节点）来替代删除节点
        else:
            temp = findMin(root.right)
            root.val = temp.val
            root.right = deleteNode(root.right, root.val)
    return root
def findMin(node):
    while node.left:
        node = node.left
    return node
```

### 3.**树的最近公共祖先**（Lowest Common Ancestor, LCA）问题：

给定二叉树中的两个节点，找到它们的最近公共祖先，即这两个节点在树中最低的公共节点。

```python
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)
    if left and right:
        return root
    return left if left else right
```

### 4.**树的直径**：

树的直径是树上任意两个节点之间最长路径的长度。这个问题通常可以通过两次深度优先搜索（DFS）来解决。

```python
def diameterOfBinaryTree(root):
    diameter = [0]
    def depth(node):
        if not node:
            return 0
        left_depth = depth(node.left)
        right_depth = depth(node.right)
        diameter[0] = max(diameter[0], left_depth + right_depth)
        return max(left_depth, right_depth) + 1
    depth(root)
    return diameter[0]
```

### 5.**路径和问题**：

给定一个二叉树和一个目标和，确定树中是否存在从根节点到叶节点的路径，这条路径上所有节点值相加等于目标和。

```python
def hasPathSum(root, sum):
    if not root:
        return False
    if not root.left and not root.right and root.val == sum:
        return True
    sum -= root.val
    return hasPathSum(root.left, sum) or hasPathSum(root.right, sum)
```

### 6.**树的子结构**：

检查一个树是否是另一个树的子结构。这通常涉及到两个步骤：首先确定一个树的某个节点和另一个树的根节点值相同；其次，检查第一个树在该节点下的子树是否和第二个树具有相同的结构和节点值。

```python
def isSubtree(s, t):
    if not s:
        return False
    if isSameTree(s, t):
        return True
    return isSubtree(s.left, t) or isSubtree(s.right, t)
def isSameTree(s, t):
    if not s and not t:
        return True
    if not s or not t:
        return False
    if s.val != t.val:
        return False
    return isSameTree(s.left, t.left) and isSameTree(s.right, t.right)
```

## 图

### 1.**图遍历**：

**深度优先搜索（DFS）**：一种用于遍历或搜索树或图的算法。从图中的某个顶点开始，探索尽可能深的分支，直到所有的顶点被访问过为止。

```python
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)

    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited
```

**广度优先搜索（BFS）**：从图中的某个顶点开始，先访问距离开始顶点最近的顶点，再依次访问距离开始顶点更远的顶点。

```python
from collections import deque
def bfs(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(set(graph[vertex]) - visited)
    return visited
```

### 2.**最短路径问题**

**Dijkstra算法**：用于在加权图中找到一个顶点到其他所有顶点的最短路径。

Dijkstra算法的核心思想是贪心算法。从起始节点开始，逐步扩展到达图中所有其他节点的最短路径。算法维护两组节点集合：已经找到最短路径的节点集合和还没有找到最短路径的节点集合。初始时，起始节点的最短路径值设为0，其他所有节点的最短路径值设为无穷大。算法重复以下步骤直到所有节点的最短路径都被找到：

1. 从还没有找到最短路径的节点集合中选择一个与起始节点最短距离最小的节点。
2. 更新该节点相邻的节点的最短路径值：如果通过该节点到达相邻节点的路径比当前记录的路径更短，就更新这个最短路径值。
3. 将该节点移动到已经找到最短路径的节点集合中。
4. 重复上述步骤，直到所有节点的最短路径都被找到。

```py
import heapq
def dijkstra(graph, start):
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    return distances
```

### 3.**最小生成树问题**

给定一个带权的无向图，如何找到一个树形结构，使得这个树包含图中的所有顶点，并且树的所有边的权重之和最小。

**Kruskal算法**: 适合处理稀疏图,遵循贪心策略

1. **排序**：首先，将图中的所有边按照权重从小到大排序。如果权重相同，可以任意顺序。
2. **初始化**：初始化一个空的林（森林），即一个包含所有顶点但不包含边的集合。这些顶点最终会通过边连接成一个最小生成树。
3. **选择边**：按照边的权重顺序（从最小到最大）考虑每条边。对于每条边，如果加入这条边不会在林中形成环，就将其加入到林中。这一步使用了并查集数据结构来高效地检测环的存在。
4. **重复步骤3**：继续重复上述步骤，直到林中的边数等于顶点数减1，此时林变成了一个最小生成树。

预备知识：**并查集**

并查集是一种数据结构，用于处理一些不交集（Disjoint Sets）的合并及查询问题。它支持两种操作：

1. **查找（Find）**：确定某个元素属于哪个子集。这通常通过找到该元素的“根”元素来实现，根元素代表了该子集的标识。
2. **合并（Union）**：将两个子集合并成一个集合。这通常意味着将一个集合的根元素连接到另一个集合的根元素。

并查集通过数组或链表等数据结构实现，其中每个节点存储指向其父节点的引用，根节点则指向自己，表示该节点是集合的代表。

- **路径压缩（Path Compression）**：在执行查找操作时，将查找路径上的每个节点直接连接到根节点，这样可以减少后续操作的时间复杂度。
- **按秩合并（Union by Rank）**：将较小的树连接到较大的树上，可以通过维护每个树的秩（通常是树的高度）来实现。这样做可以避免形成过深的树，从而优化操作的时间复杂度。

```python
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))  # 初始化每个顶点的父节点为自己
        self.rank = [0] * size  # 初始化树的高度
    def find(self, x):
        # 寻找根节点，并进行路径压缩
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        # 合并两个集合
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1
```

```python
def kruskal(graph_edges, num_vertices):
    # graph_edges 是边的列表，每个元素是一个元组 (weight, vertex1, vertex2)
    # 按照边的权重从小到大排序
    graph_edges.sort()
    uf = UnionFind(num_vertices)  # 初始化并查集
    mst = []  # 用来存储最小生成树的边
    for weight, u, v in graph_edges:
        # 如果加入这条边不会形成环
        if uf.find(u) != uf.find(v):
            uf.union(u, v)  # 合并两个集合
            mst.append((u, v, weight))  # 加入到最小生成树中
    return mst
```

**Prim算法：**适合处理稠密图

1. **初始化**：选择一个起点顶点放入最小生成树的顶点集合中（任意选择一个顶点作为起点）。
2. **找边**：在连接最小生成树的顶点集合和图中其他顶点的所有边中，选择一条权重最小的边，并将这条边以及它连接的未在最小生成树中的顶点加入到最小生成树中。
3. **重复**：重复第2步，直到最小生成树中包含了原图的所有顶点。
4. **结束**：当所有顶点都被包含在最小生成树中时，算法结束，此时的边集合构成了最小生成树。

```python
import heapq
def prim(graph, start_vertex):
    # graph 是一个字典，键是顶点，值是一个列表，列表中的元素是(邻居, 权重)
    mst = []  # 存储最小生成树的边
    visited = set([start_vertex])  # 已访问的顶点
    edges = [(weight, start_vertex, to) for to, weight in graph[start_vertex]]  # 从起始顶点出发的边
    heapq.heapify(edges)  # 将边转换成最小堆，以便高效地获取最小边
    while edges:
        weight, frm, to = heapq.heappop(edges)  # 获取权重最小的边
        if to not in visited:
            visited.add(to)  # 标记为已访问
            mst.append((frm, to, weight))  # 加入到最小生成树中
            for next_to, next_weight in graph[to]:
                if next_to not in visited:
                    heapq.heappush(edges, (next_weight, to, next_to))  # 将与新顶点相连的边加入堆中
    return mst
```

### 4.**最大流问题**

在一个流网络中找到从源点到汇点的最大流量，同时不违反任何边的容量限制。这里的“流网络”是一个有向图，其中每条边都有一个非负容量，源点是流入量不受限制的节点，而汇点是流出量不受限制的节点。

**Ford-Fulkerson 算法**

1. **初始化**：流 *f* 的初始值设为 0。
2. **寻找增广路径**：在残存网络 *Gf* 中寻找一条从源点 *s* 到汇点 *t* 的路径 *P*。如果找不到这样的路径，算法结束。
3. **增加流量**：对于路径 *P* 上的每条边，增加尽可能多的流量，直到路径 *P* 上的某条边达到其容量上限。增加的流量是路径 *P* 上具有最小残存容量的边的容量。
4. **更新残存网络**：根据步骤3中推送的流量更新残存网络 *G**f*。
5. **重复**：返回步骤2，直到找不到增广路径为止。

```python
# 使用邻接表表示图
class Graph:
    def __init__(self, graph):
        self.graph = graph  # 原始图
        self.ROW = len(graph)
        self.col = len(graph[0])
    # 使用DFS寻找从给定源到汇的路径，并存储路径，如果找到则返回True
    def dfs(self, s, t, parent):
        visited = [False] * self.ROW
        stack = []
        stack.append(s)
        visited[s] = True
        while stack:
            u = stack.pop()
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    stack.append(ind)
                    visited[ind] = True
                    parent[ind] = u
                    if ind == t:
                        return True
        return False
    # 使用Ford-Fulkerson算法返回图中从s到t的最大流量
    def fordFulkerson(self, source, sink):
        parent = [-1] * self.ROW
        max_flow = 0  # 最初最大流量为0
        # 增加流量只要能找到增广路径
        while self.dfs(source, sink, parent):
            path_flow = float("Inf")
            s = sink
            while (s != source):
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]
            # 更新残余容量
            v = sink
            while (v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]
            max_flow += path_flow
        return max_flow
```

### 5.**图着色问题**

**贪心算法**：用于给图的顶点着色，使得相邻的顶点颜色不同，目标是使用最小数量的颜色。

```python
def graph_coloring(graph):
    result = {}
    for node in sorted(graph, key=lambda x: len(graph[x]), reverse=True):
        neighbor_colors = {result[neighbor] for neighbor in graph[node] if neighbor in result}
        color = 1
        while color in neighbor_colors:
            color += 1
        result[node] = color
    return result
```

### 6.**拓扑排序**

- 用于有向无环图（DAG），为图中的所有顶点生成一个线性序列，每个顶点出现且仅出现一次，且如果图中存在一条从顶点A到顶点B的路径，则在序列中顶点A出现在顶点B之前。

1. **计算图中所有顶点的入度**：遍历图中每一条边，计算图中每个顶点的入度值。
2. **选择入度为0的顶点**：从图中选出一个入度为0的顶点，即没有任何前驱（指向它的边）的顶点。
3. **移除顶点及其边**：将这个顶点输出（或记录下来），并从图中移除这个顶点及其所有的出边，这一操作会导致该顶点的直接后继顶点的入度减少。
4. **重复步骤2和3**：重复步骤2和3，直到图中没有顶点（图为空），或者图中不再存在入度为0的顶点（这时图中必定存在环，因为至少存在一个顶点的入度不为0）。

```python
from collections import deque
def topological_sort(graph):
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1
    queue = deque([u for u in in_degree if in_degree[u] == 0])
    visited = 0
    top_order = []
    while queue:
        u = queue.popleft()
        top_order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
        visited += 1
    if visited != len(graph):
        return []  
    return top_order
```

