# 数据结构与算法知识体系（DSA）

Updated 1125 GMT+8 May 29, 2025

2025 spring, Complied by Hongfei Yan



数据结构与算法（DSA，数算）的学习重点是树和图，及其涉及的各种核心算法。尽管栈和队列是基础的数据结构，且现代编程语言对其提供了直接支持，这使得我们可以方便地使用它们，但要真正掌握并高效利用这些结构，仍需深入理解其内部原理及适用场景。此外，其他一些复杂而强大的数据结构如并查集、前缀树等，同样值得我们去学习和探索。

一旦掌握了基本的数据结构（如数组、矩阵）和基础算法（包括递归、搜索、动态规划、贪心算法等），你会发现《数据结构与算法》这门课程变得更加容易理解。该课程主要探讨线性结构和非线性结构（如树和图）。其中：  

- **线性结构**可以视为数组概念的延伸，而数组本质上是一种隐式的链表。  
- **树**尤其是二叉树，通常使用递归来实现各种操作；并查集也是基于递归思想的应用，字典树（Trie）则可以通过嵌套字典（`dict{dict}`）来构建。  
- **图**作为矩阵概念的一种扩展，既可以通过二维数组表示矩阵来模拟，也可以用邻接表的形式（如`dict{list}`）来表示更复杂的图结构。

通过练习这些经典题目，可以帮助同步不同进度学生的学习步伐，还能有效提高解决实际问题的能力，并为进一步深入学习数据结构和算法提供必要的准备。  
例如，在《计算概论》阶段如果已经掌握了双指针技术，那么在《数据结构与算法》课程中遇到链表结构时，快慢指针的概念就会变得容易理解。同样地，掌握了递归思想后，并查集的实现也会更加直观。对于嵌套使用基本数据结构有了深入了解之后，构建字典树（Trie）以及图的表示方法（如邻接表dict{list}或邻接矩阵形式的二维数组）也将变得顺理成章。  
此外，搜索算法中的广度优先搜索（BFS）和深度优先搜索（DFS）是解决许多问题的基础。当你熟悉了BFS的应用场景后，将其应用于树结构中就成为了按层次遍历的有效工具。而一旦掌握了动态规划（DP），将这种思维方式扩展到树形结构上，即树形DP，也仅仅是遍历树的同时应用动态规划思想的过程。  
所以数算比计概简单，如果计概投入时间少，需要刷力扣热题100补足。https://leetcode.cn/studyplan/top-100-liked/  

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250207152537952.png" alt="image-20250207152537952" style="zoom: 33%;" />

<center>图1 数据结构与算法、计算概论课程内容覆盖知识点</center>



图1左侧为**数据结构**（Data Structure），右侧为**算法**（Algorithm），二者结合构成了 DSA 的核心内容。



## 一 、数据结构分类


### 1. 基础数据结构

- **栈（Stack）**：后进先出（LIFO），用于表达式求值、括号匹配等
- **队列（Queue）**：先进先出（FIFO）；包括双端队列（Deque）
- **哈希表（Hash Table / Dictionary / Map）**：常数时间插入/查询
- **链表（Linked List）**：单链表、双向链表、循环链表
- **数组（Array）**：支持随机访问
  - 前缀和 / 差分数组：用于区间求和/更新
- **字符串（String）**：特殊线性结构，支持哈希、匹配、变形等操作

---


### 2. 树结构

#### 🌳 通用树（General Tree）

- 节点、边、根、子树、父/子/兄弟节点、叶子节点
- 表示法：嵌套括号、缩进式、邻接表

#### 🌲 二叉树（Binary Tree）

- 遍历：先序、中序、后序、层序
- 常见树：
  - **解析树**（表达式计算）
  - **霍夫曼树**（最优前缀编码）
  - **优先队列**（基于堆实现）
  - **二叉搜索树（BST）**、AVL 自平衡树
  - *线段树（Segment Tree）
  - *树状数组（Fenwick Tree / Binary Indexed Tree）


---

### 3. 图结构（Graph）

- 顶点、边、路径、环、连通性、权重
- 表示方式：邻接矩阵、邻接表


---

## 二、 算法分类

### 1. 基础算法技巧

- 滑动窗口 & 双指针：子数组/子串枚举与优化
- 二分算法：查找 / 二分答案（最大化最小值 / 最小化最大值 ）
- 单调栈 & 单调队列：最近较大/较小元素、区间维护
- 辅助栈/队列：如最小栈、双栈实现队列
- 链表技巧：快慢指针


---

### 2. 经典算法思想

- 分治算法（Divide & Conquer）：快速排序、归并排序
- 动态规划（DP）：背包、树形、滚动数组优化用于压缩 DP 空间等
- 贪心算法：构造法、区间调度、反悔贪心
- 回溯与剪枝：组合、排列、子集、N 皇后、数独等
- 数学类：卡特兰数、同余等
- 表达式求值：
  - 调度场算法（Shunting Yard）
  - 逆波兰表达式（RPN）


---

### 3. 字符串算法

- KMP 算法：字符串匹配
- *Manacher 算法：最长回文子串
- 字符串哈希：双模哈希

---

### 4. 图论算法

- 图遍历：DFS / BFS（迷宫、连通块、染色等）
- 最短路径：
  - Dijkstra（贪心）
  - Bellman-Ford（可负权）
  - Floyd-Warshall（多源最短路径）
- 拓扑排序（DAG）
- 最小生成树：
  - Kruskal（并查集）、Prim（堆优化）
- 并查集（Union-Find）：路径压缩、连通性维护
- 强连通分量（SCC）：Kosaraju/ 2 DFS、*Tarjan


---

### 5. 树上算法

- LCA（最近公共祖先）：DFS + 递归（朴素法）
- 树的直径：DFS 
- 树上 DP





图2是数算内容汇总知识图谱。因为树、图的概念和相应算法较多，单独给出了树和图的知识图谱。

图3是树的知识图谱，描述了树的基本概念，到具体应用的各个方面。

图4是图的知识图谱，聚焦于图的基本概念、表示方法及其相关算法。

希望这种组织方式有助于清晰地理解各个知识点之间的关系，从而更加系统地学习数据结构与算法。



```mermaid
mindmap
  root(**DSA**)
    Structure{{**DATA STRUCTURE**}}
    	Stack and Queue
    	Hash Table
    	Linked List
    	Generic Tree
    		Adjacency List
    		Disjoint Set
    		Trie
    	Binary Tree(Binary Tree)
    		Priority Queues with Binary Heaps
    		Binary Search Tree
    		AVL Tree
    		*Segment Tree, *Binary Indexed Tree
    	Graph
      
    Algorithm{{**ALGORITHM**}}
    	BFS and DFS
    	DC(Divide & Conquer, and other sortings)
    		Quick Sort
    		Merge Sort
    	Shunting Yard
    	Parsing Tree
    	Tree Traversals
    	Huffman
    	Shortest Path(Shortest Path)
    		Dijkstra
    		*Bellman-Ford
    		Floyd-Warshall 
    	Topological Sorting
    	MST(Minimum Spanning Tree)
    		Prim
    		Kruskal
    	KMP
```

<center>图2 数算知识图谱</center>







```mermaid
mindmap
  root(Generic Tree)
    Notations{{**NOTATIONS**}}
    	Node,Edge
    	Root,Subtree
    	Parent,Children,Sibling,Leaf
    	Path: Level,Height,Depth
      
    Representation{{**REPRESENTATION**}}
      Nested Parentheses
      Node-Based
      Indented Tree
      Adjacency List
      	Disjoint Set
      	Trie
      
    Binary Tree{{**Binary Tree**}}
      Applications
      	Parse Tree
      	Tree Traversals
      	Huffman
      Priority Queues with Binary Heaps
      Binary Search Tree
      AVL Tree
      *Segment Tree,BIT
      *KD Tree
```

<center>图3 树的知识图谱</center>





```mermaid
mindmap
  Graph (Graph)
    Notation{{**NOTATIONS**}}
    	Vertex,Edge
    	Path, Weight
      
    Representation{{**REPRESENTATION**}}
      Matrix
      Adjacency List
      
    Algorithm{{**ALGORITHM**}}
    	Elementary Graph Algorithms(Elementary Graph Algorithm)
    		BFS
    		DFS
    	Shortest Path(Shortest Path)
    		Dijkstra
    		*Bellman-Ford
    		Floyd-Warshall 
    	Topological Sorting
    		Depth First Forest
    		Karn / BFS
    	*Critical Path Method
    	MST(Minimum Spanning Tree)
    		Prim
    		Kruskal
    	SCC(Strongly Connected Component)
    		Kosaraju(Kosaraju / 2 DFS)
    		*Tarjan
      

```

<center>图4 图的知识图谱</center>





我们班机考为主，`cs201数算 2025spring每日选做` 有161个题目（一半OJ题目，一半是力扣题目），https://github.com/GMyhf/2025spring-cs201/blob/main/problem_list_2025spring.md

准备分类总结；再就是结合这学期的课件；以及往年笔试题目。



## 三、课件

课件网址 https://github.com/，下面只列出考试范围内的课件，大语言模型相关课件也可以在线查阅。

20250304_week3_DSA_OOP.md	面向对象编程

20250311_week4-5_timeComplexity_linearStructure.md	时间复杂度、排序、线性结构

20250325_week6-9_tree.md	树

20250421_week10-13_graph.md	图

20250520_HashTable_KMP.md	散列表、KMP





在认真学习上面4个课件内容后，可以练习和完成每日选做题面。

problem_list_2025spring.md	每日选做题目

20250526_dsa_mindmap.md	数算知识体系（本文件）



两个主要的题解：

https://github.com/GMyhf/2024spring-cs201/blob/main/2024spring_dsa_problems.md

https://github.com/GMyhf/2024fall-cs101/blob/main/2024fall_LeetCode_problems.md



## 四、必须掌握题目

从每日选做中挑了一些重点需要掌握的题目如下。

| 问题编号与名称                   | 标签              | 难度   | 链接                                      |
| -------------------------------- | ----------------- | ------ | ----------------------------------------- |
| 04089:电话号码                   | Trie              | Medium | http://cs101.openjudge.cn/practice/04089/ |
| 20106:走山路                     | Dijkstra          | Medium | http://cs101.openjudge.cn/practice/20106/ |
| 09202: 舰队、海域出击！          | Topological Order | Medium | http://cs101.openjudge.cn/practice/09202/ |
| 05442: 兔子与星空                | MST               | Medium | http://cs101.openjudge.cn/practice/05442/ |
| 27635:判断无向图是否连通有无回路 | dfs, union-find   | Medium | http://cs101.openjudge.cn/practice/27635/ |
| 28046: 词梯                      | bfs               | Tough  | http://cs101.openjudge.cn/practice/28046/ |
| 04123: 马走日                    | backtracking      | Medium | http://cs101.openjudge.cn/practice/04123  |
| 02524: 宗教信仰                  | disjoint set      | Medium | http://cs101.openjudge.cn/practice/02524/ |
| 04078: 实现堆结构                | heap              | Medium | http://cs101.openjudge.cn/practice/04078/ |
| 22158: 根据二叉树前中序序列建树  | tree              | Medium | http://cs101.openjudge.cn/practice/22158/ |
| 24750: 根据二叉树中后序序列建树  | tree              | Medium | http://cs101.openjudge.cn/practice/24750/ |
| 24591:中序表达式转后序表达式     | stack             | Tough  | http://cs101.openjudge.cn/practice/24591/ |
| 03704: 括号匹配问题              | stack             | Easy   | http://cs101.openjudge.cn/practice/03704  |
| 02299: Ultra-QuickSort           | 归并排序          | Tough  | http://cs101.openjudge.cn/practice/02299/ |
| 08210:河中跳房子                 | Binary search     | Medium | http://cs101.openjudge.cn/practice/08210  |
| 27256: 当前队列中位数            | data structures   | Tough  | http://cs101.openjudge.cn/practice/27256/ |
|                                  |                   |        |                                           |
|                                  |                   |        |                                           |
|                                  |                   |        |                                           |



## 五、每日选做题目

课件网址 https://github.com/

Q：材料`problem_list_2025spring.md`是这学期课程内容覆盖到的题目，有各个题目的名称、标签、难度和链接。需要结合题解`2024spring_dsa_problems`，`2024fall_LeetCode_problems.md`来找题面和代码。 

将`problem_list_2025spring.md`中的题目，按照 “问题编号与名称“ 在`2024spring_dsa_problems`，或者`2024fall_LeetCode_problems.md` 中可以找到题目的题面，及相应的 AC代码等信息。

对于`problem_list_2025spring.md`中每个标签，遍历的所有161个条目，然后收集题号、名称、链接，给出对应题目简短1-2句的题面要求，及解题思路。



### 拓扑排序
**01094: Sorting It All Out**  http://cs101.openjudge.cn/practice/01094/

- 题面：给定n个大写字母和m个"A<B"形式的关系，判断是否能确定唯一排序序列，或发现矛盾  
- 思路：增量式拓扑排序，每次添加关系后检测入度变化，若存在多个入度0节点则序列不唯一，出现环则矛盾

**1857.有向图中最大颜色值**  https://leetcode.cn/problems/largest-color-value-in-a-directed-graph/

- 题面：在有向图中寻找路径使得节点颜色出现次数最大值最大  
- 思路：拓扑排序+动态规划，维护每个节点各颜色出现次数的最大值

---

### 树结构
**337.打家劫舍III**  https://leetcode.cn/problems/house-robber-iii/description/

- 题面：在二叉树中选择不相邻节点求最大和  
- 思路：树形DP，记录每个节点偷/不偷两种状态的最大值

**04082:树的镜面映射**  http://cs101.openjudge.cn/practice/04082/

- 题面：将给定二叉树进行水平镜像翻转后输出层序遍历  
- 思路：层序遍历时交换左右子节点顺序

---

### 单调栈
**84.柱状图中最大的矩形**  https://leetcode.cn/problems/largest-rectangle-in-histogram/

- 题面：在柱状图中找出面积最大的矩形  
- 思路：单调栈维护高度递增序列，计算以每个柱子为高度的最大宽度

---

### 图论（Dijkstra）
**03424: Candies**  http://cs101.openjudge.cn/practice/03424/

- 题面：求从起点到终点的最大糖果数路径  
- 思路：Dijkstra算法变形，维护最大权值路径

**743.网络延迟时间**  https://leetcode.cn/problems/network-delay-time/description/

- 题面：计算信号从起点传播到所有节点的最短时间中的最大值  
- 思路：标准Dijkstra算法实现

---

### 贪心算法
**01328:Radar Installation**  http://cs101.openjudge.cn/practice/01328/

- 题面：在海岸线部署雷达覆盖岛屿，求最少雷达数  
- 思路：按岛屿可覆盖区间右端点排序，贪心选择雷达位置

**781.森林中的兔子**  https://leetcode.cn/problems/rabbits-in-forest/

- 题面：根据兔子回答的同类数量计算最少兔子数  
- 思路：统计相同回答的分组，每组数量超过回答值+1时需新建组

**3362.零数组变换 III** https://leetcode.cn/problems/zero-array-transformation-iii/

- **题面**: 通过一系列操作将数组变换为零数组。
- **解题思路**: 使用<mark>堆</mark>和<mark>差分数组</mark>来实现贪心策略。

**01258: Agri-Net**http://cs101.openjudge.cn/practice/01258/

- **题面**: 使用最小生成树算法连接农场。
- **解题思路**: 使用Prim算法或Kruskal算法。

---

### 滑动窗口
**2962.统计最大元素出现至少K次的子数组**  https://leetcode.cn/problems/count-subarrays-where-max-element-appears-at-least-k-times/

- 题面：统计包含最大元素出现≥K次的子数组数量  
- 思路：滑动窗口记录最大元素出现次数，动态维护窗口有效性[citation:16]

**3556.最大质数子字符串之和**  https://leetcode.cn/problems/sum-of-largest-prime-substrings/description/

- 题面：找出字符串中连续数字组成的最大质数  
- 思路：滑动窗口生成所有子串，结合质数判断与数值比较[citation:5]

---

### 并查集
**547.省份数量**  https://leetcode.cn/problems/number-of-provinces/

- 题面：判断城市连通分量数  
- 思路：标准并查集实现，合并相连城市后统计根节点数量

**1584.连接所有点的最小费用**  https://leetcode.cn/problems/min-cost-to-connect-all-points/

- 题面：用最小成本连接所有点形成连通图  
- 思路：Kruskal算法，先构造所有边再按权值排序合并

**01611: The Suspects**

- **题面**: 找出所有与感染者直接或间接接触的人。
- **解题思路**: 使用并查集（Union-Find）。

**01703: 发现它，抓住它**

- **题面**: 判断两起案件是否是同一个犯罪团伙所为。
- **解题思路**: 使用并查集（Union-Find）。

---

### 滚动数组优化DP

01-背包，滚动数组，https://oi-wiki.org/dp/knapsack/

**23421: 小偷背包**，http://cs101.openjudge.cn/practice/23421/



```python
N, B = map(int, input().split())
values = list(map(int, input().split()))
weights = list(map(int, input().split()))

dp = [0] * (B + 1)

for i in range(N):
    prev = dp[:]  # 复制上一次的状态
    for j in range(B + 1):
        if j >= weights[i]:
            dp[j] = max(prev[j], prev[j - weights[i]] + values[i])

print(dp[B])
```



**M787.K站中转内最便宜的航班**

https://leetcode.cn/problems/cheapest-flights-within-k-stops/

在「最多经过 K 次中转」的约束下，求出从 src 到 dst 的最小费用。

```python
from typing import List

class Solution:
    def findCheapestPrice(self, 
                          n: int, 
                          flights: List[List[int]], 
                          src: int, 
                          dst: int, 
                          K: int) -> int:
        # 初始化：到各城最便宜费用
        INF = float('inf')
        dist = [INF] * n
        dist[src] = 0
        
        # 最多允许 K 次中转 -> 最多使用 K+1 条边
        for _ in range(K + 1):
            # 基于上一轮的结果创建新一轮的 dist
            prev = dist[:]  
            
            # 对每条航班边做松弛
            for u, v, w in flights:
                # 若 u 可达，则尝试用 u -> v 这条边更新 v
                if prev[u] + w < dist[v]:
                    dist[v] = prev[u] + w
            
            # 下一轮松弛时，依然要基于本轮更新后的 dist，
            # 因此不需要再额外复制
        
        return dist[dst] if dist[dst] != INF else -1
```



---

#### **一、二分查找**
1. **月度开销 (04135)**  
   - 题面：将N天开销分成M段，求各段最大值的最小可能  
   - 思路：二分查找可能的开销范围，贪心验证分段可行性  
   - 关键词：最小值最大化、贪心验证、二分框架

2. **LeetCode 1760**  
   - 题面：通过最多k次操作将数组分割成连续子数组，最小化最大子数组和  
   - 思路：二分查找可能的子数组和范围，滑动窗口验证分割次数  
   - 关键词：二分答案、滑动窗口验证

---

#### **二、图论**
1. **207. 课程表**  
   - 题面：判断课程安排是否存在循环依赖  
   - 思路：建图后拓扑排序检测环路  
   - 关键词：邻接表、入度数组、环路检测[citation:15][citation:18]

---

#### **三、树结构**
1. **236. 二叉树的最近公共祖先**  
   - 题面：找二叉树中两个节点的最低公共祖先  
   - 思路：后序遍历递归判断左右子树状态  
   - 关键词：后序遍历、状态传递[citation:10][citation:15]

2. **01145. Tree Summing**  
   - 题面：解析LISP格式二叉树，判断是否存在根到叶子的路径和等于目标值  
   - 思路：递归解析括号结构，DFS遍历路径和  
   - 关键词：LISP解析、递归DFS、路径剪枝[citation:15]

---

#### **四、动态规划**
1. **5. 最长回文子串**  
   - 题面：找字符串中的最长回文子串  
   - 思路：动态规划状态表记录子串回文性，或中心扩展法  
   - 关键词：状态转移方程、中心扩散[citation:20]

---

#### **五、回溯算法**
1. **37. 解数独**  
   - 题面：填充9×9数独的空格满足每行/列/3×3子矩阵不重复  
   - 思路：递归尝试填充数字，用位运算记录可用数字  
   - 关键词：位运算剪枝、回溯恢复状态[citation:3][citation:15]

2. **0311. 波兰表达式求值**  
   - 题面：解析逆波兰表达式并计算结果  
   - 思路：栈结构模拟运算符优先级，回溯处理嵌套表达式  
   - 关键词：栈操作、递归解析[citation:15]

---

#### **六、并查集**
1. **547. 省份数量**  
   - 题面：根据城市连接矩阵计算连通分量数  
   - 思路：并查集合并连通区域，路径压缩优化  
   - 关键词：连通分量、路径压缩[citation:3][citation:18]

2. **827. 最大人工岛**  
   - 题面：通过最多填海一个网格扩大最大岛屿面积  
   - 思路：并查集记录岛屿区域，枚举可填海位置计算潜在面积  
   - 关键词：连通块统计、枚举优化[citation:3]

---

#### **七、堆**
1. **3478. 和最大的K个元素**  
   - 题面：从数组中选择最大的k个数之和  
   - 思路：维护大小为k的最小堆，过滤小值  
   - 关键词：TopK问题、堆排序[citation:3][citation:8]

2. **剑指 Offer 40. 最小的k个数**  
   - 题面：找出数组中最小的k个数  
   - 思路：最大堆维护候选集，替换较大值  
   - 关键词：堆性质、反向筛选[citation:8]

---

#### **八、滑动窗口**
1. **2962. 统计最大元素出现至少 K 次的子数组**  
   - 题面：统计数组中满足最大元素出现≥k次的子数组数量  
   - 思路：滑动窗口维护最大值频率，双指针收缩窗口  
   - 关键词：单调队列、频率统计[citation:3][citation:8]

2. **3. 无重复字符的最长子串**  
   - 题面：找不含重复字符的最长子串长度  
   - 思路：哈希表记录字符位置，滑动窗口维护无重复区间  
   - 关键词：哈希映射、窗口收缩[citation:8]

---

### stack

- **3561.移除相邻字符** https://leetcode.cn/problems/resulting-string-after-adjacent-removals/
  - **题面**: 移除字符串中相邻的重复字符。
  - **解题思路**: 使用栈来移除相邻的重复字符。

### 前缀和

- **0521: 3355.零数组变换 I** https://leetcode.cn/problems/zero-array-transformation-i/

  - **题面**: 通过前缀和和贪心策略将数组变换为零数组。

  - **解题思路**: 使用前缀和和贪心策略。

    

### 二分查找

2070.每一个查询的最大美丽值

https://leetcode.cn/problems/most-beautiful-item-for-each-query/



### 元宝腾讯

Q：知识库 `数算课件_GMyhf` 中的 `problem_list_2025spring.md` 是这学期课程内容覆盖到的题目，对应题目，按照“问题编号与名称” 可以在 `数算计概题解_GMyhf` 中的 `2024spring_dsa_problems`，或者`2024fall_LeetCode_problems.md` 中找到原题的题面，及相应的 AC代码等信息。请按照 `problem_list_2025spring.md`的 “标签” 分类这些题目，给出对应题目简短1-2句的题面要求，及解题思路，及解法关键词等。



根据您提供的 `problem_list_2025spring.md` 文件中的题目编号与名称，我将按照“标签”分类这些题目，并给出每个题目的简短题面要求、解题思路及解法关键词。

- - 
  
    
  
- **01321: 棋盘问题**
  - **题面**: 在棋盘上摆放棋子，使得任意两个棋子不在同一行或列。
  - **解题思路**: 使用回溯算法。
  - **解法关键词**: 回溯算法
  
- **01426: Find The Multiple**
  - **题面**: 找到一个只包含0和1的数字，且是给定数的倍数。
  - **解题思路**: 使用广度优先搜索（BFS）。
  - **解法关键词**: BFS
  
- **01577: Falling Leaves**
  - **题面**: 重建二叉搜索树。
  - **解题思路**: 使用BFS重建二叉搜索树。
  - **解法关键词**: BFS, 二叉搜索树





- **01724: ROADS**
  - **题面**: 找到Bob从城市1到城市N的最短路径。
  - **解题思路**: 使用Dijkstra算法。
  - **解法关键词**: Dijkstra算法
- **01760: Disk Tree**
  - **题面**: 恢复目录结构。
  - **解题思路**: 使用前缀树（Trie）。
  - **解法关键词**: 前缀树



### MST最小生成树

- **01789: Truck History**
  - **题面**: 找到卡车类型派生的最高质量计划。
  - **解题思路**: 使用最小生成树算法。
  - **解法关键词**: 最小生成树



- **01860: Currency Exchange**
  - **题面**: 判断是否可以通过一系列货币兑换操作增加资金。
  - **解题思路**: 使用Bellman-Ford算法检测是否存在正权环。
  - **解法关键词**: Bellman-Ford算法
- **01941: The Sierpinski Fractal**
  - **题面**: 绘制Sierpinski三角形。
  - **解题思路**: 使用递归绘制Sierpinski三角形。
  - **解法关键词**: 递归, 分形
- **01944: Fiber Communications**
  - **题面**: 连接农场，使得所有农场都能通信。
  - **解题思路**: 使用贪心算法。
  - **解法关键词**: 贪心算法
- **02039: 反反复复**
  - **题面**: 还原加密信息。
  - **解题思路**: 使用模拟方法还原信息。
  - **解法关键词**: 模拟
- **02049: Finding Nemo**
  - **题面**: 计算Marlin到达Nemo的最短路径。
  - **解题思路**: 使用BFS。
  - **解法关键词**: BFS
- **02092: Grandpa is Famous**
  - **题面**: 找出比赛中出现次数第二多的选手。
  - **解题思路**: 统计每个选手的出现次数，找出第二多的。
  - **解法关键词**: 统计, 排序
- **02192: Zipper**
  - **题面**: 判断第三个字符串是否可以由前两个字符串组合而成。
  - **解题思路**: 使用递归和记忆化搜索。
  - **解法关键词**: 递归, 记忆化搜索
- **02226: Muddy Fields**
  - **题面**: 用最少的木板覆盖泥泞区域。
  - **解题思路**: 使用贪心算法和并查集。
  - **解法关键词**: 贪心算法, 并查集

### 标签: implementation

- **02701: 与7无关的数**
  - **题面**: 计算小于等于n的所有与7无关的正整数的平方和。
  - **解题思路**: 遍历所有小于等于n的正整数，检查是否与7无关，并计算平方和。
  - **解法关键词**: 遍历, 条件判断
  
- **02707: 求一元二次方程的根**
  - **题面**: 求一元二次方程的根。
  - **解题思路**: 使用公式计算根，并处理特殊情况。
  - **解法关键词**: 公式计算, 特殊情况处理
  
- 
  
- **02734: 十进制到八进制**
  - **题面**: 将十进制数转换为八进制。
  - **解题思路**: 使用栈实现转换。
  - **解法关键词**: 栈, 数制转换
  
  
  
- **02753: 菲波那契数列**
  - **题面**: 求菲波那契数列的第a个数。
  - **解题思路**: 使用递归或动态规划计算。
  - **解法关键词**: 递归, 动态规划
  
- **02767: 简单密码**
  - **题面**: 解密凯撒密码。
  - **解题思路**: 使用字符替换解密。
  - **解法关键词**: 字符替换
  
- **02783: Holiday Hotel**
  - **题面**: 找出符合条件的酒店。
  - **解题思路**: 使用排序和遍历。
  - **解法关键词**: 排序, 遍历
  
- **02786: Pell数列**
  - **题面**: 求Pell数列的第k项模32767的值。
  - **解题思路**: 使用递归和记忆化搜索。
  - **解法关键词**: 递归, 记忆化搜索
  
- **02792: 集合加法**
  - **题面**: 计算满足条件的(i, j)对的数量。
  - **解题思路**: 使用计数器统计。
  - **解法关键词**: 计数器, 统计
  
- **02804: 词典**
  - **题面**: 翻译文档。
  - **解题思路**: 使用字典进行翻译。
  - **解法关键词**: 字典, 翻译
  
- **02808: 校门外的树**
  - **题面**: 计算移走树后剩余的树的数量。
  - **解题思路**: 使用数组标记树的位置。
  - **解法关键词**: 数组, 标记
  
- **02810: 完美立方**
  - **题面**: 寻找满足条件的四元组。
  - **解题思路**: 使用暴力搜索和排序。
  - **解法关键词**: 暴力搜索, 排序
  
- **02883: Checking order**
  - **题面**: 检查数字串是否按升序排列。
  - **解题思路**: 检查并排序。
  - **解法关键词**: 检查, 排序
  
- **02899: 矩阵交换行**
  - **题面**: 交换矩阵的行。
  - **解题思路**: 检查范围并交换行。
  - **解法关键词**: 检查, 交换
  
- **02911: 受限完全平方数**
  - **题面**: 找出满足条件的四位数。
  - **解题思路**: 生成完全平方数并检查条件。
  - **解法关键词**: 生成, 检查
  
- **02913: 加密技术**
  - **题面**: 编制程序，将输入的一行字符加密解密。
  - **解题思路**: 使用字符替换和模运算进行加密和解密。
  - **解法关键词**: 字符替换, 模运算

### 标签: dp

- **02502: Subway**
  - **题面**: 计算从起点到终点的最短路径。
  - **解题思路**: 使用Dijkstra算法找到最短路径。
  - **解法关键词**: Dijkstra算法
- **03424: Candies**
  - **题面**: 计算从起点到终点的最短路径。
  - **解题思路**: 使用Dijkstra算法找到最短路径。
  - **解法关键词**: Dijkstra算法
- **01860: Currency Exchange**
  - **题面**: 判断是否可以通过一系列货币兑换操作增加资金。
  - **解题思路**: 使用Bellman-Ford算法检测是否存在正权环。
  - **解法关键词**: Bellman-Ford算法

### 标签: math

- **02733: 判断闰年**
  - **题面**: 判断某年是否是闰年。
  - **解题思路**: 根据闰年规则判断。
  - **解法关键词**: 条件判断
- **02734: 十进制到八进制**
  - **题面**: 将十进制数转换为八进制。
  - **解题思路**: 使用栈实现转换。
  - **解法关键词**: 栈, 数制转换
- **02753: 菲波那契数列**
  - **题面**: 求菲波那契数列的第a个数。
  - **解题思路**: 使用递归或动态规划计算。
  - **解法关键词**: 递归, 动态规划

### 标签: tree

- **01321: 棋盘问题**
  - **题面**: 在棋盘上摆放棋子，使得任意两个棋子不在同一行或列。
  - **解题思路**: 使用回溯算法。
  - **解法关键词**: 回溯算法
- **01577: Falling Leaves**
  - **题面**: 重建二叉搜索树。
  - **解题思路**: 使用BFS重建二叉搜索树。
  - **解法关键词**: BFS, 二叉搜索树
- **01760: Disk Tree**
  - **题面**: 恢复目录结构。
  - **解题思路**: 使用前缀树（Trie）。
  - **解法关键词**: 前缀树

### 标签: graph

- **01789: Truck History**
  - **题面**: 找到卡车类型派生的最高质量计划。
  - **解题思路**: 使用最小生成树算法。
  - **解法关键词**: 最小生成树
- **01941: The Sierpinski Fractal**
  - **题面**: 绘制Sierpinski三角形。
  - **解题思路**: 使用递归绘制Sierpinski三角形。
  - **解法关键词**: 递归, 分形
- **01944: Fiber Communications**
  - **题面**: 连接农场，使得所有农场都能通信。
  - **解题思路**: 使用贪心算法。
  - **解法关键词**: 贪心算法

### 标签: backtracking

- 01321: 棋盘问题
  - **题面**: 在棋盘上摆放棋子，使得任意两个棋子不在同一行或列。
  - **解题思路**: 使用回溯算法。
  - **解法关键词**: 回溯算法

### 标签: binary search

- 01426: Find The Multiple
  - **题面**: 找到一个只包含0和1的数字，且是给定数的倍数。
  - **解题思路**: 使用广度优先搜索（BFS）。
  - **解法关键词**: BFS

### 标签: implementation

- **02701: 与7无关的数**
  - **题面**: 计算小于等于n的所有与7无关的正整数的平方和。
  - **解题思路**: 遍历所有小于等于n的正整数，检查是否与7无关，并计算平方和。
  - **解法关键词**: 遍历, 条件判断
  
- **02707: 求一元二次方程的根**
  - **题面**: 求一元二次方程的根。
  - **解题思路**: 使用公式计算根，并处理特殊情况。
  - **解法关键词**: 公式计算, 特殊情况处理
  
  
  
- **02786: Pell数列**
  - **题面**: 求Pell数列的第k项模32767的值。
  - **解题思路**: 使用递归和记忆化搜索。
  - **解法关键词**: 递归, 记忆化搜索
  
- **02792: 集合加法**
  - **题面**: 计算满足条件的(i, j)对的数量。
  - **解题思路**: 使用计数器统计。
  - **解法关键词**: 计数器, 统计
  
- **02804: 词典**
  - **题面**: 翻译文档。
  - **解题思路**: 使用字典进行翻译。
  - **解法关键词**: 字典, 翻译
  
- **02808: 校门外的树**
  - **题面**: 计算移走树后剩余的树的数量。
  - **解题思路**: 使用数组标记树的位置。
  - **解法关键词**: 数组, 标记
  
- **02810: 完美立方**
  - **题面**: 寻找满足条件的四元组。
  - **解题思路**: 使用暴力搜索和排序。
  - **解法关键词**: 暴力搜索, 排序
  
- **02883: Checking order**
  - **题面**: 检查数字串是否按升序排列。
  - **解题思路**: 检查并排序。
  - **解法关键词**: 检查, 排序
  
- **02899: 矩阵交换行**
  - **题面**: 交换矩阵的行。
  - **解题思路**: 检查范围并交换行。
  - **解法关键词**: 检查, 交换
  
- **02911: 受限完全平方数**
  - **题面**: 找出满足条件的四位数。
  - **解题思路**: 生成完全平方数并检查条件。
  - **解法关键词**: 生成, 检查
  
- **02913: 加密技术**
  - **题面**: 编制程序，将输入的一行字符加密解密。
  - **解题思路**: 使用字符替换和模运算进行加密和解密。
  - **解法关键词**: 字符替换, 模运算

### 标签: BFS

- 01577: Falling Leaves
  - **题面**: 重建二叉搜索树。
  - **解题思路**: 使用BFS重建二叉搜索树。
  - **解法关键词**: BFS, 二叉搜索树

### 标签: DFS

- 01321: 棋盘问题
  - **题面**: 在棋盘上摆放棋子，使得任意两个棋子不在同一行或列。
  - **解题思路**: 使用回溯算法。
  - **解法关键词**: 回溯算法

### 标签: Union Find

- 01611: The Suspects
  - **题面**: 找出所有与感染者直接或间接接触的人。
  - **解题思路**: 使用并查集（Union-Find）。
  - **解法关键词**: 并查集

### 标签: Greedy

- **01724: ROADS**
  - **题面**: 找到Bob从城市1到城市N的最短路径。
  - **解题思路**: 使用Dijkstra算法。
  - **解法关键词**: Dijkstra算法
- **01944: Fiber Communications**
  - **题面**: 连接农场，使得所有农场都能通信。
  - **解题思路**: 使用贪心算法。
  - **解法关键词**: 贪心算法

### 标签: Trie

- 01760: Disk Tree
  - **题面**: 恢复目录结构。
  - **解题思路**: 使用前缀树（Trie）。
  - **解法关键词**: 前缀树

### 标签: Prim

- 01258: Agri-Net
  - **题面**: 使用最小生成树算法连接农场。
  - **解题思路**: 使用Prim算法或Kruskal算法。
  - **解法关键词**: Prim算法, Kruskal算法

### 标签: Bellman-Ford

- 01860: Currency Exchange
  - **题面**: 判断是否可以通过一系列货币兑换操作增加资金。
  - **解题思路**: 使用Bellman-Ford算法检测是否存在正权环。
  - **解法关键词**: Bellman-Ford算法

### 标签: Suffix Array

- 02913: 加密技术
  - **题面**: 编制程序，将输入的一行字符加密解密。
  - **解题思路**: 使用字符替换和模运算进行加密和解密。
  - **解法关键词**: 字符替换, 模运算



## 六、笔试题目

https://github.com/GMyhf/2025spring-cs201/blob/main/written_exam_DSA-B.md
