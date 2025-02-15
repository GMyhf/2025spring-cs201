# 前言
学期前两个月压力不是很大，刷了一些leetcode的题来查漏补缺。断断续续地刷了大概一百道，选了一部分题放上来帮助和我一样基础不好的同学。
前期补的计概知识较多，动态规划以及树等新接触的知识点。后期的图论以闫老师的每日选做和作业题为主，leetcode就刷的少了，只做了一些dfs和bfs的题目。
题解有的是我写的，有的是其他大神的思路。包括链接的题解是我认为很好的伟人发布的题解，含有一些基础知识的介绍或是ppt的运用，对理解题目解法有一定帮助。
多看题解，培养快速便捷内存小的程序思路；少用复制，避免卡壳失忆不会做的尴尬瞬间。

# **一  基础数据结构（栈、链表）**

## **简单**
### 160.相交链表
https://leetcode.cn/problems/intersection-of-two-linked-lists/description/?envType=study-plan-v2&envId=top-100-liked
给你两个单链表的头节点 headA 和 headB ，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 null 。

题解：
```
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        A, B = headA, headB
        while A != B:
            A = A.next if A else headB
            B = B.next if B else headA
        return A
```
链接：https://leetcode.cn/problems/intersection-of-two-linked-lists/solutions/12624/intersection-of-two-linked-lists-shuang-zhi-zhen-l/?envType=study-plan-v2&envId=top-100-liked

### 206.反转链表
https://leetcode.cn/problems/reverse-linked-list/description/?envType=study-plan-v2&envId=top-100-liked
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。

题解：
```
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        cur, pre = head, None
        while cur:
            tmp = cur.next # 暂存后继节点 cur.next
            cur.next = pre # 修改 next 引用指向
            pre = cur      # pre 暂存 cur
            cur = tmp      # cur 访问下一节点
        return pre
```
链接:https://leetcode.cn/problems/reverse-linked-list/solutions/2361282/206-fan-zhuan-lian-biao-shuang-zhi-zhen-r1jel/?envType=study-plan-v2&envId=top-100-liked

### 234.回文链表
https://leetcode.cn/problems/palindrome-linked-list/description/?envType=study-plan-v2&envId=top-100-liked
给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true ；否则，返回 false 。

题解：
```
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        vals = []
        current_node = head
        while current_node is not None:
            vals.append(current_node.val)
            current_node = current_node.next
        return vals == vals[::-1]
```
链接：https://leetcode.cn/problems/palindrome-linked-list/solutions/457059/hui-wen-lian-biao-by-leetcode-solution/

### 141.环形链表
https://leetcode.cn/problems/linked-list-cycle/description/?envType=study-plan-v2&envId=top-100-liked
给你一个链表的头节点 head ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递 。仅仅是为了标识链表的实际情况。

题解：
```
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head  # 乌龟和兔子同时从起点出发
        while fast and fast.next:
            slow = slow.next  # 乌龟走一步
            fast = fast.next.next  # 兔子走两步
            if fast is slow:  # 兔子追上乌龟（套圈），说明有环
                return True
        return False  # 访问到了链表末尾，无环
```
链接：https://leetcode.cn/problems/linked-list-cycle/solutions/1999269/mei-xiang-ming-bai-yi-ge-shi-pin-jiang-t-c4sw/

### 21.合并两个有序链表
https://leetcode.cn/problems/merge-two-sorted-lists/description/?envType=study-plan-v2&envId=top-100-liked
将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 

题解：
```
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        cur = dum = ListNode(0)
        while list1 and list2:
            if list1.val < list2.val:
                cur.next, list1 = list1, list1.next
            else:
                cur.next, list2 = list2, list2.next
            cur = cur.next
        cur.next = list1 if list1 else list2
        return dum.next
```
链接：https://leetcode.cn/problems/merge-two-sorted-lists/solutions/2361535/21-he-bing-liang-ge-you-xu-lian-biao-shu-aisw/

### 20.有效的括号
https://leetcode.cn/problems/valid-parentheses/description/?envType=study-plan-v2&envId=top-100-liked

题解：
```
class Solution:
    def isValid(self, s: str) -> bool:
        dic = {'{': '}',  '[': ']', '(': ')', '?': '?'}
        stack = ['?']
        for c in s:
            if c in dic: stack.append(c)
            elif dic[stack.pop()] != c: return False 
        return len(stack) == 1
```
链接：https://leetcode.cn/problems/valid-parentheses/solutions/9185/valid-parentheses-fu-zhu-zhan-fa-by-jin407891080/

## **中等**

### 142.环形链表Ⅱ
https://leetcode.cn/problems/linked-list-cycle-ii/description/?envType=study-plan-v2&envId=top-100-liked
给定一个链表的头节点  head ，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。如果 pos 是 -1，则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标识链表的实际情况。

题解：
```
class Solution(object):
    def detectCycle(self, head):
        fast, slow = head, head
        while True:
            if not (fast and fast.next): return
            fast, slow = fast.next.next, slow.next
            if fast == slow: break
        fast = head
        while fast != slow:
            fast, slow = fast.next, slow.next
        return fast
```
链接：https://leetcode.cn/problems/linked-list-cycle-ii/solutions/12616/linked-list-cycle-ii-kuai-man-zhi-zhen-shuang-zhi-/

### 2.两数相加
https://leetcode.cn/problems/add-two-numbers/description/?envType=study-plan-v2&envId=top-100-liked
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

题解：
```
class Solution:
    # l1 和 l2 为当前遍历的节点，carry 为进位
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode], carry=0) -> Optional[ListNode]:
        if l1 is None and l2 is None:  # 递归边界：l1 和 l2 都是空节点
            return ListNode(carry) if carry else None  # 如果进位了，就额外创建一个节点
        if l1 is None:  # 如果 l1 是空的，那么此时 l2 一定不是空节点
            l1, l2 = l2, l1  # 交换 l1 与 l2，保证 l1 非空，从而简化代码
        carry += l1.val + (l2.val if l2 else 0)  # 节点值和进位加在一起
        l1.val = carry % 10  # 每个节点保存一个数位
        l1.next = self.addTwoNumbers(l1.next, l2.next if l2 else None, carry // 10)  # 进位
        return l1
```
链接：https://leetcode.cn/problems/add-two-numbers/solutions/2327008/dong-hua-jian-ji-xie-fa-cong-di-gui-dao-oe0di/

### 19.删除链表的倒数第N个结点
https://leetcode.cn/problems/remove-nth-node-from-end-of-list/description/?envType=study-plan-v2&envId=top-100-liked

题解：
```
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        # 由于可能会删除链表头部，用哨兵节点简化代码
        left = right = dummy = ListNode(next=head)
        for _ in range(n):
            right = right.next  # 右指针先向右走 n 步
        while right.next:
            left = left.next
            right = right.next  # 左右指针一起走
        left.next = left.next.next  # 左指针的下一个节点就是倒数第 n 个节点
        return dummy.next
```
链接：https://leetcode.cn/problems/remove-nth-node-from-end-of-list/solutions/2004057/ru-he-shan-chu-jie-dian-liu-fen-zhong-ga-xpfs/

### 24.两两交换链表中的节点
https://leetcode.cn/problems/swap-nodes-in-pairs/description/?envType=study-plan-v2&envId=top-100-liked

题解：
```
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:  # 递归边界
            return head  # 不足两个节点，无需交换

        node1 = head
        node2 = head.next
        node3 = node2.next

        node1.next = self.swapPairs(node3)  # 1 指向递归返回的链表头
        node2.next = node1  # 2 指向 1

        return node2  # 返回交换后的链表头节点
```
链接：https://leetcode.cn/problems/swap-nodes-in-pairs/solutions/2374872/tu-jie-die-dai-di-gui-yi-zhang-tu-miao-d-51ap/

### 636.函数的独占时间
https://leetcode.cn/problems/exclusive-time-of-functions/description/
有一个 单线程 CPU 正在运行一个含有 n 道函数的程序。每道函数都有一个位于  0 和 n-1 之间的唯一标识符。

函数调用 存储在一个 调用栈 上 ：当一个函数调用开始时，它的标识符将会推入栈中。而当一个函数调用结束时，它的标识符将会从栈中弹出。标识符位于栈顶的函数是 当前正在执行的函数 。每当一个函数开始或者结束时，将会记录一条日志，包括函数标识符、是开始还是结束、以及相应的时间戳。

给你一个由日志组成的列表 logs ，其中 logs[i] 表示第 i 条日志消息，该消息是一个按 "{function_id}:{"start" | "end"}:{timestamp}" 进行格式化的字符串。例如，"0:start:3" 意味着标识符为 0 的函数调用在时间戳 3 的 起始开始执行 ；而 "1:end:2" 意味着标识符为 1 的函数调用在时间戳 2 的 末尾结束执行。注意，函数可以 调用多次，可能存在递归调用 。

函数的 独占时间 定义是在这个函数在程序所有函数调用中执行时间的总和，调用其他函数花费的时间不算该函数的独占时间。例如，如果一个函数被调用两次，一次调用执行 2 单位时间，另一次调用执行 1 单位时间，那么该函数的 独占时间 为 2 + 1 = 3 。

以数组形式返回每个函数的 独占时间 ，其中第 i 个下标对应的值表示标识符 i 的函数的独占时间。

题解：
**解题思路**
由于本题是单线程CPU，一个任务进栈必然最终会由对应的该任务出栈结束。
我们只需要能在任务出栈的时候统计它和进栈的时候的时间差即可。
注意到期间可能在处理别的任务，所以我们需要排除掉其他任务的时间。
比较简单的做法是统计一个独立任务时间的总和，并在进栈的时候加入当时的总和。
那么出栈计算的时候减去中间被占用的独立时间就好了。
```
class Solution:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        def helper(log):
            idx, mark, time = log.split(":")
            return int(idx), mark == "start", int(time)

        stack, ans, total = [], [0] * n, 0
        for lg in logs:
            idx, is_start, time = helper(lg)
            if is_start:
                stack.append(total - time)
            else:
                d = stack.pop()
                diff = time + 1 + d - total
                ans[idx] += diff
                total += diff
        return ans
```
链接：https://leetcode.cn/problems/exclusive-time-of-functions/solutions/

### 394.字符串解码
https://leetcode.cn/problems/decode-string/description/?envType=study-plan-v2&envId=top-100-liked
给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为: k[encoded_string]，表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且输入的方括号总是符合格式要求的。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 k ，例如不会出现像 3a 或 2[4] 的输入。

题解：**辅助栈法**
```
class Solution:
    def decodeString(self, s: str) -> str:
        stack, res, multi = [], "", 0
        for c in s:
            if c == '[':
                stack.append([multi, res])
                res, multi = "", 0
            elif c == ']':
                cur_multi, last_res = stack.pop()
                res = last_res + cur_multi * res
            elif '0' <= c <= '9':
                multi = multi * 10 + int(c)            
            else:
                res += c
        return res
```
链接：https://leetcode.cn/problems/decode-string/solutions/19447/decode-string-fu-zhu-zhan-fa-di-gui-fa-by-jyd/

# ** 二  动态规划**

## **简单**
### LCP 07.传递信息
https://leetcode.cn/problems/chuan-di-xin-xi/

题解：
```
class Solution:
    def numWays(self, n: int, relation: List[List[int]], k: int) -> int:
        mydict=defaultdict(list)
        for x,y in relation:
            mydict[x].append(y)
        step=0
        queue=[0]
        while queue and step<k:
            for v in range(len(queue)):
                for j in mydict[queue.pop(0)]:
                    queue.append(j)
            step+=1
        return queue.count(n-1)
```

### 118.杨辉三角
https://leetcode.cn/problems/pascals-triangle/?envType=study-plan-v2&envId=top-100-liked

题解：
```
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        res = []
        if numRows == 1:
            res.append([1])
            return res

        res.append([1])
        for i in range(1, numRows):
            res.append([0] * (i + 1))
            for j in range(i + 1):
                if j == 0 or j == i:
                    res[i][j] = 1
                else:
                    a = res[i - 1][j - 1] + res[i - 1][j]
                    res[i][j] = a
        return res
```

### 70.爬楼梯
https://leetcode.cn/problems/climbing-stairs/description/?envType=study-plan-v2&envId=top-100-liked

题解：
```
class Solution:
    def climbStairs(self, n: int) -> int:
        f0=1
        f1=1
        for _ in range(2,n+1):
            new_f=f0+f1
            f0=f1
            f1=new_f
        return f1
```

### 746.使用最小花费爬楼梯
https://leetcode.cn/problems/min-cost-climbing-stairs/description/

题解：
```
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        f0=0
        f1=0
        n=len(cost)
        for i in range(1,n):
            new_f=min(f1+cost[i],f0+cost[i-1])
            f0=f1
            f1=new_f
        return f1
```

### 121.买卖股票的最佳时机
https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/

题解：
```
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        cost,profit=float('+inf'),0
        for i in prices:
            cost=min(cost,i)
            profit=max(i-cost,profit)
        return profit
```

## **中等**
### 122.买卖股票的最佳时机Ⅱ
https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/description/

题解：
```
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        tmp=0
        for i in range(len(prices)-1):
            if prices[i]<prices[i+1]:
                tmp+=prices[i+1]-prices[i]
        return tmp
```

### 198.打家劫舍
https://leetcode.cn/problems/house-robber/description/
你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

题解：
```
class Solution:
    def rob(self, nums: List[int]) -> int:
        n=len(nums)
        f0=f1=0
        for i,x in enumerate(nums):
            new_f=max(f1,f0+x)
            f0=f1
            f1=new_f
        return new_f
```

### 55.跳跃游戏
https://leetcode.cn/problems/jump-game/description/
给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。

题解：
```
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_i=0
        for i,jump in enumerate(nums):
            if i+jump>max_i and max_i>=i:
                max_i=i+jump
        return max_i>=i
```

### 45.跳跃游戏Ⅱ
https://leetcode.cn/problems/jump-game-ii/description/
给定一个长度为 n 的 0 索引整数数组 nums。初始位置为 nums[0]。

每个元素 nums[i] 表示从索引 i 向前跳转的最大长度。换句话说，如果你在 nums[i] 处，你可以跳转到任意 nums[i + j] 处:

0 <= j <= nums[i] 
i + j < n
返回到达 nums[n - 1] 的最小跳跃次数。生成的测试用例可以到达 nums[n - 1]。

题解：
```
class Solution:
    def jump(self, nums: List[int]) -> int:
        max_step=0
        end=0
        step=0
        for i in range(len(nums)-1):
            if max_step>=i:
                max_step=max(max_step,i+nums[i])
                if i==end:
                    end=max_step
                    step+=1
        return step
```

### 62.不同路径
https://leetcode.cn/problems/unique-paths/description/
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

题解：
```
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp=[[1]*m for _ in range(n)]
        for i in range(1,n):
            for j in range(1,m):
                dp[i][j]=dp[i][j-1]+dp[i-1][j]
        return dp[-1][-1]
```

## **困难**
### 42.接雨水
https://leetcode.cn/problems/trapping-rain-water/description/
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

题解：
```
class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        pre_max = [0] * n  # pre_max[i] 表示从 height[0] 到 height[i] 的最大值
        pre_max[0] = height[0]
        for i in range(1, n):
            pre_max[i] = max(pre_max[i - 1], height[i])

        suf_max = [0] * n  # suf_max[i] 表示从 height[i] 到 height[n-1] 的最大值
        suf_max[-1] = height[-1]
        for i in range(n - 2, -1, -1):
            suf_max[i] = max(suf_max[i + 1], height[i])

        ans = 0
        for h, pre, suf in zip(height, pre_max, suf_max):
            ans += min(pre, suf) - h  # 累加每个水桶能接多少水
        return ans
```
链接：https://leetcode.cn/problems/trapping-rain-water/solutions/1974340/zuo-liao-nbian-huan-bu-hui-yi-ge-shi-pin-ukwm/

### 124.二叉树中的最大路径和
https://leetcode.cn/problems/binary-tree-maximum-path-sum/description/
二叉树中的 路径 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

题解：
```
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        ans = -inf
        def dfs(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0  # 没有节点，和为 0
            l_val = dfs(node.left)  # 左子树最大链和
            r_val = dfs(node.right)  # 右子树最大链和
            nonlocal ans
            ans = max(ans, l_val + r_val + node.val)  # 两条链拼成路径
            return max(max(l_val, r_val) + node.val, 0)  # 当前子树最大链和
        dfs(root)
        return ans
```
链接：https://leetcode.cn/problems/binary-tree-maximum-path-sum/solutions/2227021/shi-pin-che-di-zhang-wo-zhi-jing-dpcong-n9s91/

# **三  树**

## **中等**
### 96.不同的二叉搜索树
https://leetcode.cn/problems/unique-binary-search-trees/description/
给你一个整数 n ，求恰由 n 个节点组成且节点值从 1 到 n 互不相同的 二叉搜索树 有多少种？返回满足题意的二叉搜索树的种数。

题解：
```
class Solution:
    def numTrees(self, n: int) -> int:
        dp=[0]*(n+1)
        dp[0],dp[1]=1,1
        for i in range(2,n+1):
            for j in range(1,i+1):
                dp[i]+=dp[j-1]*dp[i-j]
        return dp[n]
```

### 95.不同的二叉搜索树Ⅱ
https://leetcode.cn/problems/unique-binary-search-trees-ii/description/
给你一个整数 n ，请你生成并返回所有由 n 个节点组成且节点值从 1 到 n 互不相同的不同 二叉搜索树 。可以按 任意顺序 返回答案。

题解：
```
class Solution:
    def generateTrees(self, n: int) -> List[Optional[TreeNode]]:
        def dfs(l,r):
            if l>r:
                return [None]
            ans=[]
            for i in range(l,r+1):
                for j in dfs(l,i-1):
                    for x in dfs(i+1,r):
                        root=TreeNode(i)
                        root.left,root.right=j,x
                        ans.append(root)
            return ans
        return dfs(1,n)
```

### 102.二叉树的层序遍历
https://leetcode.cn/problems/binary-tree-level-order-traversal/description/
给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。

题解：
```
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        res,queue=[],collections.deque()
        queue.append(root)
        while queue:
            tmp=[]
            for _ in range(len(queue)):
                node=queue.popleft()
                tmp.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(tmp)
        return res
```

### 103.二叉树的锯齿形层序遍历
https://leetcode.cn/problems/binary-tree-zigzag-level-order-traversal/description/
给你二叉树的根节点 root ，返回其节点值的 锯齿形层序遍历 。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。

题解：
```
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        res, deque = [], collections.deque()
        deque.append(root)
        while deque:
            tmp = []
            for _ in range(len(deque)):
                node = deque.popleft()
                tmp.append(node.val)
                if node.left: deque.append(node.left)
                if node.right: deque.append(node.right)
            res.append(list(tmp))
            tmp=[]
            if not root:
                break
            for _ in range(len(deque)):
                node=deque.pop()
                tmp.append(node.val)
                if node.right:deque.appendleft(node.right)
                if node.left:deque.appendleft(node.left)
            if tmp:res.append(tmp)
        return res
```

### 114.二叉树展开为链表
https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/description/
给你二叉树的根结点 root ，请你将它展开为一个单链表：

展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。

题解：
```
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if root is None:
            return
        right=root.right
        self.flatten(root.left)
        self.flatten(root.right)
        root.left,root.right=None,root.left
        node=root
        while node.right:
            node=node.right
        node.right=right
```

### 230.二叉搜索树中第k小的元素
https://leetcode.cn/problems/kth-smallest-element-in-a-bst/description/
给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）。

题解：
```
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        if not root:
            return
        node=root
        def inorder(node):
            if node is None:
                return
            stack=[]
            res=[]
            while node or stack:
                if node:
                    stack.append(node)
                    node=node.left
                else:
                    node=stack.pop()
                    res.append(node.val)
                    node=node.right
            return res
        res=inorder(root)
        return res[k-1]
```

### 236.二叉树的最近公共祖先
https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/

题解：
```
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root==p or root==q:
            return root
        left=self.lowestCommonAncestor(root.left,p,q)
        right=self.lowestCommonAncestor(root.right,p,q)
        if not left:return right
        if not right:return left
        return root      
```

### 98.验证二叉搜索树
https://leetcode.cn/problems/validate-binary-search-tree/description/

题解：
```
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        if root is None:
            return True
        def mid(root):
            if root is None:
                return []
            res=[]
            stack=[]
            node=root
            while True:
                if node is not None:
                    stack.append(node)
                    node=node.left
                elif stack:
                    node=stack.pop()
                    res.append(node.val)
                    node=node.right
                else:
                    break
            return res
        a=mid(root)
        return a==sorted(set(a))
```

### 99.恢复二叉搜索树
https://leetcode.cn/problems/recover-binary-search-tree/description/

题解：
```
class Solution:
    def recoverTree(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        def mid(root,lst):
            if not root:
                return
            mid(root.left,lst)
            lst.append(root)
            mid(root.right,lst)
        list=[]
        mid(root,list)
        a,b=None,None
        for i in range(len(list)-1):
            if list[i].val>list[i+1].val:
                b=list[i+1]
                if a is None:
                    a=list[i]
                else:
                    break
        t=a.val
        a.val=b.val
        b.val=t
```

## **困难**
### 124.二叉树中的最大路径和
https://leetcode.cn/problems/binary-tree-maximum-path-sum/description/
二叉树中的 路径 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。

路径和 是路径中各节点值的总和。

给你一个二叉树的根节点 root ，返回其 最大路径和 。

题解：
```
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        ans = -inf
        def dfs(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0  # 没有节点，和为 0
            l_val = dfs(node.left)  # 左子树最大链和
            r_val = dfs(node.right)  # 右子树最大链和
            nonlocal ans
            ans = max(ans, l_val + r_val + node.val)  # 两条链拼成路径
            return max(max(l_val, r_val) + node.val, 0)  # 当前子树最大链和
        dfs(root)
        return ans
```
链接：https://leetcode.cn/problems/binary-tree-maximum-path-sum/solutions/2227021/shi-pin-che-di-zhang-wo-zhi-jing-dpcong-n9s91/

### 1766.互质树
https://leetcode.cn/problems/tree-of-coprimes/description/
给你一个 n 个节点的树（也就是一个无环连通无向图），节点编号从 0 到 n - 1 ，且恰好有 n - 1 条边，每个节点有一个值。树的 根节点 为 0 号点。

给你一个整数数组 nums 和一个二维数组 edges 来表示这棵树。nums[i] 表示第 i 个点的值，edges[j] = [uj, vj] 表示节点 uj 和节点 vj 在树中有一条边。

当 gcd(x, y) == 1 ，我们称两个数 x 和 y 是 互质的 ，其中 gcd(x, y) 是 x 和 y 的 最大公约数 。

从节点 i 到 根 最短路径上的点都是节点 i 的祖先节点。一个节点 不是 它自己的祖先节点。

请你返回一个大小为 n 的数组 ans ，其中 ans[i]是离节点 i 最近的祖先节点且满足 nums[i] 和 nums[ans[i]] 是 互质的 ，如果不存在这样的祖先节点，ans[i] 为 -1 。

题解：
```
# 预处理：coprime[i] 保存 [1, MX) 中与 i 互质的所有元素
MX = 51
coprime = [[j for j in range(1, MX) if gcd(i, j) == 1]
           for i in range(MX)]

class Solution:
    def getCoprimes(self, nums: List[int], edges: List[List[int]]) -> List[int]:
        n = len(nums)
        g = [[] for _ in range(n)]
        for x, y in edges:
            g[x].append(y)
            g[y].append(x)

        ans = [0] * n
        val_depth_id = [(-1, -1)] * MX  # 包含深度和节点编号
        def dfs(x: int, fa: int, depth: int) -> None:
            val = nums[x]  # x 的节点值
            # 计算与 val 互质的祖先节点值中，节点深度最大的节点编号
            ans[x] = max(val_depth_id[j] for j in coprime[val])[1]
            tmp = val_depth_id[val]  # 用于恢复现场
            val_depth_id[val] = (depth, x)  # 保存 val 对应的节点深度和节点编号
            for y in g[x]:
                if y != fa:
                    dfs(y, x, depth + 1)
            val_depth_id[val] = tmp  # 恢复现场
        dfs(0, -1, 0)
        return ans
```
链接：https://leetcode.cn/problems/tree-of-coprimes/solutions/2733992/dfs-zhong-ji-lu-jie-dian-zhi-de-shen-du-4v5d2/

# **四  图论算法**
## **DFS & BFS**
（大部分是我用dfs做的）
### 130.被围绕的区域
https://leetcode.cn/problems/surrounded-regions/description/
给你一个 m x n 的矩阵 board ，由若干字符 'X' 和 'O' 组成，捕获 所有 被围绕的区域：

连接：一个单元格与水平或垂直方向上相邻的单元格连接。
区域：连接所有 '0' 的单元格来形成一个区域。
围绕：如果您可以用 'X' 单元格 连接这个区域，并且区域中没有任何单元格位于 board 边缘，则该区域被 'X' 单元格围绕。
通过将输入矩阵 board 中的所有 'O' 替换为 'X' 来 捕获被围绕的区域。

题解：
```
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        dx=[-1,0,0,1]
        dy=[0,1,-1,0]
        m=len(board)
        n=len(board[0])
        def dfs(x,y):
            if x<0 or y<0 or x==m or y==n or board[x][y]!='O':
                return
            board[x][y]=''
            for i in range(4):
                dfs(x+dx[i],y+dy[i])
        for j in range(n):
            dfs(0,j)
            dfs(m-1,j)
        for i in range(m):
            dfs(i,0)
            dfs(i,n-1)
        for i in range(m):
            for j in range(n):
                if board[i][j]=='':
                    board[i][j]='O'
                else:
                    board[i][j]='X'
```

### 133.克隆图
https://leetcode.cn/problems/clone-graph/
给你无向 连通 图中一个节点的引用，请你返回该图的 深拷贝（克隆）。

题解：
```
from typing import Optional
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        lookup={}
        def dfs(node):
            if not node:
                return
            if node in lookup:
                return lookup[node]
            clone=Node(node.val,[])
            lookup[node]=clone
            for i in node.neighbors:
                clone.neighbors.append(dfs(i))
            return clone
        return dfs(node)
```

### 200.岛屿数量
https://leetcode.cn/problems/number-of-islands/description/
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

此外，你可以假设该网格的四条边均被水包围。

题解：
```
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        row=len(grid)
        hang=len(grid[0])
        ans=0
        def dfs(i,j):
            grid[i][j]='0'
            for r,h in [[0,1],[1,0],[0,-1],[-1,0]]:
                ni=i+r
                nj=j+h
                if 0<=ni < row and 0<=nj < hang and grid[ni][nj]=='1':
                    dfs(ni,nj)
        for i in range(row):
            for j in range(hang):
                if grid[i][j]=='1':
                    ans+=1
                    dfs(i,j)
        return ans
```

