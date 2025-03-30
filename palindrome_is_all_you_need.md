# 回文Palindrome Is All You Need

Updated 1850 GMT+8 Mar 30 2025

2024 fall, Complied by Hongfei Yan



## 简单

### LC125.验证回文串

https://leetcode.cn/problems/valid-palindrome/

如果在将所有大写字符转换为小写字符、并移除所有非字母数字字符之后，短语正着读和反着读都一样。则可以认为该短语是一个 **回文串** 。

字母和数字都属于字母数字字符。

给你一个字符串 `s`，如果它是 **回文串** ，返回 `true` ；否则，返回 `false` 。

 

**示例 1：**

```
输入: s = "A man, a plan, a canal: Panama"
输出：true
解释："amanaplanacanalpanama" 是回文串。
```

**示例 2：**

```
输入：s = "race a car"
输出：false
解释："raceacar" 不是回文串。
```

**示例 3：**

```
输入：s = " "
输出：true
解释：在移除非字母数字字符之后，s 是一个空字符串 "" 。
由于空字符串正着反着读都一样，所以是回文串。
```

 

**提示：**

- `1 <= s.length <= 2 * 10^5`
- `s` 仅由可打印的 ASCII 字符组成



```python
class Solution:
    def isPalindrome(self, s: str) -> bool:       
        s_filtered = ''.join(c.lower() for c in s if c.isalnum())

        left, right = 0, len(s_filtered) - 1
        while left < right:
            if s_filtered[left] == s_filtered[right]:
                left += 1
                right -= 1
            else:
                return False
        
        return True
```



### LC234.回文链表

linked-list, https://leetcode.cn/problems/palindrome-linked-list/

给你一个单链表的头节点 `head` ，请你判断该链表是否为

回文链表（**回文** 序列是向前和向后读都相同的序列。如果是，返回 `true` ；否则，返回 `false` 。



 

**示例 1：**

<img src="https://assets.leetcode.com/uploads/2021/03/03/pal1linked-list.jpg" alt="img" style="zoom:67%;" />

```
输入：head = [1,2,2,1]
输出：true
```

**示例 2：**

<img src="https://assets.leetcode.com/uploads/2021/03/03/pal2linked-list.jpg" alt="img" style="zoom:67%;" />

```
输入：head = [1,2]
输出：false
```

 

**提示：**

- 链表中节点数目在范围`[1, 105]` 内
- `0 <= Node.val <= 9`

 

**进阶：**你能否用 `O(n)` 时间复杂度和 `O(1)` 空间复杂度解决此题？



快慢指针查找链表的中间节点

```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True
        
        # 1. 使用快慢指针找到链表的中点
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        # 2. 反转链表的后半部分
        prev = None
        while slow:
            next_node = slow.next
            slow.next = prev
            prev = slow
            slow = next_node
        
        # 3. 对比前半部分和反转后的后半部分
        left, right = head, prev
        while right:  # right 是反转后的链表的头
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        
        return True

```



递归算法：currentNode 指针是先到尾节点，由于递归的特性再从后往前进行比较。frontPointer 是递归函数外的指针。若 currentNode.val != frontPointer.val 则返回 false。反之，frontPointer 向前移动并返回 true。

算法的正确性在于递归处理节点的顺序是相反的，而我们在函数外又记录了一个变量，因此从本质上，我们同时在正向和逆向迭代匹配。

作者：力扣官方题解

```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:

        self.front_pointer = head

        def recursively_check(current_node=head):
            if current_node is not None:
                if not recursively_check(current_node.next):
                    return False
                if self.front_pointer.val != current_node.val:
                    return False
                self.front_pointer = self.front_pointer.next
            return True

        return recursively_check()


```





```python
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
            return True

        # Count the length of the linked list
        n = 0
        cur = head
        while cur:
            n += 1
            cur = cur.next
        
        odd_f = n % 2 == 1
        n_half = n // 2
        pre = None
        cur = head
        cnt = 0
        while cnt < n_half:
            next_node = cur.next
            cur.next = pre
            pre = cur
            cur = next_node
            cnt += 1
        
        if odd_f:
            cur = cur.next  # Skip the middle node if the length is odd

        # Compare the reversed first half and the second half.
        while cur and pre:
            if cur.val != pre.val:
                return False
            cur = cur.next
            pre = pre.next
        
        return True

if __name__ == "__main__":
    sol = Solution()
    # Test case for non-palindrome linked list
    head = ListNode(1, ListNode(2))
    print(sol.isPalindrome(head))  # Expected output: False

    # Test case for palindrome linked list
    # Uncomment the following line to test a palindrome linked list
    # head = ListNode(1, ListNode(2, ListNode(2, ListNode(1))))
    # print(sol.isPalindrome(head))  # Expected output: True
```



### LC680.验证回文串II

双指针，https://leetcode.cn/problems/valid-palindrome-ii/

给你一个字符串 `s`，**最多** 可以从中删除一个字符。

请你判断 `s` 是否能成为回文字符串：如果能，返回 `true` ；否则，返回 `false` 。

 

**示例 1：**

```
输入：s = "aba"
输出：true
```

**示例 2：**

```
输入：s = "abca"
输出：true
解释：你可以删除字符 'c' 。
```

**示例 3：**

```
输入：s = "abc"
输出：false
```

 

**提示：**

- `1 <= s.length <= 10^5`
- `s` 由小写英文字母组成



```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        def is_palindrome(subs, left, right):
            """检查子串 subs[left:right+1] 是否为回文"""
            while left < right:
                if subs[left] != subs[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        left, right = 0, len(s) - 1
        
        while left < right:
            if s[left] != s[right]:  
                # 尝试删除左边或右边的字符，看是否是回文
                return is_palindrome(s, left + 1, right) or is_palindrome(s, left, right - 1)
            left += 1
            right -= 1
        
        return True  # 如果从头到尾都是回文，直接返回 True

```



### sy78: 回文字符串 简单

https://sunnywhy.com/sfbj/3/6/78

如果一个字符串逆序后与正序相同，那么称这个字符串为回文字符串。例如`abcba`是回文字符串，`abcca`不是回文字符串。

给定一个字符串，判断它是否是回文字符串。

**输入描述**

一个非空字符串（长度不超过 ，仅由小写字母组成）。

**输出描述**

如果是回文字符串，那么输出`YES`，否则输出`NO`。

样例1

输入

```
abcba
```

输出

```
YES
```

样例2

输入

```
abcca
```

输出

```
NO
```



```python
s = input()
s_rev = s[::-1]
if s == s_rev:
    print('YES')
else:
    print('NO')
```





### 04067: 回文数字（Palindrome Number）

http://cs101.openjudge.cn/practice/04067/

给出一系列非负整数，判断是否是一个回文数。回文数指的是正着写和倒着写相等的数。

**输入**

若干行，每行是一个非负整数（不超过99999999）

**输出**

对每行输入，如果其是一个回文数，输出YES。否则输出NO。

样例输入

```
11
123
0
14277241
67945497
```

样例输出

```
YES
NO
YES
YES
NO
```





```python
def isPalindrome(s):
    if len(s) < 1:
        return False
    if len(s) == 1:
        return True

    front = 0
    back = len(s) - 1
    while front < back:
        if s[front] != s[back]:
            return False
        else:
            front += 1
            back -= 1

    return True

while True:
    try:
        s = input()
        print('YES' if isPalindrome(s) else 'NO')
    except:
        break
```





Use the deque from the collections module. The is_palindrome function checks if a number is a palindrome by converting it to a string, storing it in a deque, and then comparing the first and last elements until the deque is empty or only contains one element.

```python
from collections import deque

def is_palindrome(num):
    num_str = str(num)
    num_deque = deque(num_str)
    while len(num_deque) > 1:
        if num_deque.popleft() != num_deque.pop():
            return "NO"
    return "YES"

while True:
    try:
        num = int(input())
        print(is_palindrome(num))
    except EOFError:
        break
```





## 中等

### LC5.最长回文子串

dp, two pointers, string, https://leetcode.cn/problems/longest-palindromic-substring/

给你一个字符串 `s`，找到 `s` 中最长的 

回文子串。

**示例 1：**

```
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
```

**示例 2：**

```
输入：s = "cbbd"
输出："bb"
```

 

**提示：**

- `1 <= s.length <= 1000`
- `s` 仅由数字和英文字母组成



**Plan**

1. Initialize a 2D list `dp` where `dp[i][j]` will be `True` if the substring `s[i:j+1]` is a palindrome.
2. Iterate through the string in reverse order to fill the `dp` table.
3. For each character, check if the substring is a palindrome by comparing the characters at the ends and using the previously computed values in `dp`.
4. Keep track of the start and end indices of the longest palindromic substring found.
5. Return the substring defined by the start and end indices.

对于一个子串而言，如果它是回文串，并且长度大于 2，那么将它首尾的两个字母去除之后，它仍然是个回文串。

状态：`dp[i][j]`表示子串`s[i:j+1]`是否为回文子串

状态转移方程：`dp[i][j] = dp[i+1][j-1] ∧ (S[i] == s[j])`

动态规划中的边界条件，即子串的长度为 1 或 2。对于长度为 1 的子串，它显然是个回文串；对于长度为 2 的子串，只要它的两个字母相同，它就是一个回文串。

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n == 0:
            return ""

        # Initialize the dp table
        dp = [[False] * n for _ in range(n)]
        start, max_length = 0, 1

        # Every single character is a palindrome
        for i in range(n):
            dp[i][i] = True

        # Check for palindromes of length 2
        for i in range(n - 1):
            if s[i] == s[i + 1]:
                dp[i][i + 1] = True
                start = i
                max_length = 2

        # Check for palindromes of length greater than 2
        for length in range(3, n + 1):  # 遍历所有可能的子串长度（从3到n）
            for i in range(n - length + 1):  # 遍历所有可能的起始位置 i
                j = i + length - 1  # 计算对应的结束位置 j
                if s[i] == s[j] and dp[i + 1][j - 1]:  # 检查是否满足回文条件
                    dp[i][j] = True  # 标记子串 s[i:j+1] 是回文
                    start = i  # 更新最长回文子串的起始位置
                    max_length = length  # 更新最长回文子串的长度

        return s[start:start + max_length]

if __name__ == "__main__":
    sol = Solution()
    print(sol.longestPalindrome("babad"))  # Output: "bab" or "aba"
    print(sol.longestPalindrome("cbbd"))   # Output: "bb"
```



**Plan**

1. Initialize variables to store the start and end indices of the longest palindromic substring.
2. Iterate through each character in the string, treating each character and each pair of consecutive characters as potential centers of palindromes.
3. For each center, expand outwards while the characters on both sides are equal.
4. Update the start and end indices if a longer palindrome is found.
5. Return the substring defined by the start and end indices.

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if not s:
            return ""
        
        start, end = 0, 0
        
        for i in range(len(s)):
            odd_len = self.expandAroundCenter(s, i, i)
            even_len = self.expandAroundCenter(s, i, i + 1)
            max_len = max(odd_len, even_len)
            if max_len > end - start:
                start = i - (max_len - 1) // 2
                end = i + max_len // 2
        
        return s[start:end + 1]
    
    def expandAroundCenter(self, s: str, left: int, right: int) -> int:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

if __name__ == "__main__":
    sol = Solution()
    print(sol.longestPalindrome("babad"))  # Output: "bab" or "aba"
    print(sol.longestPalindrome("cbbd"))   # Output: "bb"
```

<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20241125155859557.png" alt="image-20241125155859557" style="zoom:50%;" />

这个双指针是从中间往两边跑。



Manacher算法

https://leetcode.cn/problems/longest-palindromic-substring/solutions/255195/zui-chang-hui-wen-zi-chuan-by-leetcode-solution/

```python
class Solution:
    def expand(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return (right - left - 2) // 2

    def longestPalindrome(self, s: str) -> str:
        end, start = -1, 0
        s = '#' + '#'.join(list(s)) + '#'
        arm_len = []
        right = -1
        j = -1
        for i in range(len(s)):
            if right >= i:
                i_sym = 2 * j - i
                min_arm_len = min(arm_len[i_sym], right - i)
                cur_arm_len = self.expand(s, i - min_arm_len, i + min_arm_len)
            else:
                cur_arm_len = self.expand(s, i, i)
            arm_len.append(cur_arm_len)
            if i + cur_arm_len > right:
                j = i
                right = i + cur_arm_len
            if 2 * cur_arm_len + 1 > end - start:
                start = i - cur_arm_len
                end = i + cur_arm_len
        return s[start+1:end+1:2]

```



> 这段代码是用于解决“最长回文子串”问题的一个Python实现。它使用了Manacher算法的变种，通过在每个字符间插入特殊字符（这里是`#`）来处理奇数和偶数长度的回文字符串。让我们逐段解析这个解决方案：
>
> 扩展函数 `expand`
>
> - 此函数尝试从给定的中心点向两边扩展，以找到以该中心点为中心的最大回文半径。
> - 如果左右指针指向的字符相等且都在字符串范围内，则继续向外扩展。
> - 返回值为最大回文半径，即`(right - left - 2) // 2`。
>
> 主函数 `longestPalindrome`
>
> - 首先，将原始字符串`s`转换成一个新字符串，其中每个字符之间都插入了一个`#`，这样可以确保所有可能的回文子串都是奇数长度，便于处理。
> - `end` 和 `start` 变量用于记录最长回文子串的边界。
> - `arm_len` 列表存储每个位置的回文臂长。
> - `right` 和 `j` 分别表示当前已知最右端的回文子串的右边界及其对应的中心点。
> - 在遍历新字符串时，如果当前索引`i`位于`right`内，那么利用对称性，计算出`i`关于`j`的对称点`i_sym`，并确定最小臂长。
> - 使用`expand`函数计算当前位置的实际臂长，并更新`arm_len`列表。
> - 如果当前臂长加上`i`超过了`right`，则更新`right`和`j`。
> - 如果当前找到的回文子串比之前记录的更长，则更新`start`和`end`。
> - 最后，返回原始字符串中对应最长回文子串的部分，注意去掉添加的`#`符号。
>
> 整个算法的核心思想是利用已经计算过的回文信息来减少重复计算，从而提高效率。这种方法的时间复杂度接近O(n)，空间复杂度为O(n)。



min_arm_len = min(arm_len[i_sym], right - i)	 如何理解？

> `min_arm_len = min(arm_len[i_sym], right - i)` 这一行代码是用来确定从当前位置 `i` 开始进行中心扩展时，可以跳过的最小臂长。这里有两个关键点需要理解：
>
> 1. **`arm_len[i_sym]`**:
>    - `i_sym` 是位置 `i` 关于当前已知最右边回文子串中心 `j` 的对称点，计算方式为 `2 * j - i`。
>    - `arm_len[i_sym]` 表示的是对称点 `i_sym` 处的回文臂长。如果 `i_sym` 位置的回文臂长足够大，那么根据对称性，我们可以在 `i` 位置直接获得至少相同的臂长，因为这两个位置关于 `j` 对称。
>
> 2. **`right - i`**:
>    - `right` 是当前已知最右端回文子串的边界。
>    - `right - i` 表示从位置 `i` 到当前已知最右端的距离。由于我们已经知道在 `right` 之前的字符都是回文的一部分，因此在 `i` 位置进行扩展时，我们至少可以达到 `right` 位置，而不需要重新验证这些字符。
>
> 综合以上两点，`min_arm_len` 取两者中的较小值，确保了以下几点：
>
> - 如果 `i_sym` 位置的臂长小于或等于 `right - i`，我们可以直接使用 `i_sym` 位置的臂长，因为它不会超出已知的回文区域。
> - 如果 `i_sym` 位置的臂长大于 `right - i`，则我们只能保证到 `right` 位置是回文，所以取 `right - i` 作为最小臂长。
>
> 这样做的目的是为了减少不必要的重复计算，通过利用之前计算的结果（即 `arm_len` 中的信息）来加速找到当前位置的最长回文臂长的过程。这实际上是Manacher算法中的一种优化手段，它允许我们在某些情况下快速跳过已经确认的部分，从而提高算法的整体效率。





思路：马拉车算法（Manacher）

首先一个比较基础的想法是研究以某一个位置为中心的回文串，但考虑到可能存在`aba`和`aa`这样不同奇偶性的回文串，将其补齐成类似`#a#b#a#`的形式，这样所有的回文串都是奇数。
然后，考虑某一个位置为中心的回文串，朴素的算法就是一步一步地扩大半径，直到不再回文，即这一部分代码

```python
while 0 <= i - k and i + k < n and ns[i - k] == ns[i + k]:
    k += 1
```

而马拉车算法在这一部分朴素的算法之外，进一步考虑到在我找到这个位置最长的回文串的时候，我在后面的寻找过程中可以利用这个信息。

我们维护一个最右边的回文串的边界`l, r`，如果`i`已经超出了这一部分，那么就只能直接调用后面的朴素算法；否则，我们可以利用之前的信息，考察在目前的`l, r`下对称的那个点`l+r-i`的最长回文串，将其设为我们朴素算法的起始半径来进行循环。

特别地，如果对称过来的半径太长，超出了`r`的部分事实上我们目前还没进行研究，所以最大值只能到`r-i-1`。

每次求解之后更新最右的`r`以及对应的`l`。

朴素算法，时间复杂度O(n²); Manacher，时间复杂度O(n)。（while循环每进一次r至少变大1）

```python
# 曹以楷 24物理学院
class Solution:
    def longestPalindrome(self, s: str) -> str:
        ns = f"#{'#'.join(s)}#"
        n = len(ns)
        # Manacher start
        d = [0]*n
        l, r = 0, -1
        for i in range(n):
            k = 1 if i > r else min(d[l + r - i], r - i + 1)
            while 0 <= i - k and i + k < n and ns[i - k] == ns[i + k]:
                k += 1
            d[i] = k
            k -= 1
            if i + k > r:
                l = i - k
                r = i + k
        # Manacher end
        cnt = max(d)
        idx = d.index(cnt)

        return ns[idx-cnt+1:idx+cnt].replace("#", "")

```



思路：

- 最开始我没看到题目要求子串必须连续！我想了很久，想到了可能要把原字符串逆序但不知道逆序之后干什么，然后一个同学告诉我直接求最长公共子序列就好，感觉瞬间明白了
- 然后发现子串要求连续，在原来程序的基础上，取出所有的公共子序列，再找其中既是回文的又是最长的那个，也算是过了

```python
# 颜鼎堃 24工学院
class Solution:
    def longestPalindrome(self, s: str) -> str:
        t = "".join(reversed(s))
        n = len(s)
        strings = [["" for i in range(n + 2)] for j in range(n + 2)]
        for i in range(n):
            for j in range(n):
                if s[i] == t[j]:
                    strings[i + 1][j + 1] = strings[i][j] + s[i]
        pos_pal = set()
        max_par = s[0]
        for i in map(set, strings):
            pos_pal |= i
        for i in pos_pal:
            if i and i == i[::-1]:
                max_par = max(max_par, i, key=len)
        return max_par


if __name__ == '__main__':
    sol = Solution()
    print(sol.longestPalindrome(input()))
```



### LC131.分割回文串

dp, backtracking, https://leetcode.cn/problems/palindrome-partitioning/

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是 **回文串**。返回 `s` 所有可能的分割方案。回文串是指向前和向后读都相同的字符串。



**示例 1：**

```
输入：s = "aab"
输出：[["a","a","b"],["aa","b"]]
```

**示例 2：**

```
输入：s = "a"
输出：[["a"]]
```

 

**提示：**

- `1 <= s.length <= 16`
- `s` 仅由小写英文字母组成





```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def is_palindrome(sub):
            return sub == sub[::-1]

        def backtrack(start, path):
            if start == len(s):
                res.append(path[:])
                return
            for end in range(start + 1, len(s) + 1):
                if is_palindrome(s[start:end]):
                    path.append(s[start:end])
                    backtrack(end, path)
                    path.pop()
        
        res = []
        backtrack(0, [])
        return res
```





```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        ans = []
        def divide(ans_list,word):
            if len(word) == 0:
                ans.append(ans_list)
                return
            for i in range(1,len(word)+1):
                if word[:i] == word[:i][::-1]:
                    divide(ans_list+[word[:i]],word[i:])
        divide([],s)
        return ans
```





```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        def is_palindrome(s):
            return s == s[::-1]

        def backtracking(start, path):
            if start == len(s):
                res.append(path)
                return 
            for i in range(start, len(s)):
                if is_palindrome(s[start:i+1]):
                    backtracking(i+1, path + [s[start:i+1]])
        
        res = []
        backtracking(0, [])
        return res
```



第一部分的判断某一段子串是不是回文串的 dp 写法；第二部分是 dfs 找切片。其中第一部分的 dp 的值都是布尔值，这样方便后续判断某一个子串是不是回文串；第二部分应该是双指针的思路，用 i 来遍历所有起点，用 j 来从每一个起点开始遍历第一处断点，这种写法也值得积累。

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)
        # 预处理回文子串
        is_palindrome = [[False] * n for _ in range(n)]
        for right in range(n):
            for left in range(right + 1):
                if s[left] == s[right] and (right - left <= 1 or is_palindrome[left + 1][right - 1]):
                    is_palindrome[left][right] = True

        res = []
        path = []

        def backtrack(start):
            if start == n:
                res.append(path[:])  # 复制当前路径
                return
            for end in range(start, n):
                if is_palindrome[start][end]:  # 只在是回文的地方切割
                    path.append(s[start:end + 1])
                    backtrack(end + 1)
                    path.pop()  # 撤销选择

        backtrack(0)
        return res

        
```



### LC1328.破坏回文串

greedy, https://leetcode.cn/problems/break-a-palindrome/

给你一个由小写英文字母组成的回文字符串 `palindrome` ，请你将其中 **一个** 字符用任意小写英文字母替换，使得结果字符串的 **字典序最小** ，且 **不是** 回文串。

请你返回结果字符串。如果无法做到，则返回一个 **空串** 。

如果两个字符串长度相同，那么字符串 `a` 字典序比字符串 `b` 小可以这样定义：在 `a` 和 `b` 出现不同的第一个位置上，字符串 `a` 中的字符严格小于 `b` 中的对应字符。例如，`"abcc”` 字典序比 `"abcd"` 小，因为不同的第一个位置是在第四个字符，显然 `'c'` 比 `'d'` 小。

 

**示例 1：**

```
输入：palindrome = "abccba"
输出："aaccba"
解释：存在多种方法可以使 "abccba" 不是回文，例如 "zbccba", "aaccba", 和 "abacba" 。
在所有方法中，"aaccba" 的字典序最小。
```

**示例 2：**

```
输入：palindrome = "a"
输出：""
解释：不存在替换一个字符使 "a" 变成非回文的方法，所以返回空字符串。
```

 

**提示：**

- `1 <= palindrome.length <= 1000`
- `palindrome` 只包含小写英文字母。



```python
class Solution:
    def breakPalindrome(self, palindrome: str) -> str:
        length = len(palindrome)
    
        # 如果长度为1，则无法转换，返回空字符串
        if length == 1:
            return ""
        
        for i in range(length // 2):
            # 尝试将不是'a'的字符替换为'a'
            if palindrome[i] != 'a':
                # 如果是前半部分（忽略中点对于奇数长度的情况）
                return palindrome[:i] + 'a' + palindrome[i+1:]
        
        # 如果前半部分全部是'a'，将最后一个字符变为'b'
        return palindrome[:-1] + 'b'
```



### LC3503.子字符串连接后的最长回文串I

brute force, https://leetcode.cn/problems/longest-palindrome-after-substring-concatenation-i/

给你两个字符串 `s` 和 `t`。

你可以从 `s` 中选择一个子串（可以为空）以及从 `t` 中选择一个子串（可以为空），然后将它们 **按顺序** 连接，得到一个新的字符串。

返回可以由上述方法构造出的 **最长** 回文串的长度。

**回文串** 是指正着读和反着读都相同的字符串。

**子字符串** 是指字符串中的一个连续字符序列。

 

**示例 1：**

**输入：** s = "a", t = "a"

**输出：** 2

**解释：**

从 `s` 中选择 `"a"`，从 `t` 中选择 `"a"`，拼接得到 `"aa"`，这是一个长度为 2 的回文串。

**示例 2：**

**输入：** s = "abc", t = "def"

**输出：** 1

**解释：**

由于两个字符串的所有字符都不同，最长的回文串只能是任意一个单独的字符，因此答案是 1。

**示例 3：**

**输入：** s = "b", t = "aaaa"

**输出：** 4

**解释：**

可以选择 `"aaaa"` 作为回文串，其长度为 4。

**示例 4：**

**输入：** s = "abcde", t = "ecdba"

**输出：** 5

**解释：**

从 `s` 中选择 `"abc"`，从 `t` 中选择 `"ba"`，拼接得到 `"abcba"`，这是一个长度为 5 的回文串。

 

**提示：**

- `1 <= s.length, t.length <= 30`
- `s` 和 `t` 仅由小写英文字母组成。



```python
from typing import List

class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        # 检查字符串是否为回文
        def is_palindrome(sub: str) -> bool:
            return sub == sub[::-1]

        n1, n2 = len(s), len(t)
        max_len = 0

        # 枚举 s 和 t 的所有子串组合
        for i in range(n1 + 1):  # s 的子串起点
            for j in range(i, n1 + 1):  # s 的子串终点
                for k in range(n2 + 1):  # t 的子串起点
                    for l in range(k, n2 + 1):  # t 的子串终点
                        combined = s[i:j] + t[k:l]
                        if is_palindrome(combined):
                            max_len = max(max_len, len(combined))

        return max_len

if __name__ == '__main__':
    s = Solution()
    print(s.longestPalindrome("a", "a"))       # 输出：2
    print(s.longestPalindrome("abc", "def"))   # 输出：1
    print(s.longestPalindrome("b", "aaaa"))    # 输出：4
    print(s.longestPalindrome("abcde", "ecdba"))  # 输出：5
    print(s.longestPalindrome("n", "no"))      # 输出：2
```



### sy312: 回文链表 中等

https://sunnywhy.com/sfbj/7/3/312

现有n个结点（编号为从`1`到`n`），以及链表的第一个结点的编号，判断该链表是否是回文的，即链表的结点按数据域正序和逆序是相同的。

**输入**

第一行两个整数n、first（1<=first<=n<=100），分别表示结点的个数、链表第一个结点的编号；

接下来行，每行给出一个结点的信息：

```text
id data next
```

其中整数id（1 <=id <= n）表示结点的编号（每个id只出现一次，顺序不定），整数data（1 <= data <= 1000）表示结点的数据域，整数next（1 <= next <= n或 next = -1）表示当前结点指向的结点编号（其中next = -1表示`NULL`）。

**输出**

如果链表是回文的，那么输出`Yes`，否则输出`No`。

样例1

输入

```
5 2
1 12 3
2 10 4
3 10 -1
4 12 5
5 23 1
```

输出

```
Yes
```

解释

第一个结点编号为`2`，因此整条链表为：`2(10)->4(12)->5(23)->1(12)->3(10)->NULL`。

显然链表是回文的。

样例2

输入

```
2 1
1 23 2
2 10 -1
```

输出

```
No
```

解释

第一个结点编号为`1`，因此整条链表为：`1(23)->2(10)->NULL`。

显然链表不是回文的。



为了判断链表是否是回文的，我们可以首先遍历整个链表，将所有节点的数据域存储在一个列表中。然后，我们可以检查这个列表是否等于它的逆序。

以下是实现这个过程的Python代码：

```python
class Node:
    def __init__(self, id, data, next):
        self.id = id
        self.data = data
        self.next = next

n, first = map(int, input().split())
nodes = {}
for _ in range(n):
    id, data, next = map(int, input().split())
    nodes[id] = Node(id, data, next)

# Construct the linked list
linked_list = []
current = nodes[first]
while current is not None:
    linked_list.append(current.data)
    if current.next != -1:
        current = nodes[current.next]
    else:
        current = None

# Check if the linked list is a palindrome
if linked_list == linked_list[::-1]:
    print("Yes")
else:
    print("No")
```

这段代码首先读取输入，然后创建一个Node类的实例来存储每个节点的信息。然后，它遍历链表，将链表中的节点的数据域按照顺序添加到一个新的列表中。然后，它检查这个列表是否等于它的逆序，如果是，则输出"Yes"，否则输出"No"。



### 01159: Palindrome

dp, http://cs101.openjudge.cn/practice/01159/

A palindrome is a symmetrical string, that is, a string read identically from left to right as well as from right to left. You are to write a program which, given a string, determines the minimal number of characters to be inserted into the string in order to obtain a palindrome.

As an example, by inserting 2 characters, the string "Ab3bd" can be transformed into a palindrome ("dAb3bAd" or "Adb3bdA"). However, inserting fewer than 2 characters does not produce a palindrome.

**输入**

Your program is to read from standard input. The first line contains one integer: the length of the input string N, 3 <= N <= 5000. The second line contains one string with length N. The string is formed from uppercase letters from 'A' to 'Z', lowercase letters from 'a' to 'z' and digits from '0' to '9'. Uppercase and lowercase letters are to be considered distinct. 

**输出**

Your program is to write to standard output. The first line contains one integer, which is the desired minimal number.

样例输入

```
5
Ab3bd
```

样例输出

```
2
```

来源

IOI 2000



问题是需要找到将给定字符串转换为回文所需的最少插入字符数。回文是指从左到右和从右到左读都相同的字符串。可以通过动态规划的方法来解决这个问题，具体思路是计算原字符串与其反转字符串的最长公共子序列（LCS），然后用原字符串的长度减去LCS的长度，得到所需插入的最少字符数。

方法思路

1. **问题分析**：回文的结构具有对称性，利用动态规划来找到原字符串与其反转字符串的最长公共子序列。这个子序列的长度即为原字符串中最长回文子序列的长度。
2. **动态规划**：使用一维数组优化空间复杂度，将二维动态规划数组转换为两个一维数组，交替使用以节省空间。
3. **复杂度分析**：时间复杂度为O(n²)，空间复杂度为O(n)，其中n是字符串的长度。



Python, 4564ms AC. 如果用二维数组会超内存。

```python
def min_insertions_to_palindrome(n, s):
    t = s[::-1]
    dp = [0] * (n + 1)
    
    for i in range(n):
        current_char = s[i]
        prev_prev = 0
        for j in range(1, n + 1):
            current = dp[j]
            if current_char == t[j - 1]:
                dp[j] = prev_prev + 1
            else:
                if dp[j - 1] > dp[j]:
                    dp[j] = dp[j - 1]
            prev_prev = current
    
    return n - dp[n]

# 读取输入
n = int(input())
s = input().strip()

# 计算并输出结果
print(min_insertions_to_palindrome(n, s))

```



PyPy3, 404ms. 通过将核心逻辑封装到函数中，提升代码的执行速度。

```python
n = int(input())
s = input().strip()
t = s[::-1]

dp = [0] * (n + 1)

for i in range(n):
    current_char = s[i]
    prev_prev = 0
    for j in range(1, n + 1):
        current = dp[j]
        if current_char == t[j - 1]:
            dp[j] = prev_prev + 1
        else:
            if dp[j - 1] > dp[j]:
                dp[j] = dp[j - 1]
        prev_prev = current

print(n - dp[n])

```



### 03247:回文素数

http://cs101.openjudge.cn/practice/03247/

一个数如果从左往右读和从右往左读数字是相同的，则称这个数是回文数，如121，1221，15651都是回文数。给定位数n，找出所有既是回文数又是素数的n位十进制数。（注：不考虑超过整型数范围的情况）。

**输入**

位数n,其中1<=n<=9。

**输出**

第一行输出满足条件的素数个数。
第二行按照从小到大的顺序输出所有满足条件的素数，两个数之间用一个空格区分。

样例输入

```
1
```

样例输出

```
4
2 3 5 7
```



```python
import math

def is_prime(num):
    if num < 2:
        return False
    if num in {2, 3, 5, 7}:
        return True
    if num % 2 == 0 or num % 5 == 0:
        return False
    for i in range(3, int(math.sqrt(num)) + 1, 2):
        if num % i == 0:
            return False
    return True

def generate_palindromes(n):
    palindromes = []
    if n == 1:
        return [2, 3, 5, 7]  # 1位数的素数回文数

    half_len = (n + 1) // 2  # 只需要构造前半部分
    start, end = 10**(half_len - 1), 10**half_len
    
    for first_half in range(start, end):
        first_half_str = str(first_half)
        if n % 2 == 0:  # 偶数位
            palindrome = int(first_half_str + first_half_str[::-1])
        else:  # 奇数位
            #palindrome = int(first_half_str + first_half_str[-2::-1])
            palindrome = int(first_half_str + first_half_str[:-1][::-1])
        
        if is_prime(palindrome):
            palindromes.append(palindrome)
    
    return palindromes

def find_palindromic_primes(n):
    primes = generate_palindromes(n)
    print(len(primes))
    print(" ".join(map(str, primes)))

# 输入
n = int(input().strip())
find_palindromic_primes(n)

```



### 25815: 回文字符串

http://cs101.openjudge.cn/practice/25815/

给定一个字符串 S ，最少需要几次增删改操作可以把 S 变成一个回文字符串？

一次操作可以在任意位置插入一个字符，或者删除任意一个字符，或者把任意一个字符修改成任意其他字符。

**输入**

字符串 S。S 的长度不超过100, 只包含'A'-'Z'。

**输出**

最少的修改次数。

样例输入

```
ABAD
```

样例输出

```
1
```

来源

hihoCoder



这是一个经典的动态规划问题，可以使用动态规划来解决这个问题。定义一个二维数组 dp，其中 `dp[i][j]` 表示将字符串 S 的子串 S[i...j] 变成回文串所需要的最小操作次数。

使用状态转移方程来填充 dp 数组：

- 如果 S[i] == S[j]，那么 `dp[i][j] = dp[i+1][j-1]`，因为两端的字符已经相等，不需要进行任何操作。
- 如果 S[i] != S[j]，那么 `dp[i][j] = min(dp[i+1][j], dp[i][j-1], dp[i+1][j-1]) + 1`，可以选择删除 S[i] 或 S[j]，或者将 S[i] 或 S[j] 修改为另一个字符。

最后，`dp[0][n-1]` 就是答案，其中 n 是字符串 S 的长度。

以下是使用 Python 实现的代码：

```python
def min_operations(s):
    n = len(s)
    dp = [[0]*n for _ in range(n)]
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1]
            else:
                dp[i][j] = min(dp[i+1][j], dp[i][j-1], dp[i+1][j-1]) + 1
    return dp[0][n-1]

s = input().strip()
print(min_operations(s))
```

这段代码首先读取输入的字符串，然后调用 `min_operations` 函数计算并打印出将字符串变成回文串所需要的最小操作次数。





## 挑战

### LC132.分割回文串II

dp, https://leetcode.cn/problems/palindrome-partitioning-ii/

给你一个字符串 `s`，请你将 `s` 分割成一些子串，使每个子串都是回文串。

回文串是向前和向后读都相同的字符串。

返回符合要求的 **最少分割次数** 。

 

**示例 1：**

```
输入：s = "aab"
输出：1
解释：只需一次分割就可将 s 分割成 ["aa","b"] 这样两个回文子串。
```

**示例 2：**

```
输入：s = "a"
输出：0
```

**示例 3：**

```
输入：s = "ab"
输出：1
```

 

**提示：**

- `1 <= s.length <= 2000`
- `s` 仅由小写英文字母组成



使用动态规划解决最小回文切割问题：

1. **is_palindrome 表** 用于预计算每个子串是否是回文串。
2. **dp 表** 存储以每个字符结尾的最小切割次数。
3. **双层循环**：外层遍历字符串的每个字符，内层反向检查从每个字符到当前字符的子串是否为回文。
4. **时间复杂度** 是 $O(n^2)$ ，因为两层循环分别遍历了所有子串。

```python
class Solution:
    def minCut(self, s: str) -> int:

        n = len(s)
        is_palindrome = [[False] * n for _ in range(n)]
        dp = [0] * n

        for i in range(n):
            min_cuts = i  # max cuts possible is i (cut each character)
            for j in range(i + 1):
                if s[j] == s[i] and (i - j <= 2 or is_palindrome[j + 1][i - 1]):
                    is_palindrome[j][i] = True
                    min_cuts = 0 if j == 0 else min(min_cuts, dp[j - 1] + 1)
            dp[i] = min_cuts

        return dp[-1]

if __name__ == "__main__":
    sol = Solution()

    print(sol.minCut("aab"))  # 输出：1
    print(sol.minCut("a"))  # 输出：0
    print(sol.minCut("ab"))  # 输出：1

```

在该算法中，`dp[j - 1] + 1` 的使用是为了计算将字符串 `s` 分割成回文子串所需的最小切割次数。这里的 `+1` 代表在位置 `j-1` 和 `i` 之间进行一次新的切割。

具体来说：

当你发现从 `j` 到 `i` 的子串 `s[j:i+1]` 是一个回文时，你想要知道为了使这个回文成为可能的一部分，前面的子串（即 `s[0:j-1]`）需要多少次切割才能全部变成回文子串。这就是 `dp[j-1]` 所记录的信息：到达索引 `j-1` 之前（包括 `j-1`），最少需要几次切割来使得所有子串都是回文。

一旦你知道了 `dp[j-1]`，如果你决定把当前找到的回文子串 `s[j:i+1]` 看作一个新的不分割的整体，那么你就需要在这个回文子串之前的位置（即在 `j-1` 和 `i` 之间）做一个新的切割。因此，在 `dp[j-1]` 的基础上加 `1`，表示这次新的切割。

所以，`dp[j - 1] + 1` 实际上是在说：“如果我选择将从 `j` 到 `i` 的这部分作为一个回文子串，那么我需要之前的部分 `s[0:j-1]` 达到最少切割状态下的切割次数（即 `dp[j-1]`），再加上这一次新做的切割”。



### LC1278.分割回文串III

dp, https://leetcode.cn/problems/palindrome-partitioning-iii/

给你一个由小写字母组成的字符串 `s`，和一个整数 `k`。

请你按下面的要求分割字符串：

- 首先，你可以将 `s` 中的部分字符修改为其他的小写英文字母。
- 接着，你需要把 `s` 分割成 `k` 个非空且不相交的子串，并且每个子串都是回文串。

请返回以这种方式分割字符串所需修改的最少字符数。

 

**示例 1：**

```
输入：s = "abc", k = 2
输出：1
解释：你可以把字符串分割成 "ab" 和 "c"，并修改 "ab" 中的 1 个字符，将它变成回文串。
```

**示例 2：**

```
输入：s = "aabbc", k = 3
输出：0
解释：你可以把字符串分割成 "aa"、"bb" 和 "c"，它们都是回文串。
```

**示例 3：**

```
输入：s = "leetcode", k = 8
输出：0
```

 

**提示：**

- `1 <= k <= s.length <= 100`
- `s` 中只含有小写英文字母。



思路分为两步：

1. **预处理子串转换为回文所需的修改数**  
   用二维数组 `cost[i][j]` 表示将子串 `s[i:j+1]` 修改为回文所需的最少修改数。可以利用动态规划，从两端向中间比较字符，若两端字符不同则需要修改。

2. **动态规划求分割方案**  
   定义 `dp[i][t]` 表示将 `s[0:i]` 分割为 `t` 个回文串所需的最少修改数。转移时枚举最后一个回文串的起点位置 `p`，即  

   $dp[i][t] = \min_{p \text{ from } t-1 \text{ to } i-1} \{ dp[p][t-1] + cost[p][i-1] \}$

   初始时 `dp[i][1] = cost[0][i-1]`（即把整个子串当作一个回文串修改）。

> Q. 这个状态转移方程的意思是？
>
> **1. 变量解释**
>
> - `dp[i][t] `：表示将前  i  个元素分成  t  组的最小代价。
> -  `cost[p][i-1] `：表示将区间 [p, i-1] 作为一组的代价。
> -  p 是一个分界点，它将前  i  个元素分成  t  组，其中前  t-1  组来自 ` dp[p][t-1] `，最后一组是 `[p, i-1]`。
>
> ---
>
> **2. 递推含义**
>
> $
> dp[i][t] = \min_{p \text{ from } t-1 \text{ to } i-1} \{ dp[p][t-1] + cost[p][i-1] \}
> $
> 这表示 **要把前 \( i \) 个元素分成 \( t \) 组**，那么最优方案是从某个位置 \( p \) 进行最后一次划分：
>
> - 先把前  p  个元素分成  t-1  组（对应 ` dp[p][t-1] `）。
> - 再把区间 [p, i-1] 作为第 \( t \) 组，代价是 ` cost[p][i-1] `。
> - 遍历所有可能的  p  取最小值，确保总代价最小。
>
> ---
>
> **3. 例子**
>
> 假设有一个数组 `[A, B, C, D, E]`，我们希望分成 3 组，定义 `cost[i][j]` 为区间 `[i, j]` 的代价。
>
> - ` dp[i][t] ` 代表前  i  个元素分成  t  组的最小代价。
> - 例如，计算 ` dp[5][3] ` 时，我们需要考虑最后一组从哪个位置 \( p \) 开始：
>   -  p = 2 （前两个元素是两组，后面 `[C, D, E]` 作为一组）
>   -  p = 3 （前三个元素是两组，后面 `[D, E]` 作为一组）
>   -  p = 4 （前四个元素是两组，后面 `[E]` 作为一组）
>
> $
> dp[5][3] = \min(dp[2][2] + cost[2][4], dp[3][2] + cost[3][4], dp[4][2] + cost[4][4])
> $
>
> ---
>
> **4. 适用场景**
>
> 这个方程常用于 **分组动态规划（Partition DP）**，适用于：
> - **区间划分问题**（如将数组分成若干段，每段有计算代价）
> - **K 端分割问题**（如 k-means clustering、文本分割、段式 DP）
>
> 如果你想优化这个 DP，通常可以用 **单调队列优化**（如 Knuth 优化）来加速 `p` 的选取，从 $ O(N) $ 降低到 $ O(\log N) $ 或 $ O(1) $ 级别。



下面是完整代码：

```python
class Solution:
    def palindromePartition(self, s: str, k: int) -> int:
        n = len(s)
        # 如果 k 等于 s 长度，每个字符单独为回文，无需修改
        if k == n:
            return 0

        # 预处理：cost[i][j] 表示将 s[i:j+1] 修改为回文的最小修改数
        cost = [[0] * n for _ in range(n)]
        # 注意：单个字符天然回文，cost[i][i]=0
        # 从下标 i 从 n-1 递减，这样可以确保计算 cost[i+1][j-1] 时已被处理
        for i in range(n - 1, -1, -1):
            for j in range(i + 1, n):
                if s[i] == s[j]:
                    cost[i][j] = cost[i + 1][j - 1] if j - i > 1 else 0
                else:
                    cost[i][j] = (cost[i + 1][j - 1] if j - i > 1 else 0) + 1

        # dp[i][t] 表示 s[0:i] 分割为 t 个回文子串所需的最少修改数
        # 注意：i 的范围 0...n，t 的范围 0...k
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        # 初始化，当分割成 1 个回文串时
        for i in range(1, n + 1):
            dp[i][1] = cost[0][i - 1]

        # 动态规划
        for i in range(1, n + 1):  # i 表示 s[0:i]
            for t in range(2, min(k, i) + 1):  # t 从 2 到 k，且 t<=i
                # 枚举最后一个回文串的起点 p, 保证前面至少有 t-1 个字符
                for p in range(t - 1, i):
                    dp[i][t] = min(dp[i][t], dp[p][t - 1] + cost[p][i - 1])

        return dp[n][k]


# 测试用例
if __name__ == "__main__":
    tests = [
        ("abc", 2, 1),
        ("aabbc", 3, 0),
        ("leetcode", 8, 0)
    ]
    sol = Solution()
    for s, k, expected in tests:
        result = sol.palindromePartition(s, k)
        print(f"s = {s}, k = {k}, 最少修改数 = {result}, 预期 = {expected}")

```

说明

- **预处理 cost 数组**  
  对于每个区间 `[i, j]`，如果 `s[i] == s[j]`，则 `cost[i][j] = cost[i+1][j-1]`；否则需要增加 1 的修改量。注意边界处理，当区间长度为 2 时直接比较即可。

- **动态规划转移**  
  对于分割成多个回文串，枚举最后一个分割点 `p`，保证前面部分已经被正确分割为 `t-1` 个回文串，后面部分 `s[p:i]` 转换为回文的代价为 `cost[p][i-1]`。



### LC1745.分割回文串IV

dp, https://leetcode.cn/problems/palindrome-partitioning-iv/

给你一个字符串 `s` ，如果可以将它分割成三个 **非空** 回文子字符串，那么返回 `true` ，否则返回 `false` 。

当一个字符串正着读和反着读是一模一样的，就称其为 **回文字符串** 。

 

**示例 1：**

```
输入：s = "abcbdd"
输出：true
解释："abcbdd" = "a" + "bcb" + "dd"，三个子字符串都是回文的。
```

**示例 2：**

```
输入：s = "bcbddxy"
输出：false
解释：s 没办法被分割成 3 个回文子字符串。
```

 

**提示：**

- `3 <= s.length <= 2000`
- `s` 只包含小写英文字母。



```python
class Solution:
    def checkPartitioning(self, s: str) -> bool:
        n = len(s)
        # dp[i][j] 表示 s[i:j+1] 是否是回文串
        dp = [[False] * n for _ in range(n)]

        # 预计算所有子串的回文情况
        for j in range(n):
            for i in range(j + 1):
                if s[i] == s[j] and (j - i <= 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True

        # 检查是否可以分成三个回文子串
        for i in range(1, n - 1):  # 第一个分割点
            if dp[0][i - 1]:
                for j in range(i, n - 1):  # 第二个分割点
                    if dp[i][j] and dp[j + 1][n - 1]:
                        return True
        return False

if __name__ == "__main__":
    sol = Solution()
    print(sol.checkPartitioning("abcbdd"))
    print(sol.checkPartitioning("bcbddxy"))
```



### LC3504.子字符串连接后的最长回文串II

dp, greedy, https://leetcode.cn/problems/longest-palindrome-after-substring-concatenation-ii/

给你两个字符串 `s` 和 `t`。

你可以从 `s` 中选择一个子串（可以为空）以及从 `t` 中选择一个子串（可以为空），然后将它们 **按顺序** 连接，得到一个新的字符串。

返回可以由上述方法构造出的 **最长** 回文串的长度。

**回文串** 是指正着读和反着读都相同的字符串。

**子字符串** 是指字符串中的一个连续字符序列。

 

**示例 1：**

**输入：** s = "a", t = "a"

**输出：** 2

**解释：**

从 `s` 中选择 `"a"`，从 `t` 中选择 `"a"`，拼接得到 `"aa"`，这是一个长度为 2 的回文串。

**示例 2：**

**输入：** s = "abc", t = "def"

**输出：** 1

**解释：**

由于两个字符串的所有字符都不同，最长的回文串只能是任意一个单独的字符，因此答案是 1。

**示例 3：**

**输入：** s = "b", t = "aaaa"

**输出：** 4

**解释：**

可以选择 `"aaaa"` 作为回文串，其长度为 4。

**示例 4：**

**输入：** s = "abcde", t = "ecdba"

**输出：** 5

**解释：**

从 `s` 中选择 `"abc"`，从 `t` 中选择 `"ba"`，拼接得到 `"abcba"`，这是一个长度为 5 的回文串。

 

**提示：**

- `1 <= s.length, t.length <= 1000`
- `s` 和 `t` 仅由小写英文字母组成。



```python
class Solution:
    def longestPalindrome(self, s: str, t: str) -> int:
        # 反转 t，方便匹配 s 的前缀
        t = t[::-1]
        
        # 将字符转换为 ASCII 值数组，方便计算
        #s = [ord(c) for c in s]
        #t = [ord(c) for c in t]
        
        n, m = len(s), len(t)
        
        # 计算 s 和 t 反转的最长公共前缀长度 dp[i][j]
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                if s[i] == t[j]:
                    dp[i][j] = dp[i + 1][j + 1] + 1
        
        def compute_palindrome_prefix(x):
            """
            计算 x 数组的最长回文前缀长度 res[i]。
            res[i] 代表从索引 i 开始的子串的最长回文子串长度。
            """
            length = len(x)
            res = [0] * (length + 1)
            
            # 计算奇数长度回文
            for center in range(length):
                left = right = center #比如 "aba" 以 'b' 为中心。
                while left >= 0 and right < length and x[left] == x[right]:
                    res[left] = max(res[left], right - left + 1)
                    left -= 1
                    right += 1
            
            # 计算偶数长度回文
            for center in range(1, length):
                left, right = center - 1, center
                while left >= 0 and right < length and x[left] == x[right]:
                    res[left] = max(res[left], right - left + 1)
                    left -= 1
                    right += 1
            
            return res
        
        # 计算 s 和 t 的最长回文前缀数组
        palindrome_s = compute_palindrome_prefix(s)
        palindrome_t = compute_palindrome_prefix(t)
        
        # 计算独立于拼接的最大回文长度
        max_palindrome_length = max(max(palindrome_s), max(palindrome_t))
        
        # 遍历 s 和 t 的匹配位置，尝试拼接形成更长的回文串
        for i in range(n):
            for j in range(m):
                if dp[i][j] > 0:  # 只考虑有公共前缀的情况
                    common_length = dp[i][j]
                    remaining_s = palindrome_s[i + common_length]
                    remaining_t = palindrome_t[j + common_length]
                    max_palindrome_length = max(
                        max_palindrome_length, 
                        common_length * 2 + max(remaining_s, remaining_t)
                    )
        
        return max_palindrome_length

```



## End