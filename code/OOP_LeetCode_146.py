class DLinkedNode:
    """双向链表的节点类"""
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # 存储 key 到 DLinkedNode 的映射
        # 初始化双向链表
        self.head = DLinkedNode()  # 虚拟头节点
        self.tail = DLinkedNode()  # 虚拟尾节点
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: DLinkedNode):
        """从链表中移除节点"""
        prev = node.prev
        new = node.next
        prev.next = new
        new.prev = prev

    def _insert(self, node: DLinkedNode):
        """将节点插入到链表的头部"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key: int) -> int:
        """获取缓存中的值"""
        if key in self.cache:
            node = self.cache[key]
            # 移动到头部
            self._remove(node)
            self._insert(node)
            return node.value
        return -1

    def put(self, key: int, value: int) -> None:
        """插入/更新键值对"""
        if key in self.cache:
            # 如果键存在，先删除再插入，更新顺序
            node = self.cache[key]
            self._remove(node)
            node.value = value
            self._insert(node)
        else:
            # 如果键不存在，创建新节点
            node = DLinkedNode(key, value)
            self.cache[key] = node
            self._insert(node)
            # 如果超过容量，移除最久未使用的元素
            if len(self.cache) > self.capacity:
                # 移除链表尾部的元素，即最久未使用的
                tail = self.tail.prev
                self._remove(tail)
                del self.cache[tail.key]


if __name__ == "__main__":
    # 测试代码
    lRUCache = LRUCache(2)
    lRUCache.put(1, 1)
    lRUCache.put(2, 2)
    print(lRUCache.get(1))  # 返回 1
    lRUCache.put(3, 3)  # 该操作会使得关键字 2 作废
    print(lRUCache.get(2))  # 返回 -1 (未找到)
    lRUCache.put(4, 4)  # 该操作会使得关键字 1 作废
    print(lRUCache.get(1))  # 返回 -1 (未找到)
    print(lRUCache.get(3))  # 返回 3
    print(lRUCache.get(4))  # 返回 4
