import heapq

class MedianFinder:
    def __init__(self):
        # 最大堆（存储较小的一半，取反存入以模拟最大堆）
        self.left_heap = []
        # 最小堆（存储较大的一半）
        self.right_heap = []

    def addNum(self, num: int) -> None:
        # 先将 num 放入最大堆（但因为 Python 没有最大堆，我们存入负数来模拟）
        heapq.heappush(self.left_heap, -num)

        # 确保最大堆的最大值 ≤ 最小堆的最小值
        if self.left_heap and self.right_heap and (-self.left_heap[0] > self.right_heap[0]):
            heapq.heappush(self.right_heap, -heapq.heappop(self.left_heap))

        # 平衡两个堆的大小，使得最大堆的元素个数 ≥ 最小堆的元素个数
        if len(self.left_heap) > len(self.right_heap) + 1:
            heapq.heappush(self.right_heap, -heapq.heappop(self.left_heap))
        elif len(self.right_heap) > len(self.left_heap):
            heapq.heappush(self.left_heap, -heapq.heappop(self.right_heap))

    def findMedian(self) -> float:
        # 如果元素个数是奇数，中位数是最大堆的堆顶
        if len(self.left_heap) > len(self.right_heap):
            return -self.left_heap[0]
        # 如果元素个数是偶数，中位数是两个堆顶的平均值
        return (-self.left_heap[0] + self.right_heap[0]) / 2.0