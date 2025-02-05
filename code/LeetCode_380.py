import random

class RandomizedSet:
    def __init__(self):
        self.val_to_index = {}  # 值到索引的映射
        self.values = []        # 存储值的数组

    def insert(self, val: int) -> bool:
        if val in self.val_to_index:
            return False
        # 插入操作
        self.values.append(val)
        self.val_to_index[val] = len(self.values) - 1
        return True

    def remove(self, val: int) -> bool:
        if val not in self.val_to_index:
            return False
        # 删除操作
        last_element = self.values[-1]
        index = self.val_to_index[val]
        # 用最后一个元素替换删除的元素
        self.values[index] = last_element
        self.val_to_index[last_element] = index
        # 移除最后一个元素
        self.values.pop()
        del self.val_to_index[val]
        return True

    def getRandom(self) -> int:
        return random.choice(self.values)

if __name__ == "__main__":
    randomizedSet = RandomizedSet()
    print(randomizedSet.insert(1))  # 输出: True
    print(randomizedSet.remove(2))  # 输出: False
    print(randomizedSet.insert(2))  # 输出: True
    print(randomizedSet.getRandom())  # 输出: 1 或 2
    print(randomizedSet.remove(1))  # 输出: True
    print(randomizedSet.insert(2))  # 输出: False
    print(randomizedSet.getRandom())  # 输出: 2
