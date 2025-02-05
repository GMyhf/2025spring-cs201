from typing import List
class ATM:

    def __init__(self):
        self.cnt = [0] * 5
        self.value = [20, 50, 100, 200, 500]

    def deposit(self, banknotesCount: List[int]) -> None:
        for i in range(len(banknotesCount)):
            self.cnt[i] += banknotesCount[i]

    def withdraw(self, amount: int) -> List[int]:
        res = [0] * 5
        for i in range(4, -1, -1):
            res[i] = min(self.cnt[i], amount // self.value[i])
            amount -= res[i] * self.value[i]
        if amount == 0:
            for i in range(5):
                self.cnt[i] -= res[i]
            return res
        else:
            return [-1]

# Your ATM object will be instantiated and called as such:
if __name__ == '__main__':
    obj = ATM()
    obj.deposit([0,0,1,2,1])
    param_2 = obj.withdraw(600)
    print(param_2)
