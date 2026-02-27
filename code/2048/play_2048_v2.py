import numpy as np
import torch
import torch.nn as nn
import time
import random

# 1. 必须与 train_2048.py 中的定义完全一致
class Game2048Env:
    def __init__(self):
        self.size = 4
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.add_new_tile(); self.add_new_tile()
        return self.get_state()

    def add_new_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.board[r, c] = 2 if random.random() < 0.9 else 4

    def get_state(self):
        state = np.zeros_like(self.board, dtype=float)
        mask = self.board > 0
        state[mask] = np.log2(self.board[mask])
        return state.flatten().astype(np.float32)

    def step(self, action):
        old_board = self.board.copy()
        if action == 0: self.board, _ = self._move_up(self.board)
        elif action == 1: self.board, _ = self._move_down(self.board)
        elif action == 2: self.board, _ = self._move_left(self.board)
        elif action == 3: self.board, _ = self._move_right(self.board)
        if not np.array_equal(old_board, self.board): self.add_new_tile()
        return self.get_state(), 0, not self._can_move()

    def _can_move(self):
        if np.any(self.board == 0): return True
        for r in range(4):
            for c in range(4):
                if (r < 3 and self.board[r,c] == self.board[r+1,c]) or \
                   (c < 3 and self.board[r,c] == self.board[r,c+1]): return True
        return False

    def _move_left(self, board):
        new_board = np.zeros((4, 4), dtype=int)
        for i in range(4):
            row = board[i][board[i] != 0]
            new_row = []; skip = False
            for j in range(len(row)):
                if skip: skip = False; continue
                if j+1 < len(row) and row[j] == row[j+1]:
                    new_row.append(row[j]*2); skip = True
                else: new_row.append(row[j])
            new_board[i, :len(new_row)] = new_row
        return new_board, 0

    def _move_up(self, b): nb, s = self._move_left(b.T); return nb.T, s
    def _move_down(self, b): nb, s = self._move_left(np.flip(b.T, 1)); return np.flip(nb.T, 0), s
    def _move_right(self, b): nb, s = self._move_left(np.flip(b, 1)); return np.flip(nb, 1), s

# 2. 这里的参数必须是 256, 128 (对应你报错信息中的 shape)
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 256),  # 修改点
            nn.ReLU(),
            nn.Linear(256, 128), # 修改点
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    def forward(self, x): return self.net(x)

def play():
    env = Game2048Env()
    model = DQN()
    
    # 加载已训练的模型
    try:
        # map_location 可以确保在没有 GPU 的情况下也能加载 MPS 训练的模型
        checkpoint = torch.load('2048_bot.pth', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        model.eval()
        print("成功加载优化后的高手模型...")
    except Exception as e:
        print(f"加载失败: {e}")
        return

    state = env.reset()
    done = False
    while not done:
        print("\033[H\033[J", end="") 
        print(f"AI 正在展示... 当前最大数字: {int(env.board.max())}")
        print("-" * 22)
        for row in env.board:
            print("|" + "|".join(f"{val:4}" if val > 0 else "    " for val in row) + "|")
        print("-" * 22)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state)
            action = model(state_t).argmax().item()
        
        state, _, done = env.step(action)
        time.sleep(0.3)

    print(f"\n演示结束！最高方块: {int(env.board.max())}")

if __name__ == "__main__":
    play()
