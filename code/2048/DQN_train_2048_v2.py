import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque

# 1. 强化环境逻辑 (增加动作合法性判断与奖励优化)
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
        # 使用 log2 归一化状态，避免神经网络输入数值过大 [3]
        state = np.zeros_like(self.board, dtype=float)
        mask = self.board > 0
        state[mask] = np.log2(self.board[mask])
        return state.flatten().astype(np.float32)

    def step(self, action):
        old_board = self.board.copy()
        score = 0
        # 执行滑动操作
        if action == 0: self.board, score = self._move_up(self.board)
        elif action == 1: self.board, score = self._move_down(self.board)
        elif action == 2: self.board, score = self._move_left(self.board)
        elif action == 3: self.board, score = self._move_right(self.board)

        moved = not np.array_equal(old_board, self.board)
        
        # --- 奖励函数优化 (参考 Pong v5 经验 [1]) ---
        if moved:
            self.add_new_tile()
            # 1. 合并奖励: 鼓励产生更大的方块
            # 2. 存活奖励: 只要能动就给 1 分，鼓励增加游戏时长
            reward = score + 1.0 
        else:
            # 3. 惩罚无效移动: 防止 AI 卡死在原地做无效动作 [1]
            reward = -10.0 
        
        done = not self._can_move()
        return self.get_state(), reward, done

    def _can_move(self):
        if np.any(self.board == 0): return True
        for r in range(4):
            for c in range(4):
                if (r < 3 and self.board[r,c] == self.board[r+1,c]) or \
                   (c < 3 and self.board[r,c] == self.board[r,c+1]): return True
        return False

    def _move_left(self, board):
        new_board = np.zeros((4, 4), dtype=int)
        score = 0
        for i in range(4):
            row = board[i][board[i] != 0]
            new_row = []; skip = False
            for j in range(len(row)):
                if skip: skip = False; continue
                if j+1 < len(row) and row[j] == row[j+1]:
                    new_row.append(row[j]*2); score += np.log2(row[j]*2); skip = True
                else: new_row.append(row[j])
            new_board[i, :len(new_row)] = new_row
        return new_board, score

    def _move_up(self, b): nb, s = self._move_left(b.T); return nb.T, s
    def _move_down(self, b): nb, s = self._move_left(np.flip(b.T, 1)); return np.flip(nb.T, 0), s
    def _move_right(self, b): nb, s = self._move_left(np.flip(b, 1)); return np.flip(nb, 1), s

# 2. DQN 神经网络 (略微增加深度以处理复杂棋局)
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )
    def forward(self, x): return self.net(x)

# 3. 核心训练逻辑优化
def train():
    # 配置设备: 优先使用 Mac GPU (MPS) [2]
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"正在使用设备: {device}")

    env = Game2048Env()
    model = DQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # 降低学习率以求稳定
    
    # 参考 Pong v6: 将 Buffer 从 1w 提高到 10w [2]
    memory = deque(maxlen=100000) 
    
    # 策略控制参数
    epsilon = 0.9      # 初始高探索
    epsilon_min = 0.01 # 参考经验：后期 epsilon 过大会导致表现不稳定 [2]
    gamma = 0.99
    batch_size = 128   # 增大 Batch 以提高收敛稳定性
    
    # 增加训练量: 至少 50,000 回合以上
    total_episodes = 50000 
    
    for ep in range(total_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    st = torch.FloatTensor(state).to(device)
                    action = model(st).argmax().item()
            
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                s_b, a_b, r_b, ns_b, d_b = zip(*batch)
                
                s_b = torch.FloatTensor(np.array(s_b)).to(device)
                a_b = torch.LongTensor(a_b).unsqueeze(1).to(device)
                r_b = torch.FloatTensor(r_b).to(device)
                ns_b = torch.FloatTensor(np.array(ns_b)).to(device)
                d_b = torch.FloatTensor(d_b).to(device)

                q_v = model(s_b).gather(1, a_b)
                # 修复 .max(1).values 报错
                next_q = model(ns_b).max(1).values.detach()
                target = r_b + (1 - d_b) * gamma * next_q
                
                loss = nn.MSELoss()(q_v.squeeze(), target)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        # 缓慢衰减 epsilon，确保前期充分学习规则，后期稳定策略
        epsilon = max(epsilon_min, epsilon * 0.9995)
        
        if ep % 100 == 0:
            print(f"Ep: {ep}, Max: {env.board.max()}, Reward: {total_reward:.1f}, Eps: {epsilon:.3f}")
            
        # 每 1000 回合保存快照，防止训练意外中断 [4]
        if ep % 1000 == 0:
            torch.save(model.state_dict(), '2048_bot.pth')

    torch.save(model.state_dict(), '2048_bot.pth')
    print("训练完成，模型已保存。")

if __name__ == "__main__":
    train()
