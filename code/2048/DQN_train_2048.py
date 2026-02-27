import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# 1. 2048 环境逻辑
class Game2048Env:
    def __init__(self):
        self.size = 4
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.add_new_tile()
        self.add_new_tile()
        return self.get_state()

    def add_new_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            r, c = random.choice(empty_cells)
            self.board[r, c] = 2 if random.random() < 0.9 else 4 # 90%概率生成2 [3]

    def get_state(self):
        state = np.zeros_like(self.board, dtype=float)
        mask = self.board > 0
        state[mask] = np.log2(self.board[mask]) # log2平滑处理
        return state.flatten().astype(np.float32)

    def _move_left(self, board):
        new_board = np.zeros((4, 4), dtype=int)
        score = 0
        for i in range(4):
            row = board[i][board[i] != 0]
            new_row = []
            skip = False
            for j in range(len(row)):
                if skip:
                    skip = False; continue
                if j + 1 < len(row) and row[j] == row[j+1]:
                    new_row.append(row[j] * 2); score += row[j] * 2; skip = True
                else:
                    new_row.append(row[j])
            new_board[i, :len(new_row)] = new_row
        return new_board, score

    def step(self, action):
        old_board = self.board.copy()
        score = 0
        if action == 0: self.board, score = self._move_up(self.board)
        elif action == 1: self.board, score = self._move_down(self.board)
        elif action == 2: self.board, score = self._move_left(self.board)
        elif action == 3: self.board, score = self._move_right(self.board)
        
        moved = not np.array_equal(old_board, self.board)
        if moved: self.add_new_tile()
        reward = score if moved else -2 # 惩罚无效移动，参考Pong v5 [4]
        done = not self._can_move()
        return self.get_state(), reward, done

    def _can_move(self):
        if np.any(self.board == 0): return True
        for r in range(4):
            for c in range(4):
                if (r < 3 and self.board[r,c] == self.board[r+1,c]) or \
                   (c < 3 and self.board[r,c] == self.board[r,c+1]): return True
        return False

    def _move_up(self, b): nb, s = self._move_left(b.T); return nb.T, s
    def _move_down(self, b): nb, s = self._move_left(np.flip(b.T, 1)); return np.flip(nb.T, 0), s
    def _move_right(self, b): nb, s = self._move_left(np.flip(b, 1)); return np.flip(nb, 1), s

    def render(self):
        print("\033[H\033[J", end="")  # 清除终端屏幕以便观察
        print("-" * 20)
        for row in self.board:
            print("|" + "|".join(f"{val:4}" if val > 0 else "    " for val in row) + "|")
        print("-" * 20)

# 2. DQN 模型
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
    def forward(self, x): return self.net(x)

# 3. 训练主循环
def train():
    env = Game2048Env()
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    model = DQN().to(device)
    """
    1. 使用 Mac 的 GPU（通过 Metal Performance Shaders, 简称 MPS）理论上可以加快训练速度，但需要注意以下几点：
硬件支持：在搭载 Apple Silicon（M1/M2/M3 等芯片）的 Mac 上，PyTorch 支持使用 mps 设备来调用 GPU 加速。
适用场景：参考来源 提到，原项目在计算 Pong 参数时选择保护 CPU，是因为模型较小（只有 8 个参数）。目前的 2048 模型也相对简单（16 维输入，少量隐藏层），在这种规模下，CPU 训练已经很快。
如何开启：你需要将模型和数据搬运到 mps 设备上。代码示例如下：
注意：对于极小的网络，CPU 有时反而比 GPU 快，因为数据在 CPU 和 GPU 之间传输会有开销。如果之后你增加了网络深度（例如使用卷积层），GPU 的优势会非常明显。
    """


    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    memory = deque(maxlen=20000)
    batch_size = 64
    epsilon, gamma = 0.4, 0.99

    print("开始训练...")
    for episode in range(10000): # 训练10000轮
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).to(device)
                    action = model(state_t).argmax().item()
            
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            if len(memory) > batch_size:
                # 强化学习核心更新逻辑，参考 Pong v6-DQN [2]
                batch = random.sample(memory, batch_size)
                s_b, a_b, r_b, ns_b, d_b = zip(*batch)
                s_b = torch.FloatTensor(np.array(s_b)).to(device)
                a_b = torch.LongTensor(a_b).unsqueeze(1).to(device)
                r_b = torch.FloatTensor(r_b).to(device)
                ns_b = torch.FloatTensor(np.array(ns_b)).to(device)
                d_b = torch.FloatTensor(d_b).to(device)

                q_v = model(s_b).gather(1, a_b)
                next_q = model(ns_b).max(1).values.detach()
                target = r_b + (1 - d_b) * gamma * next_q
                loss = nn.MSELoss()(q_v.squeeze(), target)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        
        # epsilon = max(0.01, epsilon * 0.997)
        # if episode % 100 == 0:
        #     print(f"Episode {episode}, Max Tile: {env.board.max()}")

        epsilon = max(0.01, epsilon * 0.999)  # 缓慢衰减 epsilon 以保证训练稳定性 [2]

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward}, Max Tile: {env.board.max()}, Epsilon: {epsilon:.3f}")
            if episode % 100 == 0: env.render()

    # 训练结束保存模型 [2]
    torch.save(model.state_dict(), '2048_bot.pth')
    print("模型已保存为 2048_bot.pth")

if __name__ == "__main__":
    train()
