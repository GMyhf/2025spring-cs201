import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# =============================
# 1. 2048 环境（保持不变）
# =============================

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
            self.board[r, c] = 2 if random.random() < 0.9 else 4

    def get_state(self):
        state = np.zeros_like(self.board, dtype=float)
        mask = self.board > 0
        state[mask] = np.log2(self.board[mask])
        return state.flatten().astype(np.float32)

    def step(self, action):
        old_board = self.board.copy()
        score = 0

        if action == 0:
            self.board, score = self._move_up(self.board)
        elif action == 1:
            self.board, score = self._move_down(self.board)
        elif action == 2:
            self.board, score = self._move_left(self.board)
        elif action == 3:
            self.board, score = self._move_right(self.board)

        moved = not np.array_equal(old_board, self.board)

        if moved:
            self.add_new_tile()
            reward = score + 1.0
        else:
            reward = -2.0

        done = not self._can_move()
        return self.get_state(), reward, done

    def _can_move(self):
        if np.any(self.board == 0):
            return True
        for r in range(4):
            for c in range(4):
                if (r < 3 and self.board[r,c] == self.board[r+1,c]) or \
                   (c < 3 and self.board[r,c] == self.board[r,c+1]):
                    return True
        return False

    def _move_left(self, board):
        new_board = np.zeros((4, 4), dtype=int)
        score = 0
        for i in range(4):
            row = board[i][board[i] != 0]
            new_row = []
            skip = False
            for j in range(len(row)):
                if skip:
                    skip = False
                    continue
                if j+1 < len(row) and row[j] == row[j+1]:
                    new_row.append(row[j]*2)
                    score += np.log2(row[j]*2)
                    skip = True
                else:
                    new_row.append(row[j])
            new_board[i, :len(new_row)] = new_row
        return new_board, score

    def _move_up(self, b):
        nb, s = self._move_left(b.T)
        return nb.T, s

    def _move_down(self, b):
        nb, s = self._move_left(np.flip(b.T, 1))
        return np.flip(nb.T, 0), s

    def _move_right(self, b):
        nb, s = self._move_left(np.flip(b, 1))
        return np.flip(nb, 1), s


# =============================
# 2. Actor-Critic 网络
# =============================

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Actor 头
        self.actor = nn.Linear(128, 4)
        
        # Critic 头
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


# =============================
# 3. 训练函数
# =============================

def train():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    env = Game2048Env()
    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    gamma = 0.95
    total_episodes = 50000

    for ep in range(total_episodes):

        states = []
        actions = []
        rewards = []
        values = []
        dones = []

        state = env.reset()
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits, value = model(state_tensor)
            probs = torch.softmax(logits, dim=-1)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, done = env.step(action.item())

            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

            state = next_state

        # ===== 计算多步回报 =====
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + gamma * R * (1 - d)
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # ⭐ 关键：标准化
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        values = torch.cat(values).squeeze()
        advantages = returns - values.detach()

        # ===== Loss =====
        logits_batch, _ = model(torch.cat(states))
        probs_batch = torch.softmax(logits_batch, dim=-1)
        dist_batch = torch.distributions.Categorical(probs_batch)

        log_probs = dist_batch.log_prob(torch.stack(actions))
        entropy = dist_batch.entropy().mean()

        actor_loss = -(log_probs * advantages).mean()
        critic_loss = (returns - values).pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if ep % 100 == 0:
            print(f"Ep {ep}, MaxTile {env.board.max()}, Reward {sum(rewards)}")


        if ep % 1000 == 0:
            torch.save(model.state_dict(), "2048_actor_critic.pth")

    torch.save(model.state_dict(), "2048_actor_critic.pth")
    print("Training finished.")


if __name__ == "__main__":
    train()
