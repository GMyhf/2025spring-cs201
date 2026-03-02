import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


# ==============================
# 1. 2048 环境
# ==============================

class Game2048Env:
    def __init__(self):
        self.size = 4
        self.reset()

    def reset(self):
        self.board = np.zeros((4,4), dtype=int)
        self.add_new_tile()
        self.add_new_tile()
        return self.get_state()

    def add_new_tile(self):
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            r,c = random.choice(empty)
            self.board[r,c] = 2 if random.random() < 0.9 else 4

    def get_state(self):
        s = np.zeros_like(self.board, dtype=float)
        mask = self.board > 0
        s[mask] = np.log2(self.board[mask])
        return s.flatten().astype(np.float32)

    def step(self, action):
        old = self.board.copy()
        score = 0

        if action == 0: self.board, score = self._move_up(self.board)
        if action == 1: self.board, score = self._move_down(self.board)
        if action == 2: self.board, score = self._move_left(self.board)
        if action == 3: self.board, score = self._move_right(self.board)

        moved = not np.array_equal(old, self.board)

        if moved:
            self.add_new_tile()
            reward = score + 1.0
        else:
            reward = -1.0   # ⭐ 降低惩罚

        done = not self._can_move()
        return self.get_state(), reward, done

    def _can_move(self):
        if np.any(self.board == 0):
            return True
        for r in range(4):
            for c in range(4):
                if (r<3 and self.board[r,c]==self.board[r+1,c]) or \
                   (c<3 and self.board[r,c]==self.board[r,c+1]):
                    return True
        return False

    def _move_left(self, board):
        new = np.zeros((4,4), dtype=int)
        score = 0
        for i in range(4):
            row = board[i][board[i]!=0]
            new_row = []
            skip = False
            for j in range(len(row)):
                if skip:
                    skip=False
                    continue
                if j+1<len(row) and row[j]==row[j+1]:
                    val=row[j]*2
                    new_row.append(val)
                    score+=np.log2(val)
                    skip=True
                else:
                    new_row.append(row[j])
            new[i,:len(new_row)] = new_row
        return new, score

    def _move_up(self,b):
        nb,s=self._move_left(b.T)
        return nb.T,s

    def _move_down(self,b):
        nb,s=self._move_left(np.flip(b.T,1))
        return np.flip(nb.T,0),s

    def _move_right(self,b):
        nb,s=self._move_left(np.flip(b,1))
        return np.flip(nb,1),s


# ==============================
# 2. Actor-Critic 网络
# ==============================

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(16,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128,4)
        self.critic = nn.Linear(128,1)

    def forward(self,x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


# ==============================
# 3. 极简稳定训练
# ==============================

def train():

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:", device)

    env = Game2048Env()
    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    gamma = 0.95     # ⭐ 降低 horizon
    entropy_coef = 0.01

    episodes = 20000

    for ep in range(episodes):

        states = []
        actions = []
        rewards = []

        state = env.reset()
        done = False
        total_reward = 0

        # ===== 收集一整局（不保留计算图） =====
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                logits,_ = model(state_tensor)
                probs = torch.softmax(logits,dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()

            next_state,reward,done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            total_reward+=reward
            state=next_state

        # ===== 统一 forward =====
        states_tensor = torch.FloatTensor(states).to(device)
        logits, values = model(states_tensor)

        probs = torch.softmax(logits,dim=-1)
        dist = torch.distributions.Categorical(probs)

        actions_tensor = torch.LongTensor(actions).to(device)

        # ===== 计算多步 return =====
        returns=[]
        R=0
        for r in reversed(rewards):
            R = r + gamma*R
            returns.insert(0,R)

        returns = torch.FloatTensor(returns).to(device)

        # ⭐ 标准化，极其重要
        returns = (returns - returns.mean()) / (returns.std()+1e-8)

        advantages = returns - values.squeeze()

        # ===== Loss =====
        actor_loss = -(dist.log_prob(actions_tensor) * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy = dist.entropy().mean()

        loss = actor_loss + 0.5*critic_loss - entropy_coef*entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),3.0)
        optimizer.step()

        if ep%100==0:
            print(f"Ep {ep}, MaxTile {env.board.max()}, Reward {total_reward:.1f}")

    torch.save(model.state_dict(),"2048_A2C_stable.pth")
    print("Training finished.")


if __name__=="__main__":
    train()