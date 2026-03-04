import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random


# ======================================
# 1. 2048 环境
# ======================================

class Game2048Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((4,4), dtype=int)
        self.add_tile()
        self.add_tile()
        return self.get_state()

    def add_tile(self):
        empty = list(zip(*np.where(self.board == 0)))
        if empty:
            r,c = random.choice(empty)
            self.board[r,c] = 2 if random.random()<0.9 else 4

    def get_state(self):
        s = np.zeros_like(self.board, dtype=float)
        mask = self.board>0
        s[mask] = np.log2(self.board[mask])
        return s.flatten().astype(np.float32)

    def step(self, action):
        old = self.board.copy()
        score = 0

        if action==0: self.board,score=self._move_up(self.board)
        if action==1: self.board,score=self._move_down(self.board)
        if action==2: self.board,score=self._move_left(self.board)
        if action==3: self.board,score=self._move_right(self.board)

        moved = not np.array_equal(old,self.board)

        if moved:
            self.add_tile()
            reward = score
        else:
            reward = -1

        done = not self._can_move()
        return self.get_state(), reward, done

    def _can_move(self):
        if np.any(self.board==0): return True
        for r in range(4):
            for c in range(4):
                if (r<3 and self.board[r,c]==self.board[r+1,c]) or \
                   (c<3 and self.board[r,c]==self.board[r,c+1]):
                    return True
        return False

    def _move_left(self, board):
        new=np.zeros((4,4),dtype=int)
        score=0
        for i in range(4):
            row=board[i][board[i]!=0]
            new_row=[]
            skip=False
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
            new[i,:len(new_row)]=new_row
        return new,score

    def _move_up(self,b):
        nb,s=self._move_left(b.T)
        return nb.T,s

    def _move_down(self,b):
        nb,s=self._move_left(np.flip(b.T,1))
        return np.flip(nb.T,0),s

    def _move_right(self,b):
        nb,s=self._move_left(np.flip(b,1))
        return np.flip(nb,1),s


# ======================================
# 2. H-DQN 网络
# ======================================

GOALS = [32,64,128,256,512]

class MetaController(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16,128),
            nn.ReLU(),
            nn.Linear(128,len(GOALS))
        )

    def forward(self,x):
        return self.net(x)


class Controller(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16+1,128),
            nn.ReLU(),
            nn.Linear(128,4)
        )

    def forward(self,x):
        return self.net(x)


# ======================================
# 3. 训练 H-DQN
# ======================================

def train():

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print("Using device:",device)

    env = Game2048Env()

    meta = MetaController().to(device)
    controller = Controller().to(device)

    meta_opt = optim.Adam(meta.parameters(),lr=1e-4)
    ctrl_opt = optim.Adam(controller.parameters(),lr=1e-4)

    gamma = 0.99
    episodes=20000

    for ep in range(episodes):

        state = env.reset()
        done=False
        total_reward=0

        # ===== 高层选择目标 =====
        state_tensor=torch.from_numpy(state).unsqueeze(0).to(device)
        meta_logits = meta(state_tensor)
        goal_dist = torch.distributions.Categorical(logits=meta_logits)
        goal_idx = goal_dist.sample()
        goal = GOALS[goal_idx.item()]

        while not done:

            # 组合 state + goal
            goal_input = np.array([np.log2(goal)],dtype=np.float32)
            ctrl_input = np.concatenate([state,goal_input])
            ctrl_tensor=torch.from_numpy(ctrl_input).unsqueeze(0).to(device)

            logits=controller(ctrl_tensor)
            dist=torch.distributions.Categorical(logits=logits)
            action=dist.sample().item()

            next_state,reward,done=env.step(action)
            total_reward+=reward

            # ===== 内部奖励：是否达成目标 =====
            intrinsic_reward = 1.0 if env.board.max()>=goal else 0.0

            # Controller 更新
            ctrl_loss = -dist.log_prob(torch.tensor(action).to(device)) * intrinsic_reward
            ctrl_opt.zero_grad()
            ctrl_loss.backward()
            ctrl_opt.step()

            state=next_state

        # ===== Meta 更新（基于最终最大 tile） =====
        final_tile = env.board.max()
        meta_reward = 1.0 if final_tile>=goal else -1.0

        meta_loss = -goal_dist.log_prob(goal_idx) * meta_reward
        meta_opt.zero_grad()
        meta_loss.backward()
        meta_opt.step()

        if ep%100==0:
            print(f"Ep {ep}, MaxTile {env.board.max()}, Reward {total_reward}")

    print("Training finished")


if __name__=="__main__":
    train()
