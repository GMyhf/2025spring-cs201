import numpy as np
import torch
import torch.nn as nn
import time
import os


# 1. 神经网络结构 (必须与训练脚本 train_2048.py 完全一致: 256 -> 128)
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x): return self.net(x)


# 2. 2048 游戏环境：修复了垂直合并判定与边界死局检查
class Game2048Env:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)

    def reset(self, board=None):
        if board is not None:
            # 来源[1]：玩家在 4x4 的数字方格矩阵中操作
            self.board = np.array(board, dtype=int).copy()
        else:
            self.board = np.zeros((4, 4), dtype=int)
            self.add_new_tile()
            self.add_new_tile()
        return self.get_state()

    def get_state(self):
        # 来源建议：log2 处理可平滑状态分布，利于神经网络学习
        state = np.zeros_like(self.board, dtype=float)
        mask = self.board > 0
        state[mask] = np.log2(self.board[mask])
        return state.flatten().astype(np.float32)

    def add_new_tile(self):
        # 来源 [1]：按 90%、10% 概率随机生成 2 或 4
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            r, c = empty_cells[np.random.choice(len(empty_cells))]
            self.board[r, c] = 2 if np.random.random() < 0.9 else 4

    def test_move(self, action):
        """测试动作是否能改变棋盘 (解决卡死与提前结束的核心)"""
        old_board = self.board.copy()
        new_board, _ = self._move_logic(old_board, action)
        return not np.array_equal(self.board, new_board)

    def step(self, action):
        old_board = self.board.copy()
        self.board, _ = self._move_logic(self.board, action)
        if not np.array_equal(old_board, self.board):
            self.add_new_tile()
        # 来源 [1]：仅当局面无法进行任何滑动操作时游戏结束
        return self.get_state(), not self._global_can_move()

    def _global_can_move(self):
        """严谨检查全盘：是否有空位，或水平/垂直方向有相邻相等的数字"""
        if np.any(self.board == 0): return True
        for r in range(4):
            for c in range(4):
                # 检查水平相邻相等 (左右合并)
                if c < 3 and self.board[r, c] == self.board[r, c + 1]:
                    return True
                # 检查垂直相邻相等 (上下合并)
                if r < 3 and self.board[r, c] == self.board[r + 1, c]:
                    return True
        return False

    def _move_logic(self, b, action):
        """
        通过矩阵旋转实现四个方向的滑动逻辑。
        0:上, 1:下, 2:左, 3:右
        """
        temp_b = b.copy()
        if action == 0:  # Up (旋转使向上滑动变为向左合并)
            temp_b = np.rot90(temp_b, 1)
            nb, s = self._move_left(temp_b)
            return np.rot90(nb, -1), s
        elif action == 1:  # Down
            temp_b = np.rot90(temp_b, -1)
            nb, s = self._move_left(temp_b)
            return np.rot90(nb, 1), s
        elif action == 2:  # Left
            return self._move_left(temp_b)
        elif action == 3:  # Right
            temp_b = np.rot90(temp_b, 2)
            nb, s = self._move_left(temp_b)
            return np.rot90(nb, -2), s
        return temp_b, 0

    def _move_left(self, board):
        """来源 [2]：方块从前往后依次向前合并，且此次操作已经合并过的不能再次合并"""
        new_board = np.zeros((4, 4), dtype=int)
        score = 0
        for i in range(4):
            # 提取非零元素
            row = board[i][board[i] != 0]
            new_row, skip = [], False
            for j in range(len(row)):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(row) and row[j] == row[j + 1]:
                    new_row.append(row[j] * 2)
                    score += row[j] * 2
                    skip = True  # 标记已合并
                else:
                    new_row.append(row[j])
            new_board[i, :len(new_row)] = new_row
        return new_board, score


def render_board(board, is_game_over=False):
    """将清屏与打印逻辑提取为独立函数"""
    print("\033[H\033[J", end="")
    title = "游戏结束！" if is_game_over else "2048 AI 演示中..."
    print(f"{title} 当前最大数字: {int(board.max())}")
    print("-" * 22)
    for row in board:
        print("|" + "|".join(f"{val:4}" if val > 0 else "    " for val in row) + "|")
    print("-" * 22)


def play():
    env = Game2048Env()
    model = DQN()

    # 尝试加载模型权重 (来源 [3] 建议在 history 文件夹或根目录查找)
    model_path = '2048_bot.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print("成功加载模型，开始高手演示...")
        time.sleep(1)  # 给用户一点时间看到提示
    else:
        print("未发现模型文件。")
        return

    env.reset()
    done = False

    while not done:
        # 每步渲染当前盘面
        render_board(env.board)

        with torch.no_grad():
            q_values = model(torch.FloatTensor(env.get_state()))
            # 策略：按 Q 值高低排序，并强制校验合法性 (避免卡死逻辑)
            sorted_actions = torch.argsort(q_values, descending=True).tolist()

            best_action = None
            for a in sorted_actions:
                if env.test_move(a):  # 关键：只要该动作能令棋盘改变，就执行
                    best_action = a
                    break

            if best_action is None:
                # 理论上 _global_can_move 应该已经拦截，这行作为双重保险
                break

        _, done = env.step(best_action)
        # 延时以便看清过程，可以自行调整
        #time.sleep(0.3)

        # 循环结束后，额外打印一次最终的“死局盘面”，消除视觉误导
    render_board(env.board, is_game_over=True)
    print(f"\n游戏真正结束！最高方块: {int(env.board.max())}")


if __name__ == "__main__":
    play()
