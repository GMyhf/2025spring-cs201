import numpy as np
import torch
import torch.nn as nn
import time
from DQN_train_2048 import Game2048Env, DQN # 复用训练脚本中的类

def play():
    env = Game2048Env()
    model = DQN()
    
    # 加载预训练的“大脑” [2]
    try:
        model.load_state_dict(torch.load('2048_bot.pth'))
        model.eval()
        print("成功加载高手模型，开始自动游戏...")
    except FileNotFoundError:
        print("未发现模型文件，请先运行训练脚本。")
        return

    state = env.reset()
    done = False
    
    while not done:
        # 清除终端屏幕，实现动态刷新效果 [2]
        print("\033[H\033[J", end="") 
        print(f"AI 正在操作... 当前最大数字: {env.board.max()}")
        
        # 显示棋盘 [3]
        print("-" * 22)
        for row in env.board:
            print("|" + "|".join(f"{val:4}" if val > 0 else "    " for val in row) + "|")
        print("-" * 22)
        
        # 高手决策：不再随机探索，直接取最优解
        with torch.no_grad():
            state_t = torch.FloatTensor(state)
            action = model(state_t).argmax().item()
        
        state, reward, done = env.step(action)
        time.sleep(0.2) # 减慢速度以便观察

    print(f"\n游戏结束！最终得分: {env.board.sum()}，最高方块: {env.board.max()}")

if __name__ == "__main__":
    play()
