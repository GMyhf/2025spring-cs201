# 强化学习，2048游戏

*Updated 2026-02-27 17:47 GMT+8*
 *Compiled by Hongfei Yan (2026 Spring)*



​	选择2048游戏作为教学案例。2048 游戏是 2014 年由 Gabriele Cirulli 推出的单人数字游戏。设计 2048 AI 的算法思路与技巧在游戏算法设计中很具有代表性，且 2048 游戏规则简单，在探索解法的过程中充满趣味性，是一个非常适合作为计算机科学方面相关教学的益智游戏 。本节以问题求解为导向，以流行的单人数字游戏 2048 的 AI 算法设计问题为主线，试图在剖析问题、提出思路、拓展延伸等完整的问题求解过程中激发学习者能动性，使学习者对游戏 AI 算法的设计思路和常见算法达到较高的认知水平。

​	2048 游戏界面如图5-23所示，玩家需要在 $4\times4$ 的数字方格矩阵中做上、下、左、右四个方向之一的滑动操作。每次滑动操作会使得所有方块全部沿着某个方向滑动直至被边界或其他方块阻挡。当两个数字方格中的数字相同，且滑动后两方格相邻，则这两个数字方格会被合并为一个方格，其数字为原来两方块之和，如图5-24所示。执行上滑操作后，最左列的两个 2 合并为 4，并上移至第二行；左数第三列的上方两个相邻 8 合并为 16 并上移；最右一行的上下两对相邻的 2 均合并为 4，并上移至第1和第 2 行，这一列值得注意的是，该次操作合并过的两个方块 4 并不会继续合并。每次滑动操作后，系统随机在一个没有数字的方格中按照分别 90%、10% 的概率随机生成 2 或 4 两种之一的数字方格。游戏的初始局面为在空棋盘上按上述规则随机生成两个方块后的棋盘。游戏目标是产生数字尽可能大的方块。为便于之后测试算法效率，本书设计的 2048 游戏任务无方块数字的上限，达到 2048 后游戏仍然继续，当且仅当局面无法进行任何滑动操作时游戏结束。



<img src="https://raw.githubusercontent.com/GMyhf/img/main/img/image-20230109211246988.png" alt="image-20230109211246988" style="zoom:50%;" />

<center>图 2048 游戏初始界面</center>



![image-20230109211322682](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20230109211322682.png)

<center>图 一次向上滑动示意图，右图左下侧的2为系统生成的新方块</center>



​	2048 游戏规则的具体实现有一个易错之处：在同行或者同列中进行某方向的滑动时，这行或这列中的方块从前往后（记滑动方向的一侧为前）依次向前合并，且此次操作已经合并过的方块不能再次合并。与游戏相关的题目，包括2048的游戏规则模拟（http://cs101.openjudge.cn/practice/20052/），和一个需要深入挖掘2048规则性质的动态规划问题（http://codeforces.com/problemset/problem/413/D），帮助学生深入理解游戏规则。在线体验 2048 游戏的网页有： https://play2048.co/ ，https://2048game.com 。



请参照 https://github.com/DawnTilDusk/RL-pong，是几个强化学习的实例，都用自己创建的一个乒乓球小游戏的环境在cpu上面跑通过。

> # Pong RL Training Repository
>
> 
>
> 这是一个包含多种强化学习算法（Q-learning, DQN, A2C）用于训练 Pong 游戏智能体的仓库。本项目提供了完整的训练环境、算法实现以及可视化工具。
>
> ## 一、自叙
>
> 
>
> - 这个仓库是作为笨人练习RL使用的，包含了从基础的Q-learning到复杂的A2C的实现。
> - 实际上是学习[动手学强化学习](https://hrl.boyuai.com/chapter/1/初探强化学习)这本书的一个实践
> - 欢迎大家参考，探讨RL相关的一些算法原理以及训练的心得
> - 每个版本都有详细的README.md，说明其核心特点、实现原理、代码架构以及快速开始指南。
>
> ### 1. 环境
>
> 
>
> - 其实本来使用openai自带的gym环境也是可行的，但是出于懒得学习接口，避免因为调用接口涉及到的一系列顺序问题，以及对“纯粹性”的执着，笨人决定自己用pygame实现一个环境，也同时体验了一下pygame的常用接口。
> - 我创建了一个规则机器人Bot_0作为陪练（限速平移板，上帝视角，板心总是跟着球心走）
> - 右侧就是我们要训练的agent（初始时智障），他会通过各种算法来学习如何玩pong游戏
> - 乒乓球的速度会随着回合数越来越快，直到有一方空拍，reset
> - **输入向量**是当前对局的参数，包括两个板的位置xy、速度xy，球的位置xy、速度xy（这八个参数可以确定一张图，所以我并没有加入复杂的卷积机制，保护cpu人人有责）
> - 关于**输出向量**，我模拟了一个三入口的控制函数，分别是向上、向下、松手，对应的动作是1、-1、0，
> - 这里长按上或者下会提供一个相应方向的加速度，直至达到速度上限
> - 不移动则提供运动方向相反的加速度，直至速度为零
>
> ### 2. 版本迭代
>
> 
>
> #### v1-v5都是q-learning算法
>
> 
>
> 基于q数组的离散更新，每个状态-动作对都是一个高维的键值，对应一个q值，通过更新q值来学习最优策略。
>
> - v1-v2
>   - 其中，v1-v2是初始版本，环境甚至都有问题，物理特性和奖励函数就更别提了
> - v3
>   - v3是基于v2的改进版本，修复了移动的bug，同时引入了更优的奖励函数，使得训练更加稳定，但是一个更加严重的问题暴露出来了，就是训练过程中智能体会出现贴底现象，它倾向于向下运动，导致训练很久才开始有收敛的迹象。原因是在创建q表的时候没有赋随机值，导致argmax在开始阶段总是返回第一个下标（0，对应向下这个动作），所以会有贴底倾向
>   - v3的第二个问题是q表的输入维数不合理，刚开始的q表是离散的完整的8维数组，这样智能体可以根据任意一个状态找到最佳动作（上、下、停），但是因为我的陪练机器人会尽量保持板心对准球心，所以q表有一个维度（或者说自由度）几乎没有被激活，导致智能体并没有学习到一个有效的策略
>   - v3的胜率在一夜的cpu训练之后也没有超过35%
> - v4
>   - 开始着手v4的时候，我先让环境变得能允许更多策略出现，于是加入了动摩擦因数mu，这样板在碰撞球时的速度会附加在球的身上，从而打出类似削球的玩法
>   - 之后我修复了q表，一个是降维，另一个是随机初始化，给每个q值赋一个随机值，使得argmax在开始阶段有更多的选择，避免贴底现象，但是效果依然不算理想，训练了半天可能还是37%左右徘徊，甚至过训练之后胜率反而降低到10%以下
>   - 我的猜想还是因为左侧的规则机器人Bot_0只追逐板心，导致当球速过快的
>   - 怎么办呢？
>   - 我灵光一现，我可以尝试一下不只训练智能体，而是一起训练陪练机器人，看看效果会不会好一些
>   - 所以我给所有的方法加入了side参数，所有的参数加上了side维度，用于训练左侧的机器人
>   - 然而就像两个齐头并进（完全一样）的满月双胞胎肯定不能从对话中学习到好的策略，因为我们作为大人只能根据环境来设置奖励或者惩罚，于是乎我人为的区分了哥哥和弟弟，哥哥（right，主要的训练对象）的学习速度是弟弟（left）学习速度的两倍
>   - 我希望借助这种差异化的学习速度，来给他们更优秀的策略
> - v4-双bot
>   - 然鹅希望美好，现实残酷，训练了半天，哥哥的胜率接近70%，弟弟的胜率只有30%多，这说明他们的策略完全不同，而且哥哥确实强，但是他们的策略都不是最优的，
>   - 百分比的数据完全失去了它评判智能体优劣的作用，**纯粹只是两个智障在自娱自乐罢了**
>   - 改进：既然弟弟（左侧智能体）是智障，那么适当的让厉害的老师（规则机器人）重新出现，不妨是一个好办法
>   - 于是我让左侧机器人每100回合在“弟弟”和”老师“替换一次，右侧智能体持续训练，效果虽然更好（与规则机器人对打45%左右胜率），但是依然没有达到我的预期
> - v5
>   - 我这个时候觉得一定是奖励函数的问题，因为我写calculate_reward函数的时候看似深思熟虑，实则非常随意，相差巨大，而且没有逐帧奖励（存活奖励）
>   - 在与llm的一番交涉之后，我重新设计了一款奖励函数
>   - 关闭render之后训练了35万回合，与规则机器人对打的胜率达到了70%以上，
>   - 切换成render模式后我手动和它对打，**终于不像是与智障玩游戏了**
>   - 这说明我的奖励函数设计是合理的，只是cpu性能不够，训练时间不够，离散空间的q表无法收敛到最优策略
>
> #### v6-DQN/value-based
>
> 
>
> 从离散到连续，q表也就变成了神经网络
>
> - epsilon初始0.4，epsilon_min为0.001初始100多万回合效果不好，q一直不收敛，甚至出现发散，胜率从20慢慢回退至2%，几乎没用，初步判断epsilon过大（在0.08以上）都会使模型表现不稳定，而pong游戏取胜需要稳定的表现
> - 关闭之后重启训练，在10万步之后，epsilon很快趋近0.001，胜率开始迅速增加，在40万步左右达到70%，之后开始下降，于是结束训练，使用峰值的q表
> - 之后将epsilon_min调至0，将replay buffer从1万调至十万，继续训练，35万步之后，模型表现良好，胜率增长到了85%附近，此时epsilon约等于0.0002
> - 这应该是一个非常好的数据了
>
> #### v7-reinforce/actor-to-critic/policy-based
>
> 
>
> - policy-based属于主流RL算法，使用起来果然非常有效
> - 同一个奖励函数，虽然单回合的训练速率下降了，但是10w回合之后胜率就到达了100%并且能够保持，
> - 切换为手动模式也很难打过（即使陪练的智能体依然是单纯的规则机器人Bot_0）
>
> ## 二、整体结构
>
> 
>
> 仓库主要包含三个核心版本的环境与算法实现：
>
> - **pongenv-v1-5-qlearning**: 基于 Q-table 的 Q-learning 算法实现。适用于理解强化学习基础概念。
> - **pongenv-v6-DQN**: 基于 PyTorch 的 Deep Q-Network (DQN) 实现。引入了神经网络、经验回放等机制，支持连续状态空间。
> - **pongenv-v7-A2C**: 基于 PyTorch 的 Advantage Actor-Critic (A2C) 实现。使用策略网络与价值网络，支持更高效的训练。
>
> ## 三、快速开始
>
> 
>
> ### 1. 安装依赖
>
> 
>
> 请确保已安装 Python 3.13 及以下依赖库（详细列表见 `requirements.txt`）：
>
> ```
> pip install -r requirements.txt
> ```
>
> 
>
> 主要依赖包括：`pygame`, `numpy`, `torch`, `matplotlib`。
>
> ### 2. 运行环境
>
> 
>
> 进入对应版本的目录，运行主脚本即可开始训练或观看演示。
>
> **Q-learning 版本:**
>
> ```
> cd pongenv-v1-5-qlearning
> python pongenv-v3.py
> ```
>
> 
>
> **DQN 版本:**
>
> ```
> cd pongenv-v6-DQN
> python pong_bot_DQN.py
> ```
>
> 
>
> **A2C 版本:**
>
> ```
> cd pongenv-v7-A2C
> python pong_bot_A2C.py
> ```
>
> 
>
> ## 四、版本概览与入口
>
> 
>
> | 版本     | 算法       | 目录                     | 入口脚本          | 核心特点                                                     |
> | -------- | ---------- | ------------------------ | ----------------- | ------------------------------------------------------------ |
> | **v1-5** | Q-learning | `pongenv-v1-5-qlearning` | `pongenv-v3.py`   | 离散状态空间，使用 Q 表存储，包含基础可视化。                |
> | **v6**   | DQN        | `pongenv-v6-DQN`         | `pong_bot_DQN.py` | 连续状态空间，神经网络近似 Q 值，支持经验回放与 Target Network，HUD 实时统计。 |
> | **v7**   | A2C        | `pongenv-v7-A2C`         | `pong_bot_A2C.py` | Actor-Critic 架构，同时训练策略与价值网络，支持 On-policy 更新。 |
>
> ## 五、保存地址及说明
>
> 
>
> 训练过程中的模型权重、Q 表以及历史统计数据默认保存在各版本目录下的 `history` 文件夹或根目录中。
>
> - **模型文件**: 通常命名为 `dqn.pth` (DQN), `A2C.pth` (A2C), 或 `q_table-v*.pkl` (Q-learning)。
> - **历史记录**:
>   - DQN 和 A2C 版本会自动在当前脚本所在目录下的 `history` 文件夹中创建备份（已配置为相对路径）。
>   - 备份内容包括：带时间戳的模型权重文件、训练曲线图表等。
>
> ## 六、注意事项
>
> 
>
> - 默认开启渲染，如果要大量训练，在入口文件中注释掉render
>   - 例如： [pong_bot_DQN.py env.render()](https://github.com/DawnTilDusk/RL-pong/blob/main/pongenv-v6-DQN/pong_bot_DQN.py#L628)）
> - 运行时请确保当前目录下包含 `bg.png` 背景图片文件，否则环境将使用纯色背景。
> - 部分版本支持按键交互（如 `F2` 切换人机/自动模式，`F4` 手动保存快照），具体请参考各版本的 README 说明：
>   - [v1-5-qlearning/README.md](https://github.com/DawnTilDusk/RL-pong/blob/main/pongenv-v1-5-qlearning/README.md)
>   - [v6-DQN/README.md](https://github.com/DawnTilDusk/RL-pong/blob/main/pongenv-v6-DQN/README.md)
>   - [v7-A2C/README.md](https://github.com/DawnTilDusk/RL-pong/blob/main/pongenv-v7-A2C/README.md)
> - 项目远程仓库：[RL-pong](https://github.com/DawnTilDusk/RL-pong.git)。

帮我改造这个 1_game_2048_gui.py, 用上强化学习算法，在Mac terminal中可以运行的 2048 游戏。





```python
#!/usr/bin/env python3
import pygame
import random
import sys

# ==========================================
# 1. 游戏配置与常量定义
# ==========================================
BOARD_SIZE = 4
TARGET = 2048
WINDOW_SIZE = 500
GRID_SIZE = 400
TILE_MARGIN = 10
TILE_SIZE = (GRID_SIZE - (BOARD_SIZE + 1) * TILE_MARGIN) // BOARD_SIZE

# 颜色配置 (RGB)
COLOR_BG = (187, 173, 160)
COLOR_EMPTY = (205, 193, 180)
COLOR_TEXT_DARK = (119, 110, 101)
COLOR_TEXT_LIGHT = (249, 246, 242)

TILE_COLORS = {
    2: (238, 228, 218), 4: (237, 224, 200), 8: (242, 177, 121),
    16: (245, 149, 99), 32: (245, 124, 95), 64: (246, 94, 59),
    128: (237, 207, 114), 256: (237, 204, 97), 512: (237, 200, 80),
    1024: (237, 197, 63), 2048: (237, 194, 46),
}


# ==========================================
# 2. 核心逻辑函数
# ==========================================

def init_board():
    """初始化棋盘"""
    board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    add_random_tile(board)
    add_random_tile(board)
    return board


def add_random_tile(board):
    """随机增加方块"""
    empty = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i][j] == 0]
    if not empty: return
    i, j = random.choice(empty)
    board[i][j] = 4 if random.random() < 0.1 else 2


def slide_and_merge(line):
    """处理单行的滑动和合并"""
    new_line = [num for num in line if num != 0]
    merged_line = []
    skip = False
    for i in range(len(new_line)):
        if skip:
            skip = False
            continue
        if i + 1 < len(new_line) and new_line[i] == new_line[i + 1]:
            merged_line.append(new_line[i] * 2)
            skip = True
        else:
            merged_line.append(new_line[i])
    merged_line += [0] * (BOARD_SIZE - len(merged_line))
    return merged_line


# 矩阵变换逻辑
def move_left(board): return [slide_and_merge(row) for row in board]


def reverse(board): return [list(reversed(row)) for row in board]


def transpose(board): return [list(row) for row in zip(*board)]


def move_right(board): return reverse(move_left(reverse(board)))


def move_up(board): return transpose(move_left(transpose(board)))


def move_down(board): return transpose(move_right(transpose(board)))


def can_move(board):
    """检查是否还能移动"""
    for row in board:
        if 0 in row: return True
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if j + 1 < BOARD_SIZE and board[i][j] == board[i][j + 1]: return True
            if i + 1 < BOARD_SIZE and board[i][j] == board[i + 1][j]: return True
    return False


# ==========================================
# 3. 优化后的图形渲染函数
# ==========================================

def draw_game(screen, board, fonts, msg=""):
    """
    负责绘图。注意：fonts 现在是预先创建好的字典，不再重复创建。
    """
    # 1. 刷底色
    screen.fill((250, 248, 239))

    # 2. 画背景大底盘
    offset = (WINDOW_SIZE - GRID_SIZE) // 2
    board_rect = pygame.Rect(offset, offset, GRID_SIZE, GRID_SIZE)
    pygame.draw.rect(screen, COLOR_BG, board_rect, border_radius=8)

    # 3. 遍历并画出每个数字方块
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            value = board[r][c]
            rect_x = board_rect.left + TILE_MARGIN + c * (TILE_SIZE + TILE_MARGIN)
            rect_y = board_rect.top + TILE_MARGIN + r * (TILE_SIZE + TILE_MARGIN)

            # 确定方块颜色
            tile_color = TILE_COLORS.get(value, (60, 58, 50)) if value > 0 else COLOR_EMPTY
            pygame.draw.rect(screen, tile_color, (rect_x, rect_y, TILE_SIZE, TILE_SIZE), border_radius=5)

            # 如果有数字，从预设好的 fonts 字典中取字体并渲染
            if value > 0:
                text_color = COLOR_TEXT_DARK if value <= 4 else COLOR_TEXT_LIGHT
                # 根据数字大小选择不同的字体缓存
                if value < 100:
                    s_font = fonts['large']
                elif value < 1000:
                    s_font = fonts['medium']
                else:
                    s_font = fonts['small']

                text_surf = s_font.render(str(value), True, text_color)
                text_rect = text_surf.get_rect(center=(rect_x + TILE_SIZE / 2, rect_y + TILE_SIZE / 2))
                screen.blit(text_surf, text_rect)

    # 4. 绘制遮罩和消息 (如有)
    if msg:
        overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
        overlay.fill((255, 255, 255, 180))  # 半透明白色
        screen.blit(overlay, (0, 0))

        msg_surf = fonts['msg'].render(msg, True, COLOR_TEXT_DARK)
        msg_rect = msg_surf.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 - 20))
        screen.blit(msg_surf, msg_rect)

        sub_surf = fonts['sub'].render("Press 'R' to Restart", True, COLOR_TEXT_DARK)
        sub_rect = sub_surf.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE // 2 + 30))
        screen.blit(sub_surf, sub_rect)

    # 5. 刷新屏幕
    pygame.display.flip()


# ==========================================
# 4. 主程序循环
# ==========================================

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("2048 - CPU Optimized")

    # 【优化 1】预先创建所有字体对象，避免在 draw 函数中频繁查找系统字体
    fonts = {
        'large': pygame.font.SysFont("arial", TILE_SIZE // 2, bold=True),
        'medium': pygame.font.SysFont("arial", TILE_SIZE // 3, bold=True),
        'small': pygame.font.SysFont("arial", TILE_SIZE // 4, bold=True),
        'msg': pygame.font.SysFont("arial", 45, bold=True),
        'sub': pygame.font.SysFont("arial", 20, bold=False)
    }

    # 【优化 2】创建一个时钟对象来控制帧率
    clock = pygame.time.Clock()

    board = init_board()
    game_over = False
    win_announced = False
    msg = ""

    # 【优化 3】绘制标志位，只有当画面需要改变时才执行绘图操作
    needs_update = True

    while True:
        # 如果需要更新，则重绘画面
        if needs_update:
            draw_game(screen, board, fonts, msg)
            needs_update = False  # 重绘完后重置标志位

        # 监听事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                # 重置逻辑
                if event.key == pygame.K_r:
                    board = init_board()
                    game_over, win_announced, msg = False, False, ""
                    needs_update = True
                    continue

                if not game_over:
                    old_board = [row[:] for row in board]

                    # 按键映射
                    if event.key in [pygame.K_UP, pygame.K_w]:
                        board = move_up(board)
                    elif event.key in [pygame.K_DOWN, pygame.K_s]:
                        board = move_down(board)
                    elif event.key in [pygame.K_LEFT, pygame.K_a]:
                        board = move_left(board)
                    elif event.key in [pygame.K_RIGHT, pygame.K_d]:
                        board = move_right(board)

                    # 检查棋盘是否真的发生了变化
                    if any(board[i][j] != old_board[i][j] for i in range(BOARD_SIZE) for j in range(BOARD_SIZE)):
                        add_random_tile(board)
                        needs_update = True  # 棋盘变了，标记需要重绘

                        # 检查胜负状态
                        if not win_announced and any(c >= TARGET for r in board for c in r):
                            msg, win_announced = "YOU WIN!", True
                        if not can_move(board):
                            msg, game_over = "GAME OVER!", True

        # 【优化 4】限制帧率为 60。
        # clock.tick 会让循环在执行完后休眠，直到达到预定时间，从而极大降低 CPU 占用。
        clock.tick(60)


if __name__ == "__main__":
    main()
```



