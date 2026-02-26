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