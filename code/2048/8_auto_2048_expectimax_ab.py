#!/usr/bin/env python3
import random
import os
import time
from functools import lru_cache

BOARD_SIZE = 4
TARGET = 2048
MAX_CACHE_SIZE = 10000

def board_to_key(board):
    """将棋盘转化为不可变的哈希键"""
    return tuple(tuple(row) for row in board)

def key_to_board(key):
    """将哈希键转回棋盘"""
    return [list(row) for row in key]

@lru_cache(maxsize=MAX_CACHE_SIZE)
def expectimax_ab(board_key, depth, is_player, alpha, beta):
    """
    使用 alpha-beta 剪枝的 Expectimax 搜索。
    注意：仅在玩家（max）节点使用剪枝，
    在 chance 节点不做剪枝，直接计算期望值。
    """
    board = key_to_board(board_key)
    if depth == 0 or not can_move(board):
        return evaluate_board(board)
    
    if is_player:
        best = -float('inf')
        a = alpha
        for move_func in [move_up, move_down, move_left, move_right]:
            new_board = move_func(board)
            if boards_equal(board, new_board):
                continue
            val = expectimax_ab(board_to_key(new_board), depth - 1, False, a, beta)
            best = max(best, val)
            a = max(a, best)
            if a >= beta:
                break  # 剪枝：当前分支已经足够好
        return best
    else:
        # 在 chance 节点直接计算所有可能情况的期望值，不进行剪枝
        empty = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i][j] == 0]
        if not empty:
            return evaluate_board(board)
        total = 0.0
        for (i, j) in empty:
            for tile, prob in [(2, 0.9), (4, 0.1)]:
                board_copy = [row[:] for row in board]
                board_copy[i][j] = tile
                val = expectimax_ab(board_to_key(board_copy), depth - 1, True, alpha, beta)
                total += prob * val
        return total / len(empty)

def init_board():
    board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    add_random_tile(board)
    add_random_tile(board)
    return board

def add_random_tile(board):
    empty = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i][j] == 0]
    if not empty:
        return
    i, j = random.choice(empty)
    board[i][j] = 4 if random.random() < 0.1 else 2

def print_board(board):
    # 使用 ANSI 转义码清屏
    print('\033[H\033[J', end='')
    print("-" * (BOARD_SIZE * 7 + 1))
    for row in board:
        print("|", end="")
        for num in row:
            if num == 0:
                print("      |", end="")
            else:
                print(f"{num:^6}|", end="")
        print()
        print("-" * (BOARD_SIZE * 7 + 1))

def slide_and_merge(line):
    new_line = [num for num in line if num != 0]
    merged_line = []
    skip = False
    for i in range(len(new_line)):
        if skip:
            skip = False
            continue
        if i + 1 < len(new_line) and new_line[i] == new_line[i+1]:
            merged_line.append(new_line[i] * 2)
            skip = True
        else:
            merged_line.append(new_line[i])
    merged_line += [0] * (BOARD_SIZE - len(merged_line))
    return merged_line

def move_left(board):
    return [slide_and_merge(row) for row in board]

def reverse(board):
    return [list(reversed(row)) for row in board]

def transpose(board):
    return [list(row) for row in zip(*board)]

def move_right(board):
    return reverse(move_left(reverse(board)))

def move_up(board):
    return transpose(move_left(transpose(board)))

def move_down(board):
    return transpose(move_right(transpose(board)))

def boards_equal(b1, b2):
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if b1[i][j] != b2[i][j]:
                return False
    return True

def can_move(board):
    for row in board:
        if 0 in row:
            return True
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if j + 1 < BOARD_SIZE and board[i][j] == board[i][j+1]:
                return True
            if i + 1 < BOARD_SIZE and board[i][j] == board[i+1][j]:
                return True
    return False

def reached_target(board):
    for row in board:
        if any(num >= TARGET for num in row):
            return True
    return False

def count_empty(board):
    return sum(row.count(0) for row in board)

def calculate_smoothness(board):
    """计算平滑度：相邻单元格差值的总和（差值越小越好）"""
    smooth = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE - 1):
            smooth -= abs(board[i][j] - board[i][j+1])
    for j in range(BOARD_SIZE):
        for i in range(BOARD_SIZE - 1):
            smooth -= abs(board[i][j] - board[i+1][j])
    return smooth

def calculate_monotonicity(board):
    """计算单调性：如果行或列单调性好则奖励"""
    mono_rows = 0
    mono_cols = 0
    for row in board:
        if all(row[i] >= row[i+1] for i in range(BOARD_SIZE-1)) or all(row[i] <= row[i+1] for i in range(BOARD_SIZE-1)):
            mono_rows += 1
    for col in zip(*board):
        if all(col[i] >= col[i+1] for i in range(BOARD_SIZE-1)) or all(col[i] <= col[i+1] for i in range(BOARD_SIZE-1)):
            mono_cols += 1
    return mono_rows + mono_cols

def evaluate_board(board):
    empty = count_empty(board)
    smoothness = calculate_smoothness(board)
    monotonicity = calculate_monotonicity(board)
    # 权重矩阵（蛇形布局），鼓励大数集中在角落
    weights = [
        [65536, 32768, 16384, 8192],
        [512,   1024,  2048,  4096],
        [256,    512,  1024,  2048],
        [128,    256,   512,  1024]
    ]
    weight_score = sum(board[i][j] * weights[i][j] for i in range(BOARD_SIZE) for j in range(BOARD_SIZE))
    return weight_score + empty * 500 + smoothness * 5 + monotonicity * 100

def get_dynamic_depth(board):
    """根据当前空格数量动态调整搜索深度：空格越少，局势越紧张，搜索深度加深"""
    empty_cells = count_empty(board)
    if empty_cells >= 6:
        return 4
    elif empty_cells >= 3:
        return 5
    else:
        return 6

def choose_best_move(board):
    moves = {
        'w': move_up,
        'a': move_left,
        's': move_down,
        'd': move_right
    }
    best_move = None
    best_score = -float('inf')
    best_new_board = None
    depth = get_dynamic_depth(board)
    for move, func in moves.items():
        new_board = func(board)
        if boards_equal(board, new_board):
            continue
        score = expectimax_ab(board_to_key(new_board), depth, False, -float('inf'), float('inf'))
        if score > best_score:
            best_score = score
            best_move = move
            best_new_board = new_board
    return best_move, best_new_board

def main():
    board = init_board()
    has_printed_win = False
    print_board(board)
    while True:
        if not can_move(board):
            print("游戏结束，无法移动！")
            break
        move, new_board = choose_best_move(board)
        if move is None:
            print("没有有效移动，游戏结束！")
            break
        print(f"自动移动：{move}")
        board = new_board
        add_random_tile(board)
        print_board(board)
        if reached_target(board) and not has_printed_win:
            print("已达到 2048！继续自动移动以获得更高分数。")
            has_printed_win = True
        time.sleep(0.0001)

if __name__ == "__main__":
    main()

