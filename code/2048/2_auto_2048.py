#!/usr/bin/env python3
import random
import os
import sys
import time

BOARD_SIZE = 4
TARGET = 2048

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
    # 以 10% 的概率生成 4，其余生成 2
    board[i][j] = 4 if random.random() < 0.1 else 2

def print_board(board):
    os.system('clear')  # 清屏
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
    # 去除所有零值
    new_line = [num for num in line if num != 0]
    merged_line = []
    skip = False
    for i in range(len(new_line)):
        if skip:
            skip = False
            continue
        # 若相邻数字相等，则合并
        if i + 1 < len(new_line) and new_line[i] == new_line[i + 1]:
            merged_line.append(new_line[i] * 2)
            skip = True
        else:
            merged_line.append(new_line[i])
    # 补0使长度固定
    merged_line += [0] * (BOARD_SIZE - len(merged_line))
    return merged_line

def move_left(board):
    new_board = []
    for row in board:
        new_board.append(slide_and_merge(row))
    return new_board

def reverse(board):
    return [list(reversed(row)) for row in board]

def transpose(board):
    return [list(row) for row in zip(*board)]

def move_right(board):
    return reverse(move_left(reverse(board)))

def move_up(board):
    transposed = transpose(board)
    moved = move_left(transposed)
    return transpose(moved)

def move_down(board):
    transposed = transpose(board)
    moved = move_right(transposed)
    return transpose(moved)

def boards_equal(b1, b2):
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if b1[i][j] != b2[i][j]:
                return False
    return True

def can_move(board):
    # 若有空格则肯定能移动
    for row in board:
        if 0 in row:
            return True
    # 检查水平和垂直是否有可合并的数字
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if j + 1 < BOARD_SIZE and board[i][j] == board[i][j + 1]:
                return True
            if i + 1 < BOARD_SIZE and board[i][j] == board[i + 1][j]:
                return True
    return False

def reached_target(board):
    for row in board:
        if any(num >= TARGET for num in row):
            return True
    return False

def main():
    board = init_board()
    has_printed_win = False  # 避免重复打印获胜消息
    print_board(board)
    while True:
        if not can_move(board):
            print("游戏结束，无法移动！")
            break

        # 自动随机选择一个方向
        move = random.choice(['w', 'a', 's', 'd'])
        print(f"自动移动：{move}")
        if move == 'w':
            new_board = move_up(board)
        elif move == 'a':
            new_board = move_left(board)
        elif move == 's':
            new_board = move_down(board)
        elif move == 'd':
            new_board = move_right(board)

        if not boards_equal(board, new_board):
            board = new_board
            add_random_tile(board)
        else:
            print("无效移动，尝试其他方向。")

        print_board(board)
        if reached_target(board) and not has_printed_win:
            print("已达到 2048！继续自动移动以获得更高分数。")
            has_printed_win = True

        time.sleep(0.5)  # 延时0.5秒，便于观察演示效果

if __name__ == "__main__":
    main()

