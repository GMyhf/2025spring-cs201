#!/usr/bin/env python3
import random
import os
import sys
import time
import copy

BOARD_SIZE = 4
TARGET = 2048
SEARCH_DEPTH = 3  # 增加搜索深度

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
    os.system('clear')  # Mac/Linux 下清屏
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
        if i + 1 < len(new_line) and new_line[i] == new_line[i + 1]:
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

def evaluate_board(board):
    score = 0
    # 奖励空格
    empty = sum(row.count(0) for row in board)
    score += empty * 100

    # 使用一个预设的权重矩阵，鼓励最大值位于角落（这里采用蛇形布局）
    weights = [
        [65536, 32768, 16384, 8192],
        [512,   1024,  2048,  4096],
        [256,    512,  1024,  2048],
        [128,    256,   512,  1024]
    ]
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            score += board[i][j] * weights[i][j]

    # 奖励行单调性
    for row in board:
        if all(row[i] >= row[i+1] for i in range(len(row)-1)) or all(row[i] <= row[i+1] for i in range(len(row)-1)):
            score += sum(row)
    # 奖励列单调性
    for col in zip(*board):
        if all(col[i] >= col[i+1] for i in range(len(col)-1)) or all(col[i] <= col[i+1] for i in range(len(col)-1)):
            score += sum(col)

    return score

def expectimax(board, depth, is_player):
    if depth == 0 or not can_move(board):
        return evaluate_board(board)
    if is_player:
        best = -float('inf')
        # 玩家节点：尝试所有四个方向
        for move_func in [move_up, move_down, move_left, move_right]:
            new_board = move_func(board)
            if boards_equal(board, new_board):
                continue
            value = expectimax(new_board, depth, False)
            best = max(best, value)
        return best if best != -float('inf') else evaluate_board(board)
    else:
        # 概率节点：模拟在所有空格上随机出现 2 或 4
        empty = [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i][j] == 0]
        if not empty:
            return evaluate_board(board)
        total = 0.0
        for (i, j) in empty:
            for tile, prob in [(2, 0.9), (4, 0.1)]:
                new_board = copy.deepcopy(board)
                new_board[i][j] = tile
                total += prob * expectimax(new_board, depth - 1, True)
        return total / len(empty)

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
    for move, func in moves.items():
        new_board = func(board)
        if boards_equal(board, new_board):
            continue
        # 此处使用多层期望搜索，后续节点为概率节点
        score = expectimax(new_board, SEARCH_DEPTH, False)
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

        time.sleep(0.2)

if __name__ == "__main__":
    main()

