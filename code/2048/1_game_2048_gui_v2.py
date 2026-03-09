#!/usr/bin/env python3
import pygame, random, sys, json, os

# ================= 配置 =================
BOARD_SIZE = 4
TARGET = 2048
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 580
HEADER = 140
GRID_SIZE = 400
TILE_MARGIN = 10
TILE_SIZE = (GRID_SIZE - (BOARD_SIZE + 1) * TILE_MARGIN)//BOARD_SIZE
SAVE_FILE = "best_score.json"

COLOR_BG = (250, 248, 239)
COLOR_BOARD = (187, 173, 160)
COLOR_EMPTY = (205, 193, 180)
COLOR_TEXT_DARK = (119, 110, 101)
COLOR_TEXT_LIGHT = (249, 246, 242)
BUTTON_COLOR = (143, 122, 102)
BUTTON_HOVER = (170, 140, 110)

TILE_COLORS = {
2:(238,228,218),4:(237,224,200),8:(242,177,121),
16:(245,149,99),32:(245,124,95),64:(246,94,59),
128:(237,207,114),256:(237,204,97),512:(237,200,80),
1024:(237,197,63),2048:(237,194,46)
}

# =============== 分数存档 ===============
def load_best():
    if os.path.exists(SAVE_FILE):
        return json.load(open(SAVE_FILE))["best"]
    return 0

def save_best(v):
    json.dump({"best":v}, open(SAVE_FILE,"w"))

# =============== 棋盘逻辑 ===============
def init_board():
    board=[[0]*BOARD_SIZE for _ in range(BOARD_SIZE)]
    add_random(board)
    add_random(board)
    return board

def add_random(board):
    empty=[(i,j) for i in range(BOARD_SIZE)
                  for j in range(BOARD_SIZE) if board[i][j]==0]
    if not empty: return
    i,j=random.choice(empty)
    board[i][j]=4 if random.random()<0.1 else 2

def slide(line):
    global score
    new=[x for x in line if x!=0]
    merged=[]
    skip=False
    for i in range(len(new)):
        if skip:
            skip=False
            continue
        if i+1<len(new) and new[i]==new[i+1]:
            v=new[i]*2
            score+=v
            merged.append(v)
            skip=True
        else:
            merged.append(new[i])
    merged+=[0]*(BOARD_SIZE-len(merged))
    return merged

def move_left(board): return [slide(row) for row in board]
def reverse(board): return [list(reversed(r)) for r in board]
def transpose(board): return [list(r) for r in zip(*board)]
def move_right(board): return reverse(move_left(reverse(board)))
def move_up(board): return transpose(move_left(transpose(board)))
def move_down(board): return transpose(move_right(transpose(board)))

def can_move(board):
    for r in board:
        if 0 in r: return True
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if j+1<BOARD_SIZE and board[i][j]==board[i][j+1]: return True
            if i+1<BOARD_SIZE and board[i][j]==board[i+1][j]: return True
    return False

# =============== 绘图函数 ===============
def draw(screen, board, fonts, msg):
    screen.fill(COLOR_BG)
    # 标题
    title = fonts["title"].render("2048", True, COLOR_TEXT_DARK)
    screen.blit(title, (20,20))

    # ====== Score / Best 背景框固定大小 ======
    score_box = pygame.Rect(300,20,90,40)
    best_box  = pygame.Rect(300,70,90,40)
    pygame.draw.rect(screen,COLOR_BOARD,score_box,border_radius=5)
    pygame.draw.rect(screen,COLOR_BOARD,best_box,border_radius=5)

    # ====== 分数自适应字体显示 ======
    def fit_text(text, box, font_name="arial", bold=True, max_size=30, min_size=10):
        size = max_size
        while size >= min_size:
            font = pygame.font.SysFont(font_name,size,bold)
            surf = font.render(text,True,COLOR_TEXT_LIGHT)
            if surf.get_width() <= box.width-10:
                return surf
            size -=1
        return surf
    score_surf = fit_text(f"Score {score}",score_box)
    best_surf  = fit_text(f"Best {best}",best_box)

    screen.blit(score_surf,(score_box.x+(score_box.width-score_surf.get_width())//2,
                            score_box.y+(score_box.height-score_surf.get_height())//2))
    screen.blit(best_surf,(best_box.x+(best_box.width-best_surf.get_width())//2,
                           best_box.y+(best_box.height-best_surf.get_height())//2))

    # 按钮
    mouse = pygame.mouse.get_pos()
    restart=pygame.Rect(400,20,80,30)
    cont=None
    color = BUTTON_HOVER if restart.collidepoint(mouse) else BUTTON_COLOR
    pygame.draw.rect(screen,color,restart,border_radius=5)
    r = fonts["small"].render("Restart",True,COLOR_TEXT_LIGHT)
    screen.blit(r,(restart.x+10,restart.y+5))

    if msg=="YOU WIN!":
        cont = pygame.Rect(400,60,80,30)
        color = BUTTON_HOVER if cont.collidepoint(mouse) else BUTTON_COLOR
        pygame.draw.rect(screen,color,cont,border_radius=5)
        c = fonts["small"].render("Continue",True,COLOR_TEXT_LIGHT)
        screen.blit(c,(cont.x+5,cont.y+5))

    # 棋盘
    offset_x=(WINDOW_WIDTH-GRID_SIZE)//2
    offset_y=HEADER
    pygame.draw.rect(screen,COLOR_BOARD,(offset_x,offset_y,GRID_SIZE,GRID_SIZE),border_radius=8)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v=board[r][c]
            x=offset_x+TILE_MARGIN+c*(TILE_SIZE+TILE_MARGIN)
            y=offset_y+TILE_MARGIN+r*(TILE_SIZE+TILE_MARGIN)
            color=TILE_COLORS.get(v,(60,58,50)) if v>0 else COLOR_EMPTY
            pygame.draw.rect(screen,color,(x,y,TILE_SIZE,TILE_SIZE),border_radius=5)
            if v>0:
                font = fonts["big"] if v<100 else fonts["mid"] if v<1000 else fonts["small"]
                text = font.render(str(v),True,COLOR_TEXT_DARK if v<=4 else COLOR_TEXT_LIGHT)
                rect = text.get_rect(center=(x+TILE_SIZE/2,y+TILE_SIZE/2))
                screen.blit(text,rect)

    # 遮罩层
    if msg:
        overlay = pygame.Surface((WINDOW_WIDTH,WINDOW_HEIGHT),pygame.SRCALPHA)
        overlay.fill((255,255,255,180))
        screen.blit(overlay,(0,0))
        t = fonts["title"].render(msg,True,COLOR_TEXT_DARK)
        screen.blit(t,(WINDOW_WIDTH//2-110,WINDOW_HEIGHT//2-40))
        sub_text = "Press C to Continue / R to Restart" if msg=="YOU WIN!" else "Press R to Restart"
        sub = fonts["small"].render(sub_text,True,COLOR_TEXT_DARK)
        screen.blit(sub,(WINDOW_WIDTH//2-160,WINDOW_HEIGHT//2+20))

    pygame.display.flip()
    return restart,cont

# =============== 主程序 ===============
pygame.init()
screen=pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT))
pygame.display.set_caption("2048")

fonts={
"title":pygame.font.SysFont("arial",48,True),
"big":pygame.font.SysFont("arial",40,True),
"mid":pygame.font.SysFont("arial",32,True),
"small":pygame.font.SysFont("arial",20,True)
}

clock=pygame.time.Clock()
board=init_board()
score=0
best=load_best()
msg=""
win=False
game_over=False

while True:
    restart_btn,cont_btn=draw(screen,board,fonts,msg)
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            save_best(best)
            pygame.quit()
            sys.exit()
        if event.type==pygame.MOUSEBUTTONDOWN:
            pos=pygame.mouse.get_pos()
            if restart_btn.collidepoint(pos):
                board=init_board(); score=0; msg=""; win=False; game_over=False
            if cont_btn and cont_btn.collidepoint(pos) and msg=="YOU WIN!":
                msg=""
        if event.type==pygame.KEYDOWN:
            if event.key==pygame.K_r:
                board=init_board(); score=0; msg=""; win=False; game_over=False
                continue
            if event.key==pygame.K_c and msg=="YOU WIN!":
                msg=""
                continue
            if msg: continue
            old=[row[:] for row in board]
            if event.key in (pygame.K_LEFT,pygame.K_a): board=move_left(board)
            elif event.key in (pygame.K_RIGHT,pygame.K_d): board=move_right(board)
            elif event.key in (pygame.K_UP,pygame.K_w): board=move_up(board)
            elif event.key in (pygame.K_DOWN,pygame.K_s): board=move_down(board)
            if board!=old:
                add_random(board)
                best=max(best,score)
                if not win and any(v>=TARGET for r in board for v in r): msg="YOU WIN!"; win=True
                if not can_move(board): msg="GAME OVER"; game_over=True
    clock.tick(60)