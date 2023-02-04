from concurrent.futures import ThreadPoolExecutor
import os
import threading
from math import floor
import tkinter as tk
from typing import Callable

import numpy as np
from board import Board
from player import Player, Person
from rule import NoForbidden
from gomoku import Gomoku
from gomokuai import GomokuAI
from PIL import Image, ImageTk

class MyThread(threading.Thread):
    # def __init__(self, group = ..., target = ..., name = ..., args = ..., kwargs = ..., *, daemon = ...) -> None:
    #     super().__init__(group, target, name, args, kwargs, daemon=daemon)
    # def __init__(self, func, args=()):
    #     super(MyThread,self).__init__()
    #     self.func = func
    #     self.args = args
    def run(self):
        self.result = self.target(*self.args)
    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

class GomokuUI:
    def __init__(self, player0:Player, player1:Player, \
        rule=NoForbidden, board: Board = Board()) -> None:
        os.chdir(os.path.dirname(__file__))
        self.gomoku = Gomoku(player0, player1, rule, board)
        self.mode = 'set_board'
        self.result = -1
        
    def ui_init(self):
        board = self.gomoku.board
        self.piece_sz = 36
        self.board_st = (46, 46)
        self.board_ed = (self.board_st[0]+(board.col-1)*self.piece_sz,\
                         self.board_st[1]+(board.row-1)*self.piece_sz)

        self.root = tk.Tk()
        self.root.title("Gomoku")
        self.root.geometry("760x580")

        # 棋子图片
        self.black_img = Image.open(r'.\black.png')\
            .resize((self.piece_sz, self.piece_sz), Image.Resampling.LANCZOS)
        self.white_img = Image.open(r'.\white.png')\
            .resize((self.piece_sz, self.piece_sz), Image.Resampling.LANCZOS)
        self.black_img = ImageTk.PhotoImage(self.black_img)
        self.white_img = ImageTk.PhotoImage(self.white_img)

        self.canvas = tk.Canvas(self.root, \
            width = self.board_ed[0]+40, height = self.board_ed[1]+40)
        self.canvas.bind("<Button-1>", self.board_click_event) # 对鼠标进行事件绑定
        self.canvas.bind("<Button-3>", self.board_right_click_event)
        self.canvas.grid(row = 0, column = 0, rowspan = 7)

        # 线条
        for i in range(board.row):
            self.canvas.create_line(self.board_st[0], (self.board_st[1]+self.piece_sz*i), \
                self.board_ed[0], (self.board_st[1] + self.piece_sz*i))
        for i in range(board.col):
            self.canvas.create_line((self.board_st[0]+self.piece_sz*i), self.board_st[1], \
                (self.board_st[0]+self.piece_sz*i), self.board_ed[1])
        if board.col == 15 and board.row == 15:
            # 点
            point_x = [3, 3, 11, 11, 7]
            point_y = [3, 11, 3, 11, 7]
            for i in range(5):
                self.canvas.create_oval(self.board_st[0]+self.piece_sz*point_x[i]-4, \
                    self.board_st[1]+self.piece_sz*point_y[i]-4, \
                    self.board_st[0]+self.piece_sz*point_x[i]+4, \
                    self.board_st[1]+self.piece_sz*point_y[i]+4,\
                    fill = "black")
        # 数字坐标
        for i in range(board.row):
            label = tk.Label(self.canvas, text = str(i + 1), fg = "black",
                            width = 2, anchor = 'e')
            label.place(x = self.board_st[0]-self.piece_sz/2-2, \
                        y = self.board_st[1]+self.piece_sz*i, anchor = 'e')
        # 字母坐标
        count = 0
        for i in range(65, 65+board.col):
            # if chr(i) == 'I':  continue
            label = tk.Label(self.canvas, text = chr(i), fg = "black")
            label.place(x = self.board_st[0]+self.piece_sz*count, \
                        y = self.board_st[1]-self.piece_sz/2-2, anchor = 's')
            count += 1

        # 棋子图片数组
        self.pieces_img = [[0 for j in range(board.col)] for i in range(board.row)]

        # 初始化棋盘上的子
        for i in range(board.row):
            for j in range(board.col):
                if board[i][j] == 0 or board[i][j] == 1:
                    self.draw_piece(i, j, color = board[i][j])

        # 提示当前应该落什么颜色的字
        self.current_piece_canvas = tk.Canvas(self.root, \
            width = self.piece_sz+20, height = self.piece_sz+20)
        # self.current_piece_canvas.bind("<Button-1>", self.current_piece_canvas_click_event)
        self.current_piece_canvas.grid(row = 0, column = 1)
        self.current_piece = self.current_piece_canvas.create_image(10, 10, anchor='nw',\
            image=(self.black_img, self.white_img)[self.gomoku.board.turn])

        button_font = ('楷体', 12)
        self.start_button = tk.Button(self.root, text='开始游戏', font=button_font, \
            width=10, height=2, command=self.start_play)
        self.start_button.grid(row = 2, column = 1)

        self.concede_button = tk.Button(self.root, text='投子认负', font=button_font, \
            width=10, height=2, command=self.concede, state=tk.DISABLED)
        self.concede_button.grid(row = 3, column = 1)

        self.set_board_button = tk.Button(self.root, text='摆棋', font=button_font, \
            width=10, height=2, command=self.enable_set_board, state=tk.DISABLED)
        self.set_board_button.grid(row = 4, column = 1)

        self.clear_board_button = tk.Button(self.root, text='清空棋盘', font=button_font, \
            width=10, height=2, command=self.clear_board)
        self.clear_board_button.grid(row = 5, column = 1)

    # def current_piece_canvas_click_event(self, event): ...

    def clear_board(self): 
        board = self.gomoku.board
        for i in range(board.row):
            for j in range(board.col):
                if board[i][j] != -1:
                    self.delete_piece(i, j)
                    board[i][j] = -1

        if self.result != -1:
            self.result_label.destroy()
        self.result = -1

    def enable_set_board(self): 
        self.mode = 'set_board'
        self.start_button['state'] = tk.NORMAL
        self.concede_button['state'] = tk.DISABLED
        self.set_board_button['state'] = tk.DISABLED
        self.clear_board_button['state'] = tk.NORMAL

        if self.result != -1:
            self.result_label.destroy()
        self.result = -1

    def start_play(self):
        self.mode = 'play'
        self.start_button['state'] = tk.DISABLED
        self.concede_button['state'] = tk.NORMAL
        self.set_board_button['state'] = tk.DISABLED
        self.clear_board_button['state'] = tk.DISABLED
        self.__already_start = False
        
        if self.result != -1:
            self.result_label.destroy()
        self.result = -1

        self.gomoku_t = threading.Thread(target=self.gomoku.play, \
            args=(), name='gomoku_t')
        self.gomoku_t.setDaemon(True)
        self.gomoku_t.start()

        # with ThreadPoolExecutor(max_workers=1) as executor:
        #     self.gomoku_t = executor.submit(self.gomoku.play)

        self.root.after(100, self.check_gomoku_seq)

    def concede(self):
        if self.mode == 'play':
            player = self.gomoku.player[self.gomoku.board.turn]
            if isinstance(player, Person):
                player.que.put((-1, -1))

    def game_end(self):
        self.mode = 'end'
        self.start_button['state'] = tk.DISABLED
        self.concede_button['state'] = tk.DISABLED
        self.set_board_button['state'] = tk.NORMAL
        self.clear_board_button['state'] = tk.NORMAL
        result_text = '黑方胜' if self.result == 0 else '白方胜'
        self.result_label = tk.Label(self.root, text=result_text, font=('楷体', 14),\
            width=10, height=2)
        self.result_label.grid(row = 1, column = 1)

    def refresh_current_piece(self):
        self.current_piece_canvas.delete(self.current_piece)
        self.current_piece = self.current_piece_canvas.create_image(10, 10, anchor='nw',\
            image=(self.black_img, self.white_img)[self.gomoku.board.turn])

    def board_click_event(self, event):
        y = floor((event.x-self.board_st[0]+self.piece_sz/2) / self.piece_sz)
        x = floor((event.y-self.board_st[1]+self.piece_sz/2) / self.piece_sz)
        if x < 0 or y < 0 or x >= self.gomoku.board.row or y >= self.gomoku.board.col: 
            return
        if self.mode == 'set_board':
            if self.gomoku.board[x][y] != -1:  return
            self.gomoku.board[x][y] = self.gomoku.board.turn
            self.draw_piece(x, y, self.gomoku.board.turn)
            self.gomoku.board.turn ^= 1
            self.refresh_current_piece()
        elif self.mode == 'play':
            player = self.gomoku.player[self.gomoku.board.turn]
            if isinstance(player, Person):
                player.que.put((x, y))

    def board_right_click_event(self, event):
        y = floor((event.x-self.board_st[0]+self.piece_sz/2) / self.piece_sz)
        x = floor((event.y-self.board_st[1]+self.piece_sz/2) / self.piece_sz)
        if self.mode == 'set_board':
            if self.gomoku.board[x][y] != -1:
                self.gomoku.board[x][y] = -1
                self.delete_piece(x, y)

    def delete_piece(self, x, y):
        self.canvas.delete(self.pieces_img[x][y])

    def draw_piece(self, x, y, color): # color == 0: black
        self.pieces_img[x][y] = self.canvas.create_image(\
            self.board_st[0]+y*self.piece_sz, \
            self.board_st[1]+x*self.piece_sz, \
            anchor='c', image=(self.black_img, self.white_img)[color])

    def check_gomoku_seq(self):
        while not self.gomoku.seq.empty():
            action = self.gomoku.seq.get()
            if not self.__already_start:
                if action == (-2, -2, -2):
                    self.__already_start = True
                continue
            self.refresh_current_piece()
            if action[0] == -2 and action[1] == -2:
                self.result = action[2]
                self.game_end()
                return
            elif action[0] >= 0 and action[1] >= 0:
                self.draw_piece(action[0], action[1], action[2])
        self.root.after(100, self.check_gomoku_seq)

    def draw_ui(self):
        self.ui_init()
        self.root.mainloop()

if __name__ == '__main__':
    board = Board()
    # board[4][4] = 1
    # board[7][6] = 0
    # board[7][7] = 0
    # board[6][7] = 1
    # p0, p1 = Person('Alice'), Person('Bob')
    # p0 = GomokuAI('Alice')
    p0 = Person('Alice')
    p1 = Person('Bob')
    g = GomokuUI(p0, p1, NoForbidden, board)
    g.draw_ui()
