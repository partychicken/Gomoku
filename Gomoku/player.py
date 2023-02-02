import copy
from queue import Queue
import threading
from board import Board
from rule import NoForbidden
import tkinter as tk

class Player:
    def __init__(self, name = '') -> None:
        self.name  = name

    def init_game(self, board: Board, color, turn): ...

    def start_play(self): ...

    def next_action(self, sec = 0): ...

    def opponent_action(self, action): ...

class Person(Player):
    def __init__(self, name = '') -> None:
        super().__init__(name)
        self.que = Queue()

    def init_game(self, board: Board, color, turn):
        self.board = copy.deepcopy(board)
        self.color = color
        self.turn  = turn

    def next_action(self, sec = 0):
        self.que.queue.clear()
        while True:
            (x, y) = self.que.get()
            if NoForbidden.play(self.board, (x, y, self.color)):
                return (x, y, self.color)

    def opponent_action(self, action):
        NoForbidden.play(self.board, action)
