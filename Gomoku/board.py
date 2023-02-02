import numpy as np

class Board:
    # -1: empty; 0: black; 1: white
    def __init__(self, row=15, column=0, turn=0) -> None:
        if column == 0:  column = row
        self.row   = row
        self.col   = column
        self.board = -np.ones(shape=(self.row, self.col), dtype=np.int8)
        self.win   = -1
        self.turn  = turn

    def __getitem__(self, index):
        return self.board.__getitem__(index)

    def clear(self):
        self.board = -np.ones(shape=(self.row, self.col), dtype=np.int8)
        self.win   = -1
        self.turn  = 0