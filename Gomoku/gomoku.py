import copy
from queue import Queue
import threading
from rule import NoForbidden
from board import Board
from player import Player, Person

class Gomoku:

    # player0 is black, player1 is white
    # turn=0/1: player0/1 plays first
    def __init__(self, player0:Player, player1:Player, \
        rule=NoForbidden, board:Board=Board(), turn=0) -> None:
        self.player = (player0, player1)
        self.rule   = rule
        self.board  = copy.deepcopy(board)
        self.turn   = turn
        self.seq    = Queue()

    def play(self):
        self.seq.queue.clear()
        self.board.win = -1
        self.player[0].init_game(self.board, 0, self.turn)
        self.player[1].init_game(self.board, 1, self.turn)
        self.player[0].start_play()
        self.player[1].start_play()
        self.seq.put((-2, -2, -2))

        result = self.rule.final(self.board)
        while result == -1:
            current_player = self.player[self.turn]
            opposite_player = self.player[self.turn ^ 1]

            action = current_player.next_action()
            if not self.rule.play(self.board, action):
                result = self.turn ^ 1
                break
            self.seq.put(action)
            opposite_player.opponent_action(action)
            
            self.turn ^= 1
            result = self.rule.final(self.board)
        self.seq.put((-2, -2, result))
        return result

# if __name__ == '__main__':
#     p0, p1 = Person('Alice'), Person('Bob')
#     gomoku = Gomoku(p0, p1)
#     gomoku.play()
