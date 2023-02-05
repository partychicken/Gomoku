from board import Board

class Rule:
    # judge whether action is legal
    @classmethod
    def judge(cls, board: Board, action) -> bool: ...

    # judge whether game is end.
    # 0: black win, 1: white win, -1: not end, 2: tie.
    @classmethod
    def final(cls, board: Board) -> int: ...

    # if judge(board, action) is True, do the action and return True.
    # otherwise, do nothing and return False.
    @classmethod
    def play(cls, board: Board, action) -> bool: ...

    # action is a tuple (a, b, c)
    # a,b > 0 and c == 0/1: player 0/1 puts piece at (a, b)
    # a,b == -1 and c == 0/1: player 0/1 concedes defeat
    # (a, b, c) == (-2, -2, c): 
    #   c == -2: start the game
    #   c == 0/1: end the game, and winner is player 0/1

class NoForbidden(Rule):
    @classmethod
    def judge(cls, board: Board, action) -> bool:
        try:
            if action == (-2, -2, -2):  return True

            if action[2] != board.turn:  return False
            if action[0] == -1 and action[1] == -1:  return True
            if action[0] >= 0 and action[1] >= 0 and\
              action[0] < board.row and action[1] < board.col and\
              board.get(action[0], action[1]) == -1:
                return True
            return False
        except Exception:
            return False

    @classmethod
    def final(cls, board: Board, action = (-2, -2, -2)) -> int:
        if board.n == board.col * board.row:  return 2
        if action == (-2, -2, -2):
            if board.win != -1:  return board.win
            flag = True
            for i in range(board.row):
                for j in range(board.col):
                    if board.get(i, j) == -1:  continue
                    if i+5 <= board.row:
                        for k in range(i+1, i+5): 
                            if board.get(k, j) != board.get(i, j): 
                                flag = False
                                break
                        if flag:  return board.get(i, j)
                        flag = True
                    if j+5 <= board.col:
                        for k in range(j+1, j+5): 
                            if board.get(i, k) != board.get(i, j): 
                                flag = False
                                break
                        if flag:  return board.get(i, j)
                        flag = True
                    if i+5 <= board.row and j+5 <= board.col:
                        for k in range(1, 5):
                            if board.get(i+k, j+k) != board.get(i, j):
                                flag = False
                                break
                        if flag:  return board.get(i, j)
                        flag = True
                    if i+5 <= board.row and j-4 >= 0:
                        for k in range(1, 5):
                            if board.get(i+k, j-k) != board.get(i, j):
                                flag = False
                                break
                        if flag:  return board.get(i, j)
                        flag = True
            return -1
        else:
            if action[0] == -1 and action[1] == -1:  return board.win
            if action[0] >= 0 and action[1] >= 0:
                i, j = action[0], action[1]
                flag = True
                if i+5 <= board.row:
                    for k in range(i+1, i+5): 
                        if board.get(k, j) != board.get(i, j): 
                            flag = False
                            break
                    if flag:  return board.get(i, j)
                    flag = True
                if j+5 <= board.col:
                    for k in range(j+1, j+5): 
                        if board.get(i, k) != board.get(i, j): 
                            flag = False
                            break
                    if flag:  return board.get(i, j)
                    flag = True
                if i+5 <= board.row and j+5 <= board.col:
                    for k in range(1, 5):
                        if board.get(i+k, j+k) != board.get(i, j):
                            flag = False
                            break
                    if flag:  return board.get(i, j)
                    flag = True
                if i+5 <= board.row and j-4 >= 0:
                    for k in range(1, 5):
                        if board.get(i+k, j-k) != board.get(i, j):
                            flag = False
                            break
                    if flag:  return board.get(i, j)
                    flag = True
                return -1
            return -1

    @classmethod
    def play(cls, board: Board, action) -> bool:
        if cls.judge(board, action): 
            board.turn ^= 1
            if action[0] == -1 and action[1] == -1:
                board.win = action[2]^1
            board.put(action[0], action[1], action[2])
            return True
        return False