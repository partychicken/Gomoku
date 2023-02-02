import copy
from rule import NoForbidden
from board import Board
from player import Player
import torch
from torch import nn
from torchvision import models
import gomokunet

class MCTS_node():
    def __init__(self) -> None:
        # self.board = copy.deepcopy(board)
        self.n     = 0  # 搜索次数，结点自身通过神经网络推断占用一次搜索次数
        self.p     = [] # 神经网络输出的各个行动的概率分布，是tensor
        self.v     = 0  # 神经网络输出的价值/胜率，是float
        self.q     = 0  # n次搜索后的平均价值
        self.pn    = torch.zeros(15*15, device=gomokunet.Environment.device) # 子结点访问次数
        self.nodes = {} # 子结点字典，用下一次落子的坐标x*15+y作为key

    def inferr(self, board:torch.Tensor, policynet:nn.Module, valuenet:nn.Module):
        self.p = policynet(board)
        self.v = valuenet(board).item()

    def search(self, board:torch.Tensor, policynet:nn.Module, valuenet:nn.Module, \
        calc_tot:int):
        if self.n == 0:
            self.n += 1
            self.inferr(board, policynet, valuenet)
            if calc_tot == 1:  return
        for i in range(self.n, calc_tot):
            ...
        ...
        
    def choose_action(self):
        sample = torch.multinomial(self.pn, 1)
        x = sample / 15
        y = sample % 15
        subtree = self.nodes[sample]
        return x, y, subtree
        
    def get_subtree(self, action):
        index = action[0]*15 + action[1]
        if index in self.nodes:  return self.nodes[index]
        else:  return MCTS_node()


class GomokuAI(Player):
    def __init__(self, name = '', policynet:nn.Module = None, valuenet:nn.Module = None):
        super().__init__(name)
        if policynet is None:
            policynet = models.resnet18()
            policynet.fc = nn.Sequential(nn.Linear(512, 15 * 15), nn.Softmax(dim=1))
        if valuenet is None:
            valuenet  = models.resnet18()
            valuenet.fc  = nn.Sequential(nn.Linear(512, 1), nn.Tanh())
        self.policynet = policynet
        self.valuenet  = valuenet

    def init_game(self, board:Board, color, turn):
        self.board = copy.deepcopy(board)
        self.color = color
        self.turn  = turn
        self.policynet.eval()
        self.valuenet.eval()

    def start_play(self): 
        self.root = MCTS_node()

    def next_action(self, sec = 0, calc_tot = 100): 
        tensor = gomokunet.board_to_tensor(self.board)
        self.root.search(tensor, self.policynet, self.valuenet, calc_tot)
        x, y, subtree = self.root.choose_action()

        self.root = subtree
        action = (x, y, self.board.turn)
        NoForbidden.play(self.board, action)
        return action

    def opponent_action(self, action):
        self.root = self.root.get_subtree(action)
        NoForbidden.play(self.board, action)