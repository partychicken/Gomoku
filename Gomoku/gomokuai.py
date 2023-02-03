import copy
from math import sqrt
from rule import NoForbidden
from board import Board
from player import Player
import torch
from torch import nn
from torchvision import models
import gomokunet
from gomokunet import Env

class MCTS_node():
    c_puct = 5.0      # PUCT的常数因子
    diri_ratio = 0.25 # Dirichlet噪声所占比率
    diri_alpha = 0.3  # Dirichlet噪声的浓度参数alpha

    def __init__(self) -> None:
        # self.board = copy.deepcopy(board)
        # self.board = [] # 棋盘状态的tensor
        self.n      = 0  # 搜索次数，结点自身通过神经网络推断并扩展（expand）占用一次搜索次数
        self.q      = 0  # n次搜索后的平均价值，是各个子结点价值和self.v根据访问次数的加权平均
        self.p      = [] # 神经网络输出的各个行动的概率分布，是二维1x225tensor
        self.v      = 0  # 神经网络输出的价值/胜率，是float
        self.nn     = [] # 子结点访问次数+1，是一维tensor
        self.qn     = [] # 子结点的q取负，是一维tensor
        self.legal  = [] # 所有能落子的位置，是一维tensor，用0/1表示
        self.p_real = [] # 去除p中不能落子位置的概率分布再乘上c_puct，是一维tensor
        self.nodes  = {} # 子结点字典，用下一次落子的坐标x*15+y作为key

    def expand(self, board:Board, policynet:nn.Module, valuenet:nn.Module, self_play):
        state = gomokunet.board_to_tensor(board)
        self.n  = 1
        self.p  = policynet(state)
        self.q  = self.v = valuenet(state).item()
        self.nn = torch.ones(Env.board_sz, device=Env.device)
        self.qn = torch.zeros(Env.board_sz, device=Env.device)
        self.p_real = self.p.clone().detach().squeeze_(0)
        self.legal  = torch.ones(Env.board_sz)
        for i in range(0, Env.board_shape[0]):
            for j in range(0, Env.board_shape[1]):
                if board[i][j] != -1:
                    self.legal[i*Env.board_shape[0]+j] = 0
                    self.p_real[i*Env.board_shape[0]+j] = 0
                else: self.nodes[i*Env.board_shape[0]+j] = MCTS_node()
        self.p_real *= (self.c_puct / torch.sum(self.p_real))
        if self_play:
            ...

    def select_node(self, self_play):
        # PUCT(s, a) = Q(s, a) + U(s, a)
        # U(s, a) = c_puct * P(s, a) * sqrt(N_root) / (1+N_sub)
        u = (self.p_real * sqrt(self.n)).div(self.nn)
        puct = self.qn + u
        return torch.argmax(puct).item()

    def search(self, board:Board, policynet:nn.Module, valuenet:nn.Module, \
        calc_tot:int, self_play = False):
        if self.n == 0:
            self.expand(board, policynet, valuenet, self_play)
            if calc_tot == 1:  return self.v
        for i in range(0, calc_tot):
            # node = self.select_node(self_play)
            index = self.select_node(self_play)
            node = self.nodes[index]
            x, y = index/Env.board_shape[1],index%Env.board_shape[1]
            
            board[x][y] = board.turn
            board.turn ^= 1
            q_new = node.search(board, policynet, valuenet, 1, self_play)
            board.turn ^= 1
            board[x][y] = -1

            self.q = (self.n*self.q - q_new) / (self.n+1)
            self.n += 1
            self.qn[index] = -self.nodes[index].q
            self.nn[index] += 1
        return -q_new if calc_tot == 1 else self.q # 返回本次search获得的价值
        
    def choose_action(self, self_play = False):
        dis = self.nn - 1
        # 自学习时，加入狄利克雷噪声
        if self_play:
            noise = torch.distributions.dirichlet.Dirichlet(\
                     torch.ones(Env.board_sz)*self.diri_alpha).sample()
            noise = torch.where(self.legal == 1, noise, torch.zeros(Env.board_sz))
            dis += ((self.diri_ratio/(1-self.diri_ratio)) * torch.sum(dis)\
                 / torch.sum(noise)) * noise
        sample = torch.multinomial(dis, 1)
        x = sample / Env.board_shape[1]
        y = sample % Env.board_shape[1]
        subtree = self.nodes[sample]
        return x, y, subtree
        
    def get_subtree(self, action):
        index = action[0]*Env.board_shape[1] + action[1]
        if index in self.nodes:  return self.nodes[index]
        else:  return MCTS_node()


class GomokuAI(Player):
    def __init__(self, name = '', policynet:nn.Module = None, valuenet:nn.Module = None):
        super().__init__(name)
        if policynet is None:
            policynet = models.resnet18()
            policynet.fc = nn.Sequential(nn.Linear(512, Env.board_sz), nn.Softmax(dim=1))
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
        self.root.search(self.board, self.policynet, self.valuenet, calc_tot)
        x, y, subtree = self.root.choose_action()

        self.root = subtree
        action = (x, y, self.board.turn)
        NoForbidden.play(self.board, action)
        return action

    def opponent_action(self, action):
        self.root = self.root.get_subtree(action)
        NoForbidden.play(self.board, action)