import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from gomoku import Gomoku
from gomokuenv import *
from gomokuai import GomokuAI
from board import Board
from rule import NoForbidden

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index][0], self.y[index][1]

def get_selfplay_data(policynet:nn.Module, valuenet:nn.Module):
    p0 = GomokuAI('p0', policynet, valuenet, True)
    p1 = GomokuAI('p1', policynet, valuenet, True)
    g = Gomoku(p0, p1, NoForbidden, Board(Env.board_shape[0], Env.board_shape[1]))
    result = g.play()
    if result == 0:
        return p0.state_seq, p0.target_seq
    elif result == 1 or result == 2:
        return p1.state_seq, p1.target_seq
    else:  assert False

def data_augmentation(states, targets):
    s, t = [], []
    for index, state in enumerate(states):
        p, q = targets[index]
        p = p.reshape(shape=Env.board_shape)
        s1, p1 = torch.flip(state, [1]), torch.flip(p, [0])
        s2, p2 = torch.flip(state, [2]), torch.flip(p, [1])
        s3, p3 = torch.flip(s2   , [1]), torch.flip(p2, [0])
        s4, p4 = torch.rot90(state, 1, [1, 2]), torch.rot90(p, 1, [0, 1])
        s5, p5 = torch.flip(s4   , [1]), torch.flip(p4, [0])
        s6, p6 = torch.flip(s4   , [2]), torch.flip(p4, [1])
        s7, p7 = torch.flip(s6   , [1]), torch.flip(p6, [0])
        pp = [p1, p2, p3, p4, p5, p6, p7]
        for i in range(7): pp[i] = pp[i].reshape(shape=(Env.board_sz,))
        s += [s1, s2, s3, s4, s5, s6, s7]
        t += list(zip(pp, [q, q, q, q, q, q, q]))
    states += s
    targets += t

def init_datapool(policynet:nn.Module, valuenet:nn.Module) -> DataLoader:
    x, y = [], []
    while len(x) < Env.datapool_sz:
        x1, y1 = get_selfplay_data(policynet, valuenet)
        print('a selfplay finished.')
        data_augmentation(x1, y1)
        print('an augmentation finished.')
        x += x1
        y += y1
        print('len x = ' + str(len(x)))
    print('data collection finished.')
    dataset = MyDataset(x, y)
    dataloader = DataLoader(dataset=dataset, batch_size=Env.batch_sz\
        , shuffle=True, num_workers=Env.num_workers)
    return dataloader

def train(policynet:nn.Module, valuenet:nn.Module):
    data_loader = init_datapool(policynet, valuenet)
    policynet.train()
    valuenet.train()
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion  = nn.MSELoss()
    policy_optimizer = optim.SGD(policynet.parameters(), lr=0.001, momentum=0.9)
    value_optimizer = optim.SGD(valuenet.parameters(), lr=0.001, momentum=0.9)

    for epoch in tqdm(range(Env.epochs)):
        for index, (inputs, policy_targets, value_targets) in enumerate(data_loader):
            # 清空梯度
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()

            # forward, backward, optimize
            # inputs, policy_targets, value_targets = ...
            policy_out = policynet(inputs)
            value_out  = valuenet(inputs)
            policy_loss = policy_criterion(policy_out, policy_targets)
            value_loss  = value_criterion(value_out, value_targets)
            policy_loss.backward()
            value_loss.backward()
            policy_optimizer.step()
            value_optimizer.step()
        
        # 调整学习率
        ...

    ...

if __name__ == '__main__':
    policynet, valuenet = default_net()
    train(policynet, valuenet)
    save_default_net(policynet, valuenet)
