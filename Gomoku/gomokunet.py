import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from board import Board

class Env:
    # device      = torch.device('cuda')
    device      = torch.device('cpu')
    board_shape = (15, 15)
    board_sz    = board_shape[0]*board_shape[1]

def board_to_tensor(board:Board):
    b = board.board.flatten()
    b1 = np.array([1 if x == 0 else 0 for x in b]).reshape(Env.board_shape)
    b2 = np.array([1 if x == 1 else 0 for x in b]).reshape(Env.board_shape)
    b3 = np.ones(shape=Env.board_shape, dtype=np.int8) \
        if board.turn == 1 else np.zeros(shape=Env.board_shape,dtype=np.int8)
    t = torch.tensor(np.array([b1, b2, b3]), device=Env.device).float()
    t.unsqueeze_(0)
    return t

def default_net():
    os.chdir(os.path.dirname(__file__))
    if not os.path.exists('./model/'):
        os.makedirs('./model/')
    policynet = models.resnet18()
    valuenet  = models.resnet18()
    policynet.fc = nn.Sequential(nn.Linear(512, Env.board_sz), nn.Softmax(dim=1))
    valuenet.fc  = nn.Sequential(nn.Linear(512, 1), nn.Tanh())
    if os.path.exists('./model/policynet18.pt'):
        policynet.load_state_dict(torch.load('./model/policynet18.pt'))
    if os.path.exists('./model/valuenet18.pt'):
        valuenet.load_state_dict(torch.load('./model/valuenet18.pt'))
    policynet.to(Env.device)
    valuenet.to(Env.device)
    return policynet, valuenet

if __name__ == '__main__':
    policynet, valuenet = default_net()

    policynet.eval()
    valuenet.eval()

    # resnet18.train()
    # print(resnet18)
    # resnet18.eval()
    # b = torch.from_numpy(Board().board).float()
    # b = torch.ones(size=(15,15))
    # b.unsqueeze_(0)
    # b.unsqueeze_(0)
    bd = Board()
    bd[0][2] = 1
    bd[1][0] = 0
    b = board_to_tensor(bd)
    # print(b)
    x, y = policynet(b), valuenet(b)
    print(x)
    z = y.item()
    print(z)
    # print(x.data)
    # print(torch.sum(x.data))
    # print(x.data[0][15*15])
    ...
    torch.save(policynet.state_dict(), './model/policynet18.pt')
    torch.save(valuenet.state_dict(), './model/valuenet18.pt')
