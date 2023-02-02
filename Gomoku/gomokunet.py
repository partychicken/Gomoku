import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from board import Board

class Env:
    device      = torch.device('cuda')
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

class GomokuNet(nn.Module):
    ...

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    if not os.path.exists('./model/'):
        os.makedirs('./model/')

    policynet = models.resnet18()
    valuenet  = models.resnet18()
    # resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    policynet.fc = nn.Sequential(nn.Linear(512, Env.board_sz), nn.Softmax(dim=1))
    valuenet.fc  = nn.Sequential(nn.Linear(512, 1), nn.Tanh())
    # resnet18.fc = nn.Sequential(nn.Linear(512, 15*15+1), nn.Softmax(dim=1))

    if os.path.exists('./model/policynet.pt'):
        policynet.load_state_dict(torch.load('./model/policynet.pt'))
    if os.path.exists('./model/valuenet.pt'):
        valuenet.load_state_dict(torch.load('./model/valuenet.pt'))

    policynet.to(Env.device)
    valuenet.to(Env.device)

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
    # print(x)
    # print(y)
    # print(x.size())
    # print(y.size())
    pp = x.clone().detach()
    for i in range(0, Env.board_shape[0]):
        for j in range(0, Env.board_shape[1]):
            if b[0][0][i][j] == 1 or b[0][1][i][j] == 1:
                pp[0][i*Env.board_shape[0]+j] = 0
    pp /= torch.sum(pp)
    print(x)
    print(pp)
    z = y.item()
    print(z)
    # print(x.data)
    # print(torch.sum(x.data))
    # print(x.data[0][15*15])
    ...
    # torch.save(resnet18.state_dict(), './model/resnet18.pt')
