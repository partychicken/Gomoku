import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from board import Board

class Environment:
    device = torch.device('cuda')

def board_to_tensor(board:Board):
    b = board.board.flatten()
    b1 = np.array([1 if x == 0 else 0 for x in b]).reshape((15, 15))
    b2 = np.array([1 if x == 1 else 0 for x in b]).reshape((15, 15))
    b3 = np.ones(shape=(15,15),dtype=np.int8) \
        if board.turn == 1 else np.zeros(shape=(15,15),dtype=np.int8)
    # print(b1, b2, b3)
    t = torch.tensor(np.array([b1, b2, b3]), device=Environment.device).float()
    # print(t)
    t.unsqueeze_(0)
    return t
    # t = torch.zeros(size=(3, 15, 15), device=Environment.device)

    # b = torch.from_numpy(board.board).float().to(Environment.device)
    # b.unsqueeze_(0)
    # b.unsqueeze_(0)
    # return b

class GomokuNet(nn.Module):
    ...

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    if not os.path.exists('./model/'):
        os.makedirs('./model/')

    policynet = models.resnet18()
    valuenet  = models.resnet18()
    # resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
    policynet.fc = nn.Sequential(nn.Linear(512, 15 * 15), nn.Softmax(dim=1))
    valuenet.fc  = nn.Sequential(nn.Linear(512, 1), nn.Tanh())
    # resnet18.fc = nn.Sequential(nn.Linear(512, 15*15+1), nn.Softmax(dim=1))

    if os.path.exists('./model/policynet.pt'):
        policynet.load_state_dict(torch.load('./model/policynet.pt'))
    if os.path.exists('./model/valuenet.pt'):
        valuenet.load_state_dict(torch.load('./model/valuenet.pt'))

    policynet.to(Environment.device)
    valuenet.to(Environment.device)

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
    print(b)
    x, y = policynet(b), valuenet(b)
    print(x)
    print(y)
    z = y.item()
    print(z)
    # print(x.data)
    # print(torch.sum(x.data))
    # print(x.data[0][15*15])
    ...
    # torch.save(resnet18.state_dict(), './model/resnet18.pt')
