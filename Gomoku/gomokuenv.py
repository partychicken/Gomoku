import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from board import Board

class Env:
    device      = torch.device('cuda')
    # device      = torch.device('cpu')
    board_shape = (9, 9)
    board_sz    = board_shape[0]*board_shape[1]
    net_suffix  = '_'+str(board_shape[0])+'x'+str(board_shape[1])+'.pt'

    # train parameter
    datapool_sz = 1024
    batch_sz    = 256
    epochs      = 10
    num_workers = 0

def board_to_tensor(board:Board, device = None, unsqueeze = True):
    if device is None:  device = Env.device
    b = board.board.flatten()
    b1 = np.array([1 if x == 0 else 0 for x in b]).reshape(Env.board_shape)
    b2 = np.array([1 if x == 1 else 0 for x in b]).reshape(Env.board_shape)
    b3 = np.ones(shape=Env.board_shape, dtype=np.int8) \
        if board.turn == 1 else np.zeros(shape=Env.board_shape,dtype=np.int8)
    # print(device)
    t = torch.tensor(np.array([b1, b2, b3]), device=device).float()
    # t = torch.stack([b1, b2, b3], dim=0).to(device=Env.device).float()
    if unsqueeze:  t.unsqueeze_(0)
    return t

def default_net(device = None):
    if device is None:  device = Env.device
    os.chdir(os.path.dirname(__file__))
    if not os.path.exists('./model/'):
        os.makedirs('./model/')
    policynet = models.resnet18()
    valuenet  = models.resnet18()
    policynet.fc = nn.Sequential(nn.Linear(512, Env.board_sz), nn.Softmax(dim=1))
    valuenet.fc  = nn.Sequential(nn.Linear(512, 1), nn.Tanh())
    if os.path.exists('./model/policynet'+Env.net_suffix):
        policynet.load_state_dict(torch.load('./model/policynet'+Env.net_suffix))
    if os.path.exists('./model/valuenet'+Env.net_suffix):
        valuenet.load_state_dict(torch.load('./model/valuenet'+Env.net_suffix))
    policynet.to(device)
    valuenet.to(device)
    return policynet, valuenet

def save_default_net(policynet:nn.Module, valuenet:nn.Module):
    torch.save(policynet.state_dict(), './model/policynet'+Env.net_suffix)
    torch.save(valuenet.state_dict(), './model/valuenet'+Env.net_suffix)
