import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from board import Board
from tqdm import tqdm

class Env:
    device      = torch.device('cuda')
    # device      = torch.device('cpu')
    board_shape = (15, 15)
    board_sz    = board_shape[0]*board_shape[1]
    # train parameter
    datapool_sz = ...
    batch_sz    = ...
    epochs      = ...

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

def init_datapool():
    ...

def train(policynet:nn.Module, valuenet:nn.Module):
    init_datapool()
    policynet.train()
    valuenet.train()
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion  = nn.MSELoss()
    policy_optimizer = optim.SGD(policynet.parameters(), lr=0.001, momentum=0.9)
    value_optimizer = optim.SGD(valuenet.parameters(), lr=0.001, momentum=0.9)

    for epoch in tqdm(range(Env.epochs)):
        # 清空梯度
        policy_optimizer.zero_grad()
        value_optimizer.zero_grad()

        # forward, backward, optimize
        inputs, policy_target, value_target = ...
        policy_out = policynet(inputs)
        value_out  = valuenet(inputs)
        policy_loss = policy_criterion(policy_out, policy_target)
        value_loss  = value_criterion(value_out, value_target)
        policy_loss.backward()
        value_loss.backward()
        policy_optimizer.step()
        value_optimizer.step()
        
        # 调整学习率
        ...

        # torch.save(policynet.state_dict(), './model/policynet18.pt')
        # torch.save(valuenet.state_dict(), './model/valuenet18.pt')
    ...

if __name__ == '__main__':
    policynet, valuenet = default_net()

    # policynet.eval()
    # valuenet.eval()

    train(policynet, valuenet)
    torch.save(policynet.state_dict(), './model/policynet18.pt')
    torch.save(valuenet.state_dict(), './model/valuenet18.pt')
