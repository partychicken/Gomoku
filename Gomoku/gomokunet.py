import os
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
import multiprocessing
from multiprocessing import Pool
from queue import Queue

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index][0], self.y[index][1]

def get_selfplay_data(policynet:nn.Module, valuenet:nn.Module, device):
    p0 = GomokuAI('p0', policynet, valuenet, device, True)
    p1 = GomokuAI('p1', policynet, valuenet, device, True)
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

def init_datapool(datapool_sz:int, device, msg_que:Queue, result_que):
    try:
        print('子进程%d启动' %os.getpid())
        policynet, valuenet = default_net(device=torch.device('cpu'))
        with torch.no_grad():
            now_sz = 0
            while now_sz < datapool_sz:
                print(('[%d]a selfplay starts(' + str(device) + ').') %os.getpid())
                x1, y1 = get_selfplay_data(policynet, valuenet, device)
                print(('[%d]a selfplay finished(' + str(device) + ').') %os.getpid())
                data_augmentation(x1, y1)
                print('[%d]an augmentation finished.' %os.getpid())

                result_que.put((x1, y1))
                now_sz = msg_que.get()
                now_sz += len(x1)
                msg_que.put(now_sz)

                print(('[%d]datapool_sz = ' + str(now_sz)) %os.getpid())
            print('[%d]data collection finished.' %os.getpid())
    except KeyboardInterrupt:
        print('子进程%d中断' %os.getpid())

def train(policynet:nn.Module, valuenet:nn.Module, data_loader:DataLoader):
    policynet.train()
    valuenet.train()
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion  = nn.MSELoss()
    policy_optimizer = optim.SGD(policynet.parameters(), lr=0.001, momentum=0.9)
    value_optimizer = optim.SGD(valuenet.parameters(), lr=0.001, momentum=0.9)

    for epoch in tqdm(range(Env.epochs)):
        for index, (inputs, policy_targets, value_targets) in enumerate(data_loader):
            inputs = inputs.to(Env.device)
            policy_targets = policy_targets.to(Env.device)
            value_targets = value_targets.to(Env.device)

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
    import cProfile, pstats
    p = cProfile.Profile()
    p.enable()

    # 多进程收集数据
    try:
        # cpu_cnt = os.cpu_count()
        # if cpu_cnt is None:  cpu_cnt = 1
        cpu_cnt = 3
        print('进程池大小%d' %cpu_cnt)
        pool = Pool(cpu_cnt)
        manager = multiprocessing.Manager()
        msg_que = manager.Queue()
        msg_que.put(0)
        result_que = manager.Queue()
        x, y = [], []
        for i in range(cpu_cnt):
            # 传入自对弈模型使用的设备，cpu或cuda
            device = Env.device if i == 0 else torch.device('cpu')
            # device = torch.device('cpu')
            pool.apply_async(init_datapool\
                , args=(Env.datapool_sz, device, msg_que, result_que))
        print('等待子进程执行完毕')
        pool.close()
        pool.join()
        print('所有子进程执行完毕')
    except KeyboardInterrupt:
        pid = os.getpid()
        print('主进程%d中断' %pid)
        os.popen('taskkill.exe /f /pid:%d' %pid)
    
    # que = Queue()
    # init_datapool(Env.datapool_sz, Env.device, que)

    x, y = [], []
    while not result_que.empty():
        xx, yy = result_que.get()
        x += xx
        y += yy
    print('real dataset pool size = %d' %(len(x)))
    dataset = MyDataset(x, y)
    data_loader = DataLoader(dataset=dataset, batch_size=Env.batch_sz\
        , shuffle=True, num_workers=Env.num_workers)
    policynet, valuenet = default_net()
    train(policynet, valuenet, data_loader)
    save_default_net(policynet, valuenet)

    p.disable()
    stats = pstats.Stats(p).sort_stats('tottime')
    stats.print_stats(10)