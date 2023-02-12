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
    l = len(states)
    for index in range(l//2, l):
        state = states[index]
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

def init_datapool_by_selfplay(datapool_sz:int, device, msg_que:Queue, result_que):
    try:
        print('子进程%d启动' %os.getpid())
        policynet, valuenet = default_net(device=device)
        with torch.no_grad():
            now_sz = 0
            while now_sz < datapool_sz:
                print(('[%d]a selfplay starts(' + str(device) + ').') %os.getpid())
                x1, y1 = get_selfplay_data(policynet, valuenet, device)
                # l = len(x1)//2
                # x1 = x1[l:]
                # y1 = y1[l:]
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

def get_local_datapool():
    os.chdir(os.path.dirname(__file__))
    if not os.path.exists('./train/'):
        os.makedirs('./train/')
    x, y = [], []
    if os.path.exists('./train/datapool_inputs.pt')\
        and os.path.exists('./train/datapool_targets.pt'):
        x = torch.load('./train/datapool_inputs.pt')
        y = torch.load('./train/datapool_targets.pt')
        assert len(x) == len(y)
    return x, y

def save_datapool(x, y):
    os.chdir(os.path.dirname(__file__))
    if not os.path.exists('./train/'):
        os.makedirs('./train/')
    torch.save(x, './train/datapool_inputs.pt')
    torch.save(y, './train/datapool_targets.pt')

def train(policynet:nn.Module, valuenet:nn.Module, data_loader:DataLoader):
    policynet.train()
    valuenet.train()
    p_criterion = nn.CrossEntropyLoss()
    v_criterion  = nn.MSELoss()
    default_lr = 0.01
    # default_lr = 1e-5
    p_optimizer = optim.SGD(policynet.parameters(), lr=default_lr, momentum=0.9)
    v_optimizer = optim.SGD(valuenet.parameters() , lr=default_lr, momentum=0.9)
    p_scheduler = optim.lr_scheduler.ExponentialLR(p_optimizer, gamma=0.9)
    v_scheduler = optim.lr_scheduler.ExponentialLR(v_optimizer, gamma=0.9)

    # lr_and_loss = []

    for epoch in tqdm(range(Env.epochs)):
        avg_p_loss, avg_v_loss = 0, 0
        for index, (inputs, p_targets, v_targets) in enumerate(data_loader):
            inputs = inputs.to(Env.device)
            p_targets = p_targets.to(Env.device)
            v_targets = v_targets.to(Env.device)

            # 清空梯度
            p_optimizer.zero_grad()
            v_optimizer.zero_grad()

            # forward, backward, optimize
            # inputs, policy_targets, value_targets = ...
            p_out = policynet(inputs)
            v_out = valuenet(inputs)
            p_loss = p_criterion(p_out, p_targets)
            v_loss = v_criterion(v_out, v_targets)
            avg_p_loss += p_loss.item()
            avg_v_loss += v_loss.item()
            # print(('[%d %d] '+ str(p_loss.item()) + ', ' + str(v_loss.item()))\
            #      %(epoch, index))
            p_loss.backward()
            v_loss.backward()
            p_optimizer.step()
            v_optimizer.step()
        
            # policynet.eval()
            # valuenet.eval()
            # p_optimizer.zero_grad()
            # v_optimizer.zero_grad()
            # p_out = policynet(inputs)
            # v_out = valuenet(inputs)
            # p_loss = p_criterion(p_out, p_targets)
            # v_loss = v_criterion(v_out, v_targets)
            
            # lr_and_loss.append((p_optimizer.param_groups[0]['lr'], p_loss, v_loss))
            # policynet.train()
            # valuenet.train()

        # 调整学习率
        p_scheduler.step()
        v_scheduler.step()

        avg_p_loss /= len(data_loader)
        avg_v_loss /= len(data_loader)
        print('[%d] avg loss (%f, %f)' %(epoch, avg_p_loss, avg_v_loss))

    # for item in lr_and_loss:
    #     print(item)

if __name__ == '__main__':
    # import cProfile, pstats
    # p = cProfile.Profile()
    # p.enable()

    torch.set_num_threads(os.cpu_count()//2)
    for it in range(1):
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
                pool.apply_async(init_datapool_by_selfplay\
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
        # init_datapool_by_selfplay(Env.datapool_sz, Env.device, que)

        # x, y = [], []
        x, y = get_local_datapool()
        while not result_que.empty():
            xx, yy = result_que.get()
            x += xx
            y += yy
        lenx = len(x)
        if lenx > Env.tot_datapool_sz:
            x = x[lenx-Env.tot_datapool_sz:]
            y = y[lenx-Env.tot_datapool_sz:]
        print('real new dataset pool size = %d' %(len(x)))
        
        dataset = MyDataset(x, y)
        data_loader = DataLoader(dataset=dataset, batch_size=Env.batch_sz\
            , shuffle=True, num_workers=Env.num_workers)
        policynet, valuenet = default_net()
        train(policynet, valuenet, data_loader)
        save_default_net(policynet, valuenet)
        save_datapool(x, y)

    # p.disable()
    # stats = pstats.Stats(p).sort_stats('tottime')
    # stats.print_stats(10)
