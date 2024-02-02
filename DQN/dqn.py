import json
import math
import os.path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF
import torch.optim as optim

from Function.function import get_time_info


class DQNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNet, self).__init__()
        # 改成两层卷积层，两层池化层，一层全连接层处理图像
        # 两层全连接层处理info数据
        # 将上述两个结果拼接后，再进行一层全连接层
        # 最后输出
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, output_size)

    def forward(self, x):
        # x is a tuple, x[0] is image, x[1] is info

        x = torchF.relu(self.fc1(x))
        x = torchF.relu(self.fc2(x))
        x = torchF.relu(self.fc3(x))
        actions_value = self.out(x)
        return actions_value


# 定义经验回放类，并且使用tensor进行存储
class ReplayMemory(object):
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)  # list(tuple)
        state, action, reward, next_state = zip(*batch)
        return state, action, reward, next_state

    def save(self, path):
        data = [[buf[0].tolist(), int(buf[1]), int(buf[2]), buf[3].tolist()] for buf in self.buffer]
        if os.path.exists(path):
            data_last = json.load(open(path, 'r'))
            data += data_last
        json.dump(data, open(path, 'w'))

    def load(self, path):
        data = json.load(open(path, 'r'))
        self.buffer = [(np.array(buf[0]), buf[1], buf[2], np.array(buf[3])) for buf in data]

    def __len__(self):
        return len(self.buffer)


# 定义DQN智能体
class DQNAgent:
    def __init__(self, state_size, action_size,
                 learning_rate=0.001,  # 学习率
                 gamma=0.9,  # 折扣因子
                 epsilon_max=0.9,  # 最大探索概率
                 epsilon_min=0.01,  # 最小探索概率
                 epsilon_decay=1000,  # 探索概率衰减率
                 batch_size=64,  # 批次大小
                 memory_size=1000,  # 经验回放池大小
                 TAU=0.01,  # 目标网络更新率
                 memory_mode=None,  # 经验回放池模式
                 model_path=None,  # 模型路径
                 model_dict_path=None,  # 模型字典路径
                 device=None,  # 设备
                 memory_path=None,  # 经验回放池路径
                 ):
        # 保存超参数
        self.q_eval = None
        self.loss = None
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.tau = TAU

        self.memory_mode = memory_mode
        self.device = device
        # 保存状态和动作空间大小
        self.state_size = state_size
        self.action_size = action_size
        # 初始化经验回放池
        self.memory = ReplayMemory(self.memory_size)
        if memory_path:
            self.memory.load(memory_path)

        # 两个网络，一个eval_net，一个target_net，eval实时训练且更新，target每隔一段更新
        # 初始化模型
        # self.eval_net = DQNet(self.state_size, self.action_size)
        # self.target_net = DQNet(self.state_size, self.action_size)
        if model_path:
            self.eval_net = torch.load(model_path).to(self.device)
            self.target_net = torch.load(model_path).to(self.device)
        else:
            self.eval_net = DQNet(self.state_size, self.action_size).to(self.device)
            self.target_net = DQNet(self.state_size, self.action_size).to(self.device)
        # 初始化模型字典
        if model_dict_path:
            self.eval_net.load_state_dict(torch.load(model_dict_path))
        self.target_net.load_state_dict(self.eval_net.state_dict())

        # 初始化优化器和损失函数
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
        # 初始化学习步数
        self.train_step = 0
        print(
            'state size is {}, action size is {}, device is {}'.format(self.state_size, self.action_size, self.device))

    def select_action(self, state):
        eps_threshold = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * \
                        math.exp(-1. * self.train_step / self.epsilon_decay)
        self.train_step += 1
        p_rand = np.random.uniform(0, 1)
        if p_rand > eps_threshold:
            with torch.no_grad():
                ret = self.eval_net(state)
                ret = ret.max(0)
                ret = ret.indices
                return ret
        else:
            return torch.tensor([[random.randrange(self.action_size)]], device=self.device, dtype=torch.long)

    def train(self, batch_size=None, train_interval=None):
        if batch_size is None:
            batch_size = self.batch_size

        # 没有足够多的经验，不训练
        if len(self.memory) < batch_size:
            return

        # 如果没到训练间隔，那么就不训练
        if train_interval is not None:
            if self.train_step % train_interval != 0:
                return

        # 从经验回放池中抽取批次数据
        state, action, reward, next_state = self.memory.sample(batch_size)
        state = tuple(map(lambda x: torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0), state))
        action = tuple(
            map(lambda x: torch.tensor(x, dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(0), action))
        reward = tuple(
            map(lambda x: torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0), reward))
        next_state = tuple(
            map(lambda x: torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0), next_state))
        # 计算Q值
        state_batch = torch.cat(state).to(self.device)
        action_batch = torch.cat(action).to(self.device)
        reward_batch = torch.cat(reward).to(self.device)
        next_state_batch = torch.cat(next_state).to(self.device)

        # 计算Q值
        self.q_eval = self.eval_net(state_batch).gather(1, action_batch)

        # 计算目标Q值
        q_next = self.target_net(next_state_batch).detach()
        q_target = reward_batch + self.gamma * q_next.max(1)[0].view(batch_size, 1)

        # 计算损失
        self.loss = self.loss_func(self.q_eval, q_target)

        # 优化模型
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.update_target_net_from_eval()

    def save_dict(self, model_path):
        torch.save(self.eval_net.state_dict(), model_path)

    def save_model(self, model_path):
        torch.save(self.eval_net, model_path)

    def update_target_net_from_eval(self):
        target_net_dict = self.target_net.state_dict()
        eval_net_dict = self.eval_net.state_dict()
        for k in target_net_dict.keys():
            target_net_dict[k] = eval_net_dict[k] * self.tau + target_net_dict[k] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_dict)
