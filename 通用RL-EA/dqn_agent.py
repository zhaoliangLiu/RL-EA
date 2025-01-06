# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import namedtuple, deque

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        # 定义网络结构
        self.fc1 = nn.Linear(state_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)
        # self.fc5 = nn.Softmax(dim=1)
    
    def forward(self, state):
        # 前向传播
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        # x = self.fc5(x)

        return x

class DQNAgent():
    def __init__(self, state_size, action_size, batch_size=64, gamma=0.99,
                 lr=1e-3, update_every=20):
        self.state_size = state_size  # 状态维度
        self.action_size = action_size  # 动作数量 
        
        # 初始化Q网络和目标网络
        self.qnetwork_local = DQN(state_size, action_size)
        self.qnetwork_target = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        # 经验回放池
        self.memory = ReplayBuffer(action_size, buffer_size=int(1e6), batch_size=batch_size)
        self.t_step = 0
        self.gamma = gamma  # 折扣因子
        self.batch_size = batch_size
        self.update_every = update_every  # 网络更新频率
    
    def act_test(self, state, eps=0):
        # 根据当前策略选择动作
        state = torch.from_numpy(state).float()
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            # 选择最优动作
            return np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            # 随机选择动作
            return np.random.choice(np.arange(self.action_size), size=state.shape[0])

 

    def act(self, state, tau=1.0):
        """选择动作，使用softmax策略"""

        # 将状态转换为tensor
        state = torch.from_numpy(state).float()

        # 用网络计算动作值
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        action_values = action_values.cpu().data.numpy()
        # 标准化Q值

        action_values = action_values - np.max(action_values, axis=1, keepdims=True)
        action_values = np.clip(action_values, -10, 10)  # 限制Q值范围
        
        exp_values = np.exp(action_values / tau)
        action_probabilities = exp_values / (np.sum(exp_values, axis=1, keepdims=True) + 1e-10)
        
        # 确保概率分布有效
        action_probabilities = np.nan_to_num(action_probabilities, 0)
        action_probabilities = action_probabilities / np.sum(action_probabilities, axis=1, keepdims=True)
        
        action_choices = [np.random.choice(self.action_size, p=probs) for probs in action_probabilities]
        return np.array(action_choices)
    
    def step(self, state, action, reward, next_state, done, update_freq=20):
        # 存储经验
        self.memory.add(state, action, reward, next_state, done)
        
        # 按照更新频率进行学习
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, update_freq=update_freq) # 每隔update_freq学习一次
    
    def learn(self, experiences, update_freq=20):
        # 从经验中学习
        states, actions, rewards, next_states, dones = experiences
        
        # 计算目标Q值
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # 计算当前Q值
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # 计算损失
        loss = nn.MSELoss()(Q_expected, Q_targets) + 1e-6
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪 
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()
        
        # 软更新目标网络参数
        if self.t_step % update_freq == 0:
            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
            # self.soft_update(self.qnetwork_local, self.qnetwork_target)
    
    def soft_update(self, local_model, target_model, tau=1e-3):
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # 经验回放池
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
   
    
    def add(self, state, action, reward, next_state, done):
        # 添加新经验
        for i in range(len(action)):
            e = self.experience(state[i], action[i], reward[i], next_state[i], done[i])
            self.memory.append(e)
    
    def sample(self):
        # 随机采样一批经验
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)
