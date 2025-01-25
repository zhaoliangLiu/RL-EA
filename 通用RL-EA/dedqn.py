import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import math

class DifferentialEvolution:
    """完整的差分进化实现"""
    def __init__(self, dim, bounds, npop, strategy_pool):
        self.dim = dim
        self.bounds = np.array(bounds)
        self.npop = npop
        self.strategies = strategy_pool
        self.H = 5  # 历史记忆大小
        self.M_F = [0.5] * self.H
        self.M_CR = [0.5] * self.H
        self.S_F = []
        self.S_CR = []
        self.archive = []
        self.survival = np.ones(npop)  # 公式15
        
        # 初始化种群
        self.population = self.initialize_population()
        self.best = None  # 确保best初始化为None
        
    def initialize_population(self):
        """公式2：种群初始化"""
        pop = np.zeros((self.npop, self.dim))
        for d in range(self.dim):
            l, u = self.bounds[d]
            pop[:, d] = l + np.random.rand(self.npop) * (u - l)
        return pop
    
    def evaluate_fitness(self, func):
        """修正后的适应度评估方法"""
        self.fitness = np.array([func(ind) for ind in self.population])
        self.best_idx = np.argmin(self.fitness)
        self.best = self.population[self.best_idx].copy()  # 明确拷贝最优个体
    
    def mutation(self, strategy, F):
        """修正后的变异方法"""
        mutants = []
        for i in range(self.npop):
            # 确保选择5个不同的索引
            candidates = [j for j in range(self.npop) if j != i]
            selected = np.random.choice(candidates, 5, replace=False)
            r1, r2, r3, r4, r5 = selected
            
            current = self.population[i]
            if strategy == 'rand/1':
                mutant = self.population[r1] + F*(self.population[r2]-self.population[r3])
            elif strategy == 'best/1':
                mutant = self.best + F*(self.population[r1]-self.population[r2])
            elif strategy == 'rand/2':
                mutant = self.population[r1] + F*(self.population[r2]-self.population[r3]) + \
                         F*(self.population[r4]-self.population[r5])
            elif strategy == 'best/2':
                mutant = self.best + F*(self.population[r1]-self.population[r2]) + \
                         F*(self.population[r3]-self.population[r4])
            elif strategy == 'current-to-rand/1':
                mutant = current + F*(self.population[r1]-current) + \
                         F*(self.population[r2]-self.population[r3])
            elif strategy == 'current-to-best/1':
                # 显式检查维度
                assert self.best.shape == current.shape, \
                    f"维度不匹配: best {self.best.shape}, current {current.shape}"
                mutant = current + F*(self.best-current) + \
                         F*(self.population[r1]-self.population[r2])
            else:
                raise ValueError("未知变异策略")
            
            mutant = np.clip(mutant, self.bounds[:,0], self.bounds[:,1])
            mutants.append(mutant)
        return np.array(mutants)

    def crossover(self, mutant, CR):
        """修正后的交叉方法"""
        trials = np.copy(self.population)
        for i in range(self.npop):
            # 确保至少有一个维度交叉
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(self.dim)] = True
            trials[i] = np.where(cross_points, mutant[i], self.population[i])
        return trials
    
    def selection(self, trial, func):
        """公式10：贪婪选择"""
        new_pop = []
        new_fitness = []
        success_F = []
        success_CR = []
        
        for i in range(self.npop):
            trial_fit = func(trial[i])
            if trial_fit < self.fitness[i]:
                new_pop.append(trial[i])
                new_fitness.append(trial_fit)
                success_F.append(self.current_F[i])
                success_CR.append(self.current_CR[i])
                self.survival[i] += 1  # 公式15更新
            else:
                new_pop.append(self.population[i])
                new_fitness.append(self.fitness[i])
                self.survival[i] = 1
        
        # 更新档案
        self.archive.extend(self.population[np.array(new_fitness) >= self.fitness])
        if len(self.archive) > self.npop:
            self.archive = random.sample(self.archive, self.npop)
        
        self.population = np.array(new_pop)
        self.fitness = np.array(new_fitness)
        return success_F, success_CR
    
    def adapt_parameters(self, success_F, success_CR):
        """公式13-14, 24-25：参数自适应"""
        if len(success_F) > 0:
            weights = np.abs(self.fitness - self.prev_fitness)[:len(success_F)]
            weights /= np.sum(weights) + 1e-8
            mean_F = np.sum(weights * np.array(success_F)**2) / (np.sum(weights * np.array(success_F)) + 1e-8)
            self.M_F = np.roll(self.M_F, -1)
            self.M_F[-1] = mean_F
        
        if len(success_CR) > 0:
            weights = np.abs(self.fitness - self.prev_fitness)[:len(success_CR)]
            weights /= np.sum(weights) + 1e-8
            mean_CR = np.sum(weights * np.array(success_CR))
            self.M_CR = np.roll(self.M_CR, -1)
            self.M_CR[-1] = mean_CR
    
    def generate_parameters(self):
        """生成F和CR参数（公式11-12）"""
        self.current_F = np.zeros(self.npop)
        self.current_CR = np.zeros(self.npop)
        for i in range(self.npop):
            idx = np.random.randint(0, self.H)
            # 从Cauchy分布生成F
            self.current_F[i] = np.clip(np.random.standard_cauchy()*0.1 + self.M_F[idx], 0, 1)
            # 从正态分布生成CR
            self.current_CR[i] = np.clip(np.random.normal(loc=self.M_CR[idx], scale=0.1), 0, 1)

class FitnessLandscapeAnalyzer:
    """完整的适应度景观分析"""
    def __init__(self, dim, bounds, walk_steps=100):
        self.dim = dim
        self.bounds = np.array(bounds)  # 添加边界参数
        self.walk_steps = walk_steps
        self.epsilon_levels = np.array([0, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1])
    
    def random_walk(self, population):
        """修正后的随机漫步方法"""
        min_vals = np.min(population, axis=0)
        max_vals = np.max(population, axis=0)
        step_size = max_vals - min_vals
        
        walk = [population[np.random.choice(len(population))]]
        for _ in range(self.walk_steps-1):
            step = step_size * np.random.randn(self.dim) * 0.1
            new_point = walk[-1] + step
            new_point = np.clip(new_point, self.bounds[:,0], self.bounds[:,1])
            walk.append(new_point)
        return np.array(walk)
    
    def calculate_features(self, population, fitness_values, global_opt):
        """修正后的特征计算方法（添加fitness参数）"""
        # 特征1：FDC（公式12）
        distances = np.linalg.norm(population - global_opt, axis=1)
        fdc = np.corrcoef(fitness_values, distances)[0,1]  # 使用传入的fitness_values
        
        # 特征2：RIE（公式13-19）
        walk = self.random_walk(population)
        walk_fitness = np.array([np.sum(p**2) for p in walk])
        delta_f = np.abs(np.diff(walk_fitness))
        epsilon_star = np.max(delta_f)
        RIE = 0
        for epsilon in self.epsilon_levels * epsilon_star:
            S = []
            for i in range(1, len(walk_fitness)):
                diff = walk_fitness[i] - walk_fitness[i-1]
                if diff < -epsilon:
                    S.append(-1)
                elif diff > epsilon:
                    S.append(1)
                else:
                    S.append(0)
            
            # 计算熵（公式15）
            transitions = {}
            for i in range(len(S)-1):
                pair = (S[i], S[i+1])
                transitions[pair] = transitions.get(pair, 0) + 1
            
            total = sum(transitions.values())
            entropy = 0
            for count in transitions.values():
                p = count / total
                entropy -= p * math.log(p + 1e-8)
            RIE = max(RIE, entropy)
        
        # 特征3：自相关系数（公式20-21）
        r1 = np.correlate(walk_fitness - np.mean(walk_fitness), 
                         walk_fitness - np.mean(walk_fitness), mode='full')
        r1 = r1[len(r1)//2 + 1] / (np.var(walk_fitness)*len(walk_fitness))
        
        # 特征4：NNUM（公式22-23）
        best = walk[np.argmin(walk_fitness)]
        distances = np.linalg.norm(walk - best, axis=1)
        sorted_idx = np.argsort(distances)
        chi = 0
        for i in range(1, len(sorted_idx)):
            if walk_fitness[sorted_idx[i]] <= walk_fitness[sorted_idx[i-1]]:
                chi += 1
        NNUM = chi / len(sorted_idx)
        
        return np.array([fdc, RIE, r1, NNUM])

class DQN(nn.Module):
    """深度Q网络（4层结构）"""
    def __init__(self, input_size, hidden_size=10, output_size=3):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

class DEDQN:
    """完整的DEDQN算法实现"""
    def __init__(self, dim, bounds, npop, strategies, 
                 memory_size=100000, batch_size=32, gamma=0.95):
        self.de = DifferentialEvolution(dim, bounds, npop, strategies) 
        self.fla = FitnessLandscapeAnalyzer(dim, bounds)  # 添加bounds参数
        self.strategies = strategies
        
        # DQN参数
        self.dqn = DQN(input_size=4, output_size=len(strategies))
        self.target_dqn = DQN(input_size=4, output_size=len(strategies))
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # 训练参数
        self.global_opt = np.zeros(dim)  # 假设已知全局最优解
    def get_state(self, func):   
         # 需要先计算当前适应度
        current_fitness = np.array([func(ind) for ind in self.de.population])
        return self.fla.calculate_features(
            self.de.population,
            current_fitness,  # 传入当前适应度值
            self.global_opt
        )
    def calculate_reward(self):
        """计算进化效率奖励（公式15-17）"""
        e = []
        for i in range(self.de.npop):
            if self.de.fitness[i] < self.de.prev_fitness[i]:
                e.append(1 / self.de.survival[i])
            else:
                e.append(0)
        return np.mean(e)
    
    def train_dqn(self):
        """DQN训练"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        
        # 计算当前Q值
        current_q = self.dqn(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_dqn(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q
        
        # 计算损失
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def evolve(self, func, generations, FES_max=1e4):
        """主优化流程"""
        self.de.evaluate_fitness(func)
        best_fitness = [np.min(self.de.fitness)]
        
        for gen in range(generations):
            # 获取当前状态
            state = self.get_state(func)
            
            # 选择策略
            if np.random.rand() < self.epsilon:
                action = np.random.choice(len(self.strategies))
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state)
                    q_values = self.dqn(state_tensor)
                    action = torch.argmax(q_values).item()
            strategy = self.strategies[action]
            
            # 生成控制参数
            self.de.generate_parameters()
            
            # 变异和交叉
            mutants = self.de.mutation(strategy, self.de.current_F)
            trials = self.de.crossover(mutants, self.de.current_CR)
            
            # 选择
            self.de.prev_fitness = np.copy(self.de.fitness)
            success_F, success_CR = self.de.selection(trials, func)
            
            # 更新参数
            self.de.adapt_parameters(success_F, success_CR)
            
            # 计算奖励
            reward = self.calculate_reward()
            
            # 获取新状态
            next_state = self.get_state(func)
            
            # 存储经验
            self.memory.append((state, action, reward, next_state))
            
            # DQN训练
            self.train_dqn()
            
            # 同步目标网络
            if gen % 10 == 0:
                self.target_dqn.load_state_dict(self.dqn.state_dict())
            
            # 记录最佳适应度
            best_fitness.append(np.min(self.de.fitness))
            if (gen+1) % 10 == 0:
                print(f"Generation {gen+1}, Best Fitness: {best_fitness[-1]:.4e}")
        
        return best_fitness

# # 使用示例
if __name__ == "__main__":
    # 参数设置
    dim = 30
    bounds = [(-100, 100)] * dim
    npop = 30
    strategies = ['rand/1', 'best/1', 'current-to-best/1']
    
    # 初始化算法（假设已知全局最优在原点）
    dedqn = DEDQN(dim, bounds, npop, strategies)
    dedqn.global_opt = np.zeros(dim)
    
    # 定义测试函数
    def sphere(x):
        return np.sum(np.square(x))
    
    # 运行优化
    fitness_history = dedqn.evolve(sphere, generations=1000)
    import matplotlib.pyplot as plt
    plt.plot(fitness_history)
    plt.yscale('log')
    plt.xlabel('Generation')
    print("\n优化完成!")
    print("最终最佳适应度:", fitness_history[-1])