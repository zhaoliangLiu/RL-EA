 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from simple_function.simple_functions import all_functions
from fitness_function.cec2017.functions import all_functions
from utils import compute_state
from dqn_agent import DQNAgent
from LLH10 import operators
# from operators import operators
import random
import os

# 将 operators 转换为字典形式
operators_dict = {f'o{i+1}': operators[i] for i in range(len(operators))}
operator_keys = list(operators_dict.keys())

# 参数设置
pop_size = 50                # 种群大小
dimension = 10               # 维度
max_iter = 50              # 最大迭代次数
evaluation_iter = 2000             
start_functions = 1
num_functions = len(all_functions) - 1 - 20       
# esplion-greedy
start_esp = 0.9
end_esp = 0.01
# 软策略更新
start_tau = 10
end_tau = 0.1
step = 0
lr=0.00001
update_freq = 100
arange = (-100, 100)

# 初始化智能体
agent1 = DQNAgent(state_size=6, action_size=2, lr=lr)  # agent1，决定探索或开发
agent2 = DQNAgent(state_size=8, action_size=len(operator_keys), lr=lr)  # agent2，选择算子

# 主训练循环
for iter in range(max_iter):
    count_dqn = {f'o{i+1}': 0 for i in range(len(operator_keys))}
    # 随机选择一个适应度函数
    func_id = np.random.randint(start_functions, start_functions + num_functions)
    fitness_function = all_functions[func_id]
     
    # 初始化种群
    population = np.random.uniform(arange[0], arange[1], (pop_size, dimension))
    pre_population = np.copy(population)
    velocities = np.random.uniform(arange[0], arange[1], (pop_size, dimension)) * 0.15
    fitness_0 = fitness_function(population)
    diversity_0 = np.linalg.norm(population, axis=1)
    pBest = np.copy(population)                                                                                        
    fitness_pBest = fitness_function(pBest)
    gBest = pBest[np.argmin(fitness_pBest)]
    fitness_gBest    = fitness_function([gBest])
    
    # 初始化当前算子（初始随机选择）
    current_operator = np.random.choice(operator_keys, size=pop_size)
    count_k = [0] * pop_size    
    count_k = np.array(count_k)

    # 记录一下每个回合的奖励
    # rewards = np.zeros((pop_size, evaluation_iter))

    for eval_iter in range(evaluation_iter):
        old_pBest = np.copy(pBest)
        step += 1
        # epsilon = max(end_esp, start_esp - (start_esp - end_esp) * (step / (max_iter * evaluation_iter)))
        tau = max(end_tau, start_tau - (start_tau - end_tau) * (step / (max_iter * evaluation_iter)))
        # 计算通用状态
        state_features = compute_state(pre_population, population, pBest, gBest, fitness_function, max_iter, iter, count_k) 

        
        # agent1 选择探索(0)或开发(1)，对于每个个体
        action1 = agent1.act(state_features, tau)  # 软策略
        # action1 = agent1.act(state_features, epsilon)  # action1 的形状为 (pop_size,) epsilon-greedy策略
        
        # 构建 agent2 的状态：agent1 的动作 + 当前算子索引 + state_features
        operator_indices = np.array([operator_keys.index(op) for op in current_operator])
        state2_features = np.column_stack((action1, operator_indices, state_features))  # shape: (pop_size, 8)
        
        # agent2 选择要使用的算子，对于每个个体
        action2 = agent2.act(state2_features, tau)  #  软策略
        # action2 = agent2.act(state2_features, epsilon)  # action2 的形状为 (pop_size,) epsilon-greedy策略
        
        # 应用每个个体选择的算子
        new_population = []
        new_velocities = []
        for i in range(pop_size):
            individual = population[i]
            individual_pBest = pBest[i]
            selected_operator_key = operator_keys[action2[i]]
            count_dqn[selected_operator_key] += 1
            selected_operator = operators_dict[selected_operator_key]
     
            new_individual, new_velocity = selected_operator(i, population, pBest, gBest, fitness_function, iter, max_iter, velocities[i])
            # 修正超出边界的粒子
            new_individual = np.clip(new_individual, arange[0], arange[1])
            new_velocity = np.clip(new_velocity, arange[0], arange[1]) * 0.15
            new_population.append(new_individual)
            new_velocities.append(new_velocity)
             
        pre_population = np.copy(population)
        population = np.array(new_population)
        velocities = np.array(new_velocities)
    
        # 更新 pBest 和 gBest
        fitness_population = fitness_function(population)
        update = fitness_population < fitness_pBest
        pBest[update] = population[update]
        fitness_pBest[update] = fitness_population[update]
        count_k[update] += 1
        # 未更新的全部置为0
        count_k[~update] = 0
        
    
        # 更新 gBest 如果需要
        if np.min(fitness_pBest) < fitness_gBest:
            gBest = pBest[np.argmin(fitness_pBest)]
            fitness_gBest = fitness_function([gBest])
    
        # 计算奖励 
        pre_fitness = fitness_function(pre_population)
        #奖励为增益比
        improvement = pre_fitness - fitness_population
        reward = improvement / (pre_fitness + 1e-6) # 防止除零错误
        # 或者剪辑奖励值
        reward = np.clip(reward, -1, 1)
        # 转换为 Tensor 格式
        state_tensor = torch.from_numpy(state_features).float()
        state2_tensor = torch.from_numpy(state2_features).float()
        reward_tensor = torch.from_numpy(reward.reshape(-1, 1)).float()
        next_state_features = compute_state(pre_population, population, pBest, gBest, fitness_function, max_iter, iter, count_k)
        next_state_tensor = torch.from_numpy(next_state_features).float()

         
        # 计算next_state2的算子
        next_state2_action = agent1.act(next_state_features, tau)  # 软策略
        # next_state2_action = agent1.act(next_state_features, epsilon)  # epsilon-greedy策略
        next_state2_cur_operator = action2
        next_state2_features = np.column_stack((next_state2_action, next_state2_cur_operator, next_state_features))
        next_state2_tensor = torch.from_numpy(next_state2_features).float()
        # if torch.isnan(next_state2_tensor).any():
        #     continue
        # 优化智能体
        agent1.step(state_tensor, action1.reshape(-1, 1), reward_tensor, next_state_tensor, done=[False]*pop_size, update_freq=update_freq)

        agent2.step(state2_tensor, action2.reshape(-1, 1), reward_tensor, next_state2_tensor, done=[False]*pop_size, update_freq=update_freq)
    
        # 更新当前算子
        current_operator = np.array([operator_keys[a] for a in action2])
    
    print(f"Iteration {iter+1}/{max_iter}, function {func_id}, Best Fitness: {fitness_gBest[0]}, buffer size: {len(agent1.memory)}, count_dqn: {count_dqn}")

# 保存训练好的模型
if not os.path.exists('models'):
    os.makedirs('models')
torch.save(agent1.qnetwork_local.state_dict(), 'models/agent1.pth')
torch.save(agent2.qnetwork_local.state_dict(), 'models/agent2.pth')

print("训练完成")

 

