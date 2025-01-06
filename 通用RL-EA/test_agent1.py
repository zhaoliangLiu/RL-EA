import numpy as np 
from utils import compute_state
from  dqn_agent import DQNAgent
from operators import operators
from fitness_function.cec2017.functions import all_functions
import os
import torch


def test_agent_1(population, num_particles, dim, max_iter, fitness_function, arange):
    count_action = {1:0, 0:0}
    count = {f'o{i}': 0 for i in range(len(operators))}
    # 将 operators 转换为字典形式
    operators_dict = {f'o{i+1}': operators[i] for i in range(len(operators))}
    operator_key = "o1"
    operator = operators_dict[operator_key]

    agent1 = DQNAgent(state_size=6, action_size=2, lr=0.0001) 
    
    if not os.path.exists('models/agent1.pth') or not os.path.exists('models/agent2.pth'):
        raise FileNotFoundError("请先训练模型，确保 'models/agent1.pth' 和 'models/agent2.pth' 文件存在。")
    
    agent1.qnetwork_local.load_state_dict(torch.load('models/agent1.pth')) 
    pop_size = population.shape[0]                # 种群大小
    dimension = population.shape[1]               # 维度

 
    pre_population = np.copy(population)
    velocities = np.random.uniform(arange[0], arange[1], (pop_size, dimension)) * 0.15
    fitness_0 = fitness_function(population)
    pBest = np.copy(population)
    fitness_pBest = fitness_function(pBest)
    gBest = pBest[np.argmin(fitness_pBest)]
    fitness_gBest = fitness_function([gBest])[0]
    fitness_history = [fitness_gBest]
     
    count_k = [0] * pop_size
    
    for iter in range(max_iter):
        # 计算状态和动作
        state_features = compute_state(
            pre_population, population, pBest, gBest, 
            fitness_function, max_iter, iter, count_k
        )
        # print(f"第{iter}次迭代 特征")
        # print(state_features)
        # print("============================================")
        action1 = agent1.act(state_features, 1)
        for i in range(len(action1)):
            count_action[action1[i]] += 1 

        for i in range(pop_size):
            population[i], velocities[i] = operator(i, population, pBest, gBest, fitness_function, iter, max_iter, velocities[i])


    return count_action

if __name__ == "__main__":
    population = np.random.rand(50, 10) * 200 - 100
    num_particles = 50
    dim = 10
    max_iter = 100
    arange = (-100, 100)
    fitness_function = all_functions[15]
    count_action = test_agent_1(population, num_particles, dim, max_iter, fitness_function, arange)
    print(count_action)



             
