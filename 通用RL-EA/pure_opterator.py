import numpy as np

def exploration_operator(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    DE/current-to-rand/1: F = 0.8, K = 0.8
    向当前个体探索
    """
    F = 0.8
    K = 0.8
    
    available_indices = [i for i in range(len(population)) if i != individual_id]
    r1, r2, r3 = np.random.choice(available_indices, 3, replace=False)
    
    new_individual = population[individual_id] + K * (population[r1] - population[individual_id]) + F * (population[r2] - population[r3])
    return new_individual, velocities

import numpy as np

def exploitation_operator(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    PSO: 基于粒子群优化的纯粹开发算子
    """
    w = 0.5  # 惯性权重
    c1 = 1.5  # 认知系数
    c2 = 1.5  # 社会系数
    
    r1 = np.random.random()
    r2 = np.random.random()
    
    new_velocity = (w * velocities +
                    c1 * r1 * (pBest[individual_id] - population[individual_id]) +
                    c2 * r2 * (gBest - population[individual_id]))
    
    new_individual = population[individual_id] + new_velocity
    return new_individual, new_velocity

def test_dqn_agent1(population, num_particles, dim, max_iter, fitness_function, arange):
    from dqn_agent import DQNAgent
    import os 
    from utils import compute_state
    import torch 
    # 1是开发，0是探索
    exploitation_action = []  
    exploration_action = []
    

    agent1 = DQNAgent(state_size=6, action_size=2) 
    
    if not os.path.exists('models/agent1.pth') or not os.path.exists('models/agent2.pth'):
        raise FileNotFoundError("请先训练模型，确保 'models/agent1.pth' 和 'models/agent2.pth' 文件存在。")
    
    agent1.qnetwork_local.load_state_dict(torch.load('models/agent1.pth')) 
    # 参数设置
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
        action1 = agent1.act_test(state_features, 0)
        action1 = np.array(action1)

      
        # 应用选择的算子
        new_population = []
        new_velocities = []
        exploitation = 0
        exploration = 0
        for i in range(pop_size):
            # 1是开发，0是探索
            if (action1[i] == 0):
                exploration += 1
                selected_operator = exploration_operator
            else:
                exploitation += 1
                selected_operator = exploitation_operator
            new_individual, new_velocity = selected_operator(
                i, population, pBest, gBest, 
                fitness_function, iter, max_iter, velocities[i]
            )
            new_individual = np.clip(new_individual, -100, 100)
            new_population.append(new_individual)
            new_velocities.append(new_velocity)
        exploitation_action.append(exploitation)
        exploration_action.append(exploration)

        pre_population = np.copy(population)
        population = np.array(new_population)
        population = np.clip(population, arange[0], arange[1])
        velocities = np.array(new_velocities)
        velocities = np.clip(velocities, arange[0], arange[1]) * 0.15
        
        # 更新最优解
        fitness_population = fitness_function(population)
        update = fitness_population < fitness_pBest
        pBest[update] = population[update]
        fitness_pBest[update] = fitness_population[update]
        count_k = [count_k[i] + 1 if update[i] else count_k[i] for i in range(pop_size)]
        
        if np.min(fitness_pBest) < fitness_gBest:
            gBest = pBest[np.argmin(fitness_pBest)]
            fitness_gBest = fitness_function([gBest])[0]
            
        fitness_history.append(fitness_gBest)
         
        
    return gBest,fitness_gBest, fitness_history, exploration_action, exploitation_action
 

def test_pure_opterator():
    from fitness_function.cec2017.functions import all_functions
    
    pop_size =  50
    dim = 10
    max_iter = 1000
    arange = (-100, 100) 
    population = np.random.uniform(arange[0], arange[1], (pop_size, dim))
    fitness_function_id = 1
    func = all_functions[fitness_function_id]
    gBest,fitness_gBest, fitness_history, exploration_action, exploitation_action = \
    test_dqn_agent1(population, pop_size, dim, max_iter, func, arange)

    from plot_with_agent1 import plot_data
    rate = 0.01
    plot_data(exploration_action, exploitation_action,rate)


if __name__ == '__main__':
    test_pure_opterator()
