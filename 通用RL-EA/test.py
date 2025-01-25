import numpy as np
import torch
import matplotlib.pyplot as plt
from fitness_function.cec2017.functions import all_functions
from other_ea.clpso import clpso
from other_ea.standard_pso import standard_pso
from other_ea.jade_de import jade_de
from other_ea.sdmpso import sdms_pso
from other_ea.DEPSO import depso
from utils import compute_state
from other_ea.DE import de
from other_ea.SHDE import shade
from other_ea.RDE import rde
from dqn_agent import DQNAgent
from operators import operators
# from LLH10 import operators
import random
import os
import csv


def test_dqn_strategy(population, num_particles, dim, max_iter, fitness_function, arange):
    tau = 1
    count = {f'o{i+1}': 0 for i in range(len(operators))}
    # 将 operators 转换为字典形式
    operators_dict = {f'o{i+1}': operators[i] for i in range(len(operators))}
    operator_keys = list(operators_dict.keys())

    agent1 = DQNAgent(state_size=6, action_size=2)
    agent2 = DQNAgent(state_size=8, action_size=len(operator_keys))
    
    if not os.path.exists('models/agent1_test.pth') or not os.path.exists('models/agent2_test.pth'):
        raise FileNotFoundError("请先训练模型，确保 'models/agent1.pth' 和 'models/agent2.pth' 文件存在。")
    
    agent1.qnetwork_local.load_state_dict(torch.load('models/agent1_test.pth'))
    agent2.qnetwork_local.load_state_dict(torch.load('models/agent2_test.pth'))
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
    
    current_operator = np.random.choice(operator_keys, size=pop_size)
    count_k = [0] * pop_size
    
    for iter in range(max_iter):
        # 计算状态和动作
        state_features = compute_state(
            pre_population, population, pBest, gBest, 
            fitness_function, max_iter, iter, count_k
        )
        # action1 = agent1.act_test(state=state_features)
        action1 = agent1.act(state_features, tau=1)
        
        operator_indices = np.array([operator_keys.index(op) for op in current_operator])
        state2_features = np.column_stack((action1, operator_indices, state_features))
        # action2 = agent2.act_test(state2_features)
        action2 = agent2.act(state2_features, tau=1)
        
        # 应用选择的算子
        new_population = []
        new_velocities = []
        for i in range(pop_size):
            selected_operator_key = operator_keys[action2[i]]
            count[selected_operator_key] += 1
            selected_operator = operators_dict[selected_operator_key]
            new_individual, new_velocity = selected_operator(
                i, population, pBest, gBest, 
                fitness_function, iter, max_iter, velocities[i]
            )
            new_individual = np.clip(new_individual, -100, 100)
            new_population.append(new_individual)
            new_velocities.append(new_velocity)
            
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
        current_operator = np.array([operator_keys[a] for a in action2])
        
    return gBest,fitness_gBest, fitness_history, count
 

def test():
  
<<<<<<< HEAD
    fitness_function_id = 1
    # fitness_num = len(all_functions) - 1 
=======
    fitness_function_id = 5
    # fitness_num = len(all_functions) - 1
>>>>>>> 4b08f78d0ad876e93d5b3c2657236dd055db07b5
    fitness_num = 1
    pop_size =  50
    dim = 10
    max_iter = 50000
    arange = (-100, 100)

    res = []
    for i in range(fitness_function_id, fitness_function_id + fitness_num):
        print(f"测试函数{i}")
        fitness_function = all_functions[i]
        population = np.random.uniform(arange[0], arange[1], (pop_size, dim))
        
        gBest_dqn, fitness_gBest_dqn, fitness_history_dqn, count_dqn = test_dqn_strategy(np.copy(population), pop_size, dim, max_iter, fitness_function, arange)
        
        gBest_standard_pso, fitness_gBest_standard_pso, fitness_history_standard_pso = standard_pso(np.copy(population), pop_size, dim, max_iter, fitness_function, arange)
        gBest_jade_de, fitness_gBest_jade_de, fitness_history_jade_de = jade_de(np.copy(population), pop_size, dim, max_iter, fitness_function, arange)
        gBest_sdms, fitness_gBest_sdms, fitness_history_sdms = sdms_pso(np.copy(population), pop_size, dim, max_iter, fitness_function, arange)
        gBest_depso, fitness_gBest_depso, fitness_history_depso = depso(np.copy(population), pop_size, dim, max_iter, fitness_function, arange)
        
        gBest_shade, fitness_gBest_shade, fitness_history_shade = shade(np.copy(population), pop_size, dim, max_iter, fitness_function, arange)
        gBest_rde, fitness_gBest_rde, fitness_history_rde = rde(np.copy(population), pop_size, dim, max_iter, fitness_function, arange)
        gBest_de, fitness_gBest_de, fitness_history_de = de(np.copy(population), pop_size, dim, max_iter, fitness_function, arange)


        print(f"dqn_最优适应度: {fitness_gBest_dqn}")
        print(f"standard_pso_最优适应度: {fitness_gBest_standard_pso}")
        print(f"jade_de_最优适应度: {fitness_gBest_jade_de}")
        print(f"sdms_最优适应度: {fitness_gBest_sdms}")
        print(f"depso_最优适应度: {fitness_gBest_depso}")
        print(f"shade_最优适应度: {fitness_gBest_shade}")
        print(f"rde_最优适应度: {fitness_gBest_rde}")
        print(f"de_最优适应度: {fitness_gBest_de}")

        # 以对数缩放
        plt.yscale('log')
        plt.plot(fitness_history_dqn, label='dqn', color='black')
        plt.plot(fitness_history_standard_pso, label='standard_pso')
        plt.plot(fitness_history_jade_de, label='jade_de') 
        plt.plot(fitness_history_sdms, label='sdms_pso')
        plt.plot(fitness_history_depso, label='depso')
        plt.plot(fitness_history_shade, label='shade')
        plt.plot(fitness_history_rde, label='rde')
        plt.plot(fitness_history_de, label='de')

        plt.title(f"function_id = {i}, pop_size = {pop_size}, dim = {dim}, max_iter = {max_iter}")
        plt.legend()
        if not os.path.exists('results'):
            os.makedirs('results')
        plt.savefig(f"results/fitness_history_{i}.png")
        plt.close()

        # 记录函数的结果
        fitness_values = {
            "function_id":i,
            "dqn":fitness_gBest_dqn,
            "standard_pso":fitness_gBest_standard_pso,
            "jade_de":fitness_gBest_jade_de,
            "depso":fitness_gBest_depso,
            "sdms_pso":fitness_gBest_sdms,
            "shade":fitness_gBest_shade,
            "rde":fitness_gBest_rde,
            "de":fitness_gBest_de,
             
        } 
        # 对适应度值进行排序（从小到大）并获取排名
        sorted_methods = sorted(fitness_values.items(), key=lambda x: x[1])
        rankings = {method: rank + 1 for rank, (method, _) in enumerate(sorted_methods)}

        # 创建结果字典，同时包含适应度值和排名
        dict_res = {
            "function_id": i,
             # 添加排名
            "rank_dqn": rankings['dqn'],
            "rank_standard_pso": rankings['standard_pso'],
            "rank_jade_de": rankings['jade_de'],
            "rank_depso": rankings['depso'],
            "rank_sdms_pso": rankings['sdms_pso'],
            "rank_shade": rankings['shade'],
            "rank_rde": rankings['rde'],
            "rank_de": rankings['de'],

            "dqn": fitness_gBest_dqn,
            "standard_pso": fitness_gBest_standard_pso,
            "jade_de": fitness_gBest_jade_de,
            "depso": fitness_gBest_depso,
            "sdms_pso": fitness_gBest_sdms,
            "shade": fitness_gBest_shade,
            "rde": fitness_gBest_rde,
            "de": fitness_gBest_de,
            "count_dqn": count_dqn
        }
        res.append(dict_res)
        print(count_dqn)
     
    with open('results/summary.csv', 'w', newline='') as csvfile:
        fieldnames = res[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in res:
            writer.writerow(data)


 

if __name__ == "__main__":
    test()
    pass 
