import numpy as np
import torch
import matplotlib.pyplot as plt
from fitness_function.cec2017.functions import all_functions
from other_ea.clpso import clpso
from other_ea.standard_pso import standard_pso
from other_ea.jade_de import jade_de
from utils import compute_state
from dqn_agent import DQNAgent
from operators import operators
# from LLH10 import operators
import random
import os
import csv


def test_dqn_strategy(population, num_particles, dim, max_iter, fitness_function, arange, individual_id=1):
    agent1_action = []
    agent2_action = []

    count = {f'o{i+1}': 0 for i in range(len(operators))}
    # 将 operators 转换为字典形式
    operators_dict = {f'o{i+1}': operators[i] for i in range(len(operators))}
    operator_keys = list(operators_dict.keys())

    agent1 = DQNAgent(state_size=6, action_size=2)
    agent2 = DQNAgent(state_size=8, action_size=len(operator_keys))
    
    if not os.path.exists('models/agent1.pth') or not os.path.exists('models/agent2.pth'):
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
        action1 = agent1.act_test(state_features, 0)
        agent1_action.append(action1[individual_id]) # 统计动作频次
        operator_indices = np.array([operator_keys.index(op) for op in current_operator])
        state2_features = np.column_stack((action1, operator_indices, state_features))
        action2 = agent2.act_test(state2_features, 0)
        
        # 应用选择的算子
        new_population = []
        new_velocities = []
        for i in range(pop_size):
            selected_operator_key = operator_keys[action2[i]]
            if i == individual_id:
                agent2_action.append(selected_operator_key)
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
        
    return gBest,fitness_gBest, fitness_history, count, agent1_action, agent2_action
 

def test():
  
    fitness_function_id = 14
    fitness_num = 1
    pop_size = 50
    dim = 10
    max_iter = 1000
    arange = (-100, 100)
    individual_id = 10 # 追踪的个体
    for i in range(fitness_function_id, fitness_function_id + fitness_num):
        print(f"测试函数{i}")
        fitness_function = all_functions[i]
        population = np.random.uniform(arange[0], arange[1], (pop_size, dim))
         
        gBest_dqn, fitness_gBest_dqn, fitness_history_dqn, count_dqn, agent1_action, agent2_action \
        = test_dqn_strategy(np.copy(population), pop_size, dim, max_iter, fitness_function, arange, individual_id)
        data = agent2_action
        # 将类别映射为数值
        categories = list(set(data))
        category_to_num = {category: num for num, category in enumerate(categories)}
        num_to_category = {num: category for category, num in category_to_num.items()}

        # 将数据转换为数值形式
        numeric_data = [category_to_num[item] for item in data]

        # 绘制数据
        #改成散点图
        plt.figure(1)
        plt.scatter(range(len(numeric_data)), numeric_data, linewidths=3)
  

        # 将 y 轴刻度替换为类别标签
        plt.yticks(range(len(categories)), categories)
        plt.title('agent2 action')
        # plt.show()
        if not os.path.exists('one_results'):
            os.makedirs('one_results')
        plt.savefig("one_results/agent2_action.png")
        plt.figure(2)

        plt.scatter(range(len(agent1_action)), agent1_action)
        plt.xlabel('action')
        plt.title('agent1 action')
        plt.legend()
        plt.savefig("one_results/agent1_action.png")
        plt.show()
        from collections import Counter
        agent1_action_count = Counter(agent1_action)
        agent2_action_count = Counter(agent2_action)
        print(agent1_action_count)
        print(agent2_action_count)
         
def test2():
     
    fitness_function_id = 1
    fitness_num = 1
    pop_size = 50
    dim = 10
    step = 20
    max_iter = 2000                         
    arange = (-100, 100)
    individual_id = 20 # 追踪的个体
    action_one = []
    action_zero = []
    for i in range(fitness_function_id, fitness_function_id + fitness_num):
        print(f"测试函数{i}")
        fitness_function = all_functions[i]
        population = np.random.uniform(arange[0], arange[1], (pop_size, dim))
         
        gBest_dqn, fitness_gBest_dqn, fitness_history_dqn, count_dqn, agent1_action, agent2_action \
        = test_dqn_strategy(np.copy(population), pop_size, dim, max_iter, fitness_function, arange, individual_id)
        # action1.append(agent1_action)


        for j in range(0, len(agent1_action), step):
            data = agent1_action[j: j + step]
            action_one.append(data.count(1))
            action_zero.append(data.count(0)) 
        # csv保存原始数据，和action_zero, action_one,
        # 保存处理后的统计数据
        with open('one_results/action_statistics.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time Window', 'Exploitation Count', 'Exploration Count', 'Total Count'])
            for idx, (zero, one) in enumerate(zip(action_zero, action_one)):
                writer.writerow([f'{idx*step}-{(idx+1)*step}', one, zero, zero + one])
 

        # 保存完整的数据统计信息
        with open('one_results/data_summary.txt', 'w') as f:
            f.write(f"Analysis Parameters:\n")
            f.write(f"Step Size: {step}\n")
            f.write(f"Max Iterations: {max_iter}\n")
            f.write(f"Population Size: {pop_size}\n")
            f.write(f"Dimension: {dim}\n\n")
            
            f.write(f"Data Summary:\n")
            f.write(f"Total Exploration Actions: {sum(action_zero)}\n")
            f.write(f"Total Exploitation Actions: {sum(action_one)}\n")
            f.write(f"Total Actions: {sum(action_zero) + sum(action_one)}\n")
            f.write(f"Exploration Ratio: {sum(action_zero)/(sum(action_zero) + sum(action_one)):.4f}\n")
            f.write(f"Exploitation Ratio: {sum(action_one)/(sum(action_zero) + sum(action_one)):.4f}\n")
def test3():
    """
    统计所有粒子每次迭代时的勘探和开发情况
    """
    fitness_function_id = 9
    fitness_function = all_functions[fitness_function_id]
    fitness_num = 1
    pop_size = 50
    dim = 30
    max_iter = 1000
    arange = (-100, 100) 
    agent1_action = {'exploration': [], 'exploitation': []}
    agent2_action = []
    population = np.random.uniform(arange[0], arange[1], (pop_size, dim))

    count = {f'o{i+1}': 0 for i in range(len(operators))}
    # 将 operators 转换为字典形式
    operators_dict = {f'o{i+1}': operators[i] for i in range(len(operators))}
    operator_keys = list(operators_dict.keys())

    agent1 = DQNAgent(state_size=6, action_size=2)
    agent2 = DQNAgent(state_size=8, action_size=len(operator_keys))
    
    if not os.path.exists('models/agent1.pth') or not os.path.exists('models/agent2.pth'):
        raise FileNotFoundError("请先训练模型，确保 'models/agent1.pth' 和 'models/agent2.pth' 文件存在。")
    
    agent1.qnetwork_local.load_state_dict(torch.load('models/agent1.pth'))
    agent2.qnetwork_local.load_state_dict(torch.load('models/agent2.pth'))
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
    
    step = 20  # 添加步长参数
    exploration_counts = []
    exploitation_counts = []

    # 在开始迭代前创建CSV文件并写入表头
    with open('one_results/勘探开发.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Generation', 'Exploration Count', 'Exploitation Count', 'Total Count'])

    for iter in range(max_iter):
        # 计算状态和动作
        state_features = compute_state(
            pre_population, population, pBest, gBest, 
            fitness_function, max_iter, iter, count_k
        )
        action1 = agent1.act_test(state_features, 0)
        action_count = list(action1)
        # exploration计入这次迭代，所有个体action1等于1的次数，exploitation计入action1等于0的次数

        # 修改记录方式
        exploration_count = action_count.count(1)
        exploitation_count = action_count.count(0)
        agent1_action['exploration'].append(exploration_count)
        agent1_action['exploitation'].append(exploitation_count)

        # 直接写入当前代的数据
        with open('one_results/勘探开发.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([iter, exploration_count, exploitation_count, exploration_count + exploitation_count])

        operator_indices = np.array([operator_keys.index(op) for op in current_operator])
        state2_features = np.column_stack((action1, operator_indices, state_features))
        action2 = agent2.act_test(state2_features, 0)
        
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
    print(f"最优解:{gBest}, 最优解适应度:{fitness_gBest}")
        
    # 保持原有的绘图功能 
    plt.plot(agent1_action['exploration'], label='exploration')
    plt.legend() 
    plt.title('Exploration Counts')
    if not os.path.exists('勘探证明'):
        os.makedirs('勘探证明')
    plt.savefig("勘探证明/勘探开发粒子数.png")
    plt.close()
    return agent1_action



if __name__ == "__main__":

    test2()
    # test()
    test3()
     
