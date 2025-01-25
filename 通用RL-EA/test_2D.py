# 测试维度为2时，粒子的具体位置变化

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
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

 
def animate_points(history, best_solution):
    fig, ax = plt.subplots()
    
    def init():
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        return []

    def update(frame):
        ax.clear()
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        points = history[frame]
        ax.scatter(points[:,0], points[:,1], c='blue', alpha=0.6)
        # 添加最优点标记
        ax.scatter(best_solution[0], best_solution[1], c='red', marker='*', s=100, label='Global Optimum')
        ax.set_title(f'Frame {frame}')
        ax.legend()
        return []

    # 将interval增加到1000（1秒）或更大的值
    anim = FuncAnimation(fig, update, frames=len(history),
                        init_func=init, blit=False,
                        interval=1000)  # 这里改为1000ms = 1秒
    
    plt.show()
 

def get_history():
    """
    记录所有粒子的移动轨迹和勘探开发情况
    """
    history = []
    fitness_function_id = 4
    fitness_function = all_functions[fitness_function_id]
    fitness_num = 1
    pop_size = 50
    dim = 10
    max_iter = 1000
    arange = (-100, 100) 
    indevidual_id = 10
    fitness_history = []
    best_solution = None
    population = np.random.uniform(arange[0], arange[1], (pop_size, dim))
     

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
    history_normalized = []

    # 添加勘探开发统计
    agent1_action = {'exploration': [], 'exploitation': []}
    
    # 创建CSV文件记录勘探开发数据
    with open('one_results/勘探开发.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Generation', 'Exploration Count', 'Exploitation Count', 'Total Count'])
     

    for iter in range(max_iter):
          
        history.append(population.copy())
        # 同时记录种群再当前迭代时最大最小归一化的位置
        normalized_population = (population - np.min(population)) / (np.max(population) - np.min(population) + 1e-6)
        history_normalized.append(normalized_population.copy())
        # 计算状态和动作
        state_features = compute_state(
            pre_population, population, pBest, gBest, 
            fitness_function, max_iter, iter, count_k
        )
        action1 = agent1.act_test(state_features, 0)
         
        # 添加勘探开发统计
        action_count = list(action1)
        exploration_count = action_count.count(1)
        exploitation_count = action_count.count(0)
        agent1_action['exploration'].append(exploration_count)
        agent1_action['exploitation'].append(exploitation_count)

        # 记录当前代的勘探开发数据
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
        fitness_history.append(fitness_gBest)
    return np.array(history), np.array(history_normalized), fitness_history, gBest, agent1_action

def plot_exploration_counts(agent1_action):
    # 绘制勘探开发图
    plt.plot(agent1_action, label='exploration')
    plt.legend() 
    plt.title('Exploration Counts')
    if not os.path.exists('勘探证明'):
        os.makedirs('勘探证明')
    plt.savefig("勘探证明/勘探开发粒子数.png")
    plt.close()



def test_is_exploration(history): 
    # 记录所有粒子每次迭代时，跳跃的距离之和的变化
    jump_distance = []
    for i in range(len(history)-1):
        jump_distance.append(np.sum(np.linalg.norm(history[i+1] - history[i], axis=1)))
    # 取对数，变化更明显
    plt.plot(jump_distance)
    plt.title("Change of the sum of jump distance")
    plt.xlabel("Iteration")
    plt.ylabel("Sum of jump distances")
    if not os.path.exists('勘探证明'):
        os.makedirs('勘探证明')
    plt.savefig("勘探证明/跳跃距离.png") 
    plt.close()
    jump_distance_log = np.log(jump_distance)
    plt.plot(jump_distance_log)
    plt.title("Change of the sum of jump distances (log)")
    plt.xlabel("Iteration")
    plt.ylabel("Sum of jump distances")
    if not os.path.exists('勘探证明'):
        os.makedirs('勘探证明')
    plt.savefig("勘探证明/跳跃距离(log).png")
    plt.close()
    return jump_distance, jump_distance_log
    
def compare_curves(curve1, curve2, name):
    from scipy.stats import wasserstein_distance
    from scipy.stats import spearmanr
    
    # 确保两条曲线长度相同
    min_length = min(len(curve1), len(curve2))
    curve1 = np.array(curve1[:min_length])
    curve2 = np.array(curve2[:min_length])
    
    # 1. 归一化
    norm_curve1 = (curve1 - np.min(curve1)) / (np.max(curve1) - np.min(curve1))
    norm_curve2 = (curve2 - np.min(curve2)) / (np.max(curve2) - np.min(curve2))
    
    # 2. 多种相似度指标
    # DTW距离（考虑曲线形状）
    # dtw_distance = dtw.distance(norm_curve1, norm_curve2)
    
    # Wasserstein距离（考虑分布差异）
    w_distance = wasserstein_distance(norm_curve1, norm_curve2)
    
    # Spearman等级相关系数（考虑非线性单调关系）
    spearman_corr, _ = spearmanr(norm_curve1, norm_curve2)
    
    # 计算MSE和相关系数
    mse = np.mean((norm_curve1 - norm_curve2) ** 2)
    correlation = np.corrcoef(norm_curve1, norm_curve2)[0, 1]
    
    # 3. 可视化比较 
    plt.plot(norm_curve1, label='Curve 1', alpha=0.7)
    plt.plot(norm_curve2, label='Curve 2', alpha=0.7)
    plt.title('Curve Similarity Analysis')
    # 不要标题，变为txt
    plt.text(20, 0.2, f'Curve Similarity Analysis\n'
             f'Wasserstein Distance: {w_distance:.4f}\n'
             f'Spearman Correlation: {spearman_corr:.4f}\n'
             f'MSE: {mse:.4f}\n'
             f'Correlation: {correlation:.4f}')

    plt.xlabel('Iteration')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)

    
    if not os.path.exists('勘探证明'):
        os.makedirs('勘探证明')
    plt.savefig(f"勘探证明/相似度分析_{name}.png")
    plt.close()

    # 同时计算mse和相关系数
     

    
    return {
        'wasserstein': w_distance,
        'spearman': spearman_corr,
        'mse': mse,
        'correlation': correlation
    }

 
if __name__ == "__main__":  
    history, history_normalized, fitness_history, best_solution, agent1_action = get_history()
    plot_exploration_counts(agent1_action['exploration'])
    
    jump_distance, jump_distance_log    = test_is_exploration(history)
    print(f"best_solution:{best_solution}, \nbest_fitness:{fitness_history[-1]}")
    # print(f"跳跃距离之和的变化: {jump_distance}")

    # 测试归一化位置和跳跃距离的相似度
    jump_distance_normalized, jump_distance_log_normalized = test_is_exploration(history_normalized)
    # print(f"归一化位置的变化: {history_normalized}")

    
    explore_test3 = agent1_action['exploration']
    compare_curves(jump_distance_normalized, explore_test3, '归一化位置')
    compare_curves(jump_distance_log_normalized, explore_test3, '归一化跳跃距离(log)')
    compare_curves(jump_distance, explore_test3, '跳跃距离')
    compare_curves(jump_distance_log, explore_test3, '跳跃距离(log)')
    plt.show()

