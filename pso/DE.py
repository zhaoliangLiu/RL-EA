# 文件路径：pythonProject/Swarm/DE/de.py

import numpy as np
import random
from fitness_function.cec2017.functions import all_functions, best_fitness
from operators import operators
import matplotlib.pyplot as plt

# 导入算子函数 o4
from operators import o4

# 参数设置
pop_size = 50              # 种群大小
dimension = 10             # 维度
max_iter = 1000            # 最大迭代次数
F = 0.5                    # 缩放因子
CR = 0.5                   # 交叉概率

# 随机种子以确保可重复性
# random_seed = 42
# np.random.seed(random_seed)
# random.seed(random_seed)

# 选择适应度函数
func_id = 5  # 选择函数 f5，您可以更改为其他函数编号
fitness_function = all_functions[func_id]

# 指定最优适应度值和容许误差范围
optimal_fitness = best_fitness[func_id]
error_tolerance = 1e-5

# 初始化种群
population = np.random.uniform(-100, 100, (pop_size, dimension))

# 初始化个体最优和全局最优
pBest = np.copy(population)
fitness_pBest = fitness_function(pBest)
gBest = pBest[np.argmin(fitness_pBest)]
fitness_gBest = fitness_function([gBest])[0]

# 记录每次迭代的最佳适应度值
fitness_history = [fitness_gBest]

# DE 主循环
iteration = 0
while abs(fitness_gBest - optimal_fitness) > error_tolerance:
    new_population = np.copy(population)
    for i in range(pop_size):
        # 选择三个不同的个体
        idxs = list(range(pop_size))
        idxs.remove(i)
        a_idx, b_idx, c_idx = np.random.choice(idxs, 3, replace=False)
        a, b, c = population[a_idx], population[b_idx], population[c_idx]
        # DE/rand/1 变异操作
        mutant = a + F * (b - c)
        # 修正超出边界的粒子
        mutant = np.clip(mutant, -100, 100)
        # 交叉操作
        cross_points = np.random.rand(dimension) < CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, dimension)] = True
        trial = np.where(cross_points, mutant, population[i])
        # 修正超出边界的粒子
        trial = np.clip(trial, -100, 100)
        # 选择操作
        fitness_trial = fitness_function([trial])[0]
        fitness_i = fitness_function([population[i]])[0]
        if fitness_trial < fitness_i:
            new_population[i] = trial
            fitness_pBest[i] = fitness_trial
            pBest[i] = trial
            if fitness_trial < fitness_gBest:
                gBest = trial
                fitness_gBest = fitness_trial
    population = new_population
    fitness_history.append(fitness_gBest)

    # 打印进度
    if iteration % 50 == 0 or abs(fitness_gBest - optimal_fitness) <= error_tolerance:
        print(f"迭代 {iteration}，最佳适应度：{fitness_gBest}")
    iteration += 1

print(f"\n优化在 {iteration} 次迭代后完成。")
print(f"达到的最佳适应度值：{fitness_gBest}")

# 绘制适应度变化曲线
plt.figure(figsize=(10, 6))
plt.plot(fitness_history, label='最佳适应度')
plt.xlabel('迭代次数')
plt.ylabel('适应度')
plt.title('DE 优化过程')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()