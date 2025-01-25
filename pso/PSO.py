# 文件路径：pythonProject/Swarm/PSO/standard_pso.py

import numpy as np
import random
from cec2017.functions import all_functions, best_fitness
from operators import operators
import matplotlib.pyplot as plt

# 导入算子函数 o1
from operators import o1

# 参数设置
pop_size = 50              # 种群大小
dimension = 10           # 维度
max_iter = 1000            # 最大迭代次数
w = 0.5                    # 惯性权重
c1 = 2.5                  # 个体学习因子
c2 = 1.5                   # 社会学习因子

# # 随机种子以确保可重复性
# random_seed = 42
# np.random.seed(random_seed)
# random.seed(random_seed)

# 选择适应度函数
func_id = 1  # 选择函数 f1，您可以更改为其他函数编号
fitness_function = all_functions[func_id]

# 指定最优适应度值和容许误差范围
optimal_fitness = best_fitness[func_id]  # 假设函数 f1 的最优适应度为 0.0
error_tolerance = 1e-5

# 初始化种群
population = np.random.uniform(-100, 100, (pop_size, dimension))
velocity = np.random.uniform(-1, 1, (pop_size, dimension))

# 初始化个体最优和全局最优
pBest = np.copy(population)
fitness_pBest = fitness_function(pBest)
gBest = pBest[np.argmin(fitness_pBest)]
fitness_gBest = fitness_function([gBest])[0]

# 记录每次迭代的最佳适应度值
fitness_history = [fitness_gBest]

# PSO 主循环
iteration = 0
while abs(fitness_gBest - optimal_fitness) > error_tolerance:
    # 计算速度和位置更新（使用标准 PSO 公式）
    r1 = np.random.rand(pop_size, dimension)
    r2 = np.random.rand(pop_size, dimension)
    velocity = w * velocity + c1 * r1 * (pBest - population) + c2 * r2 * (gBest - population)
    population = population + velocity

    # 应用边界条件
    population = np.clip(population, -100, 100)

    # 评估适应度
    fitness_population = fitness_function(population)

    # 更新个体最优
    better_mask = fitness_population < fitness_pBest
    pBest[better_mask] = population[better_mask]
    fitness_pBest[better_mask] = fitness_population[better_mask]

    # 更新全局最优
    min_fitness = np.min(fitness_pBest)
    if min_fitness < fitness_gBest:
        fitness_gBest = min_fitness
        gBest = pBest[np.argmin(fitness_pBest)]

    # 记录最佳适应度
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
plt.title('PSO 优化过程')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()