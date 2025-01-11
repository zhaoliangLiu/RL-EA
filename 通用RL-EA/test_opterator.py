from operators import operators
# from LLH10 import operators
from fitness_function.cec2017.functions import all_functions
# from simple_function.simple_functions import all_functions
import numpy as np
import matplotlib.pyplot as plt
from other_ea.standard_pso import standard_pso


# 配置参数
operator_id = 8  # 选择要测试的算子
function_id = 9  # 选择测试函数
fitness_function = all_functions[function_id]
true_best_fitness = 0
operator = operators[operator_id]

# 种群参数设置
pop_size = 30
dim = 10
max_iter = 1000
arange = (-100, 100)

# 初始化种群
population = np.random.rand(pop_size, dim) * (arange[1] - arange[0]) + arange[0]
velocities = np.zeros_like(population)  # 初始化速度矩阵

# 运行标准PSO作为对比
st_gbest, st_gbest_value, st_gbest_history = standard_pso(population.copy(), pop_size, dim, max_iter, fitness_function, arange)

# 初始化个体最优和全局最优
pBest = population.copy()
fitness = fitness_function(population)
pBest_fitness = fitness.copy()
gBest_fitness = np.min(fitness)
gBest = population[np.argmin(fitness)]

# 记录历史最优值
gBest_fitness_history = []

# 主循环
for t in range(max_iter):
    # 调用算子更新种群，现在正确处理velocities
    for i in range(pop_size):
        population[i], velocities[i] = operator(i, population, pBest, gBest, fitness_function, t, max_iter, velocities[i])
    
    # 边界处理
    population = np.clip(population, arange[0], arange[1])
    
    # 计算新种群的适应度
    fitness = fitness_function(population)
    
    # 更新个体最优
    better_fitness_mask = fitness < pBest_fitness
    pBest = np.where(better_fitness_mask[:, np.newaxis], population, pBest)
    pBest_fitness = np.where(better_fitness_mask, fitness, pBest_fitness)
    
    # 更新全局最优
    if np.min(fitness) < gBest_fitness:
        gBest_fitness = np.min(fitness)
        gBest = population[np.argmin(fitness)]
    
    # 记录历史
    gBest_fitness_history.append(gBest_fitness)
    
    # 打印进度
    if t % 100 == 0:
        print(f"迭代 {t}: 全局最优适应度 = {gBest_fitness}, 真实最优值 = {true_best_fitness}")

# 输出最终结果
# y轴对数变换
gBest_fitness_history = np.log10(gBest_fitness_history)
st_gbest_history = np.log10(st_gbest_history)
print(f"最终最优适应度 = {gBest_fitness}\n 真实最优值 = {true_best_fitness}")
print(f"误差 = {gBest_fitness - true_best_fitness}")
print(f"标准PSO最优适应度 = {st_gbest_value}")
print(f"标准PSO误差 = {st_gbest_value - true_best_fitness}")

# 绘制收敛曲线
plt.figure(figsize=(10, 6))
plt.plot(gBest_fitness_history, label=f"Operator{operator_id} gBest_fitness")
plt.plot(st_gbest_history, label="Standard PSO gBest_fitness")
plt.title(f"Function{function_id}; Population={pop_size}; Dimension={dim}; Max_iterations={max_iter}")
plt.xlabel("Iterations")
plt.ylabel("Fitness Value")
plt.legend()
plt.grid(True)
plt.show()

# 关闭图像
plt.close()
