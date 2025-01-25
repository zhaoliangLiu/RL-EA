import numpy as np

def rand_to_best_de(func, dim, pop_size, max_iter, bounds=(-100, 100), F=0.5, CR=0.7):
    """
    基础的DE算法实现，使用DE/rand-to-best/1策略
    
    参数:
        func: 目标函数
        dim: 问题维度
        pop_size: 种群大小
        max_iter: 最大迭代次数
        bounds: 搜索范围的上下界，默认为(-100, 100)
        F: 缩放因子，默认0.5
        CR: 交叉率，默认0.7
    
    返回:
        best_solution: 找到的最优解
        best_fitness: 最优解的适应度值
        history: 每代最优适应度值的历史记录
    """
    # 初始化种群
    population = np.random.uniform(
        bounds[0], 
        bounds[1], 
        (pop_size, dim)
    )
    
    # 计算初始种群的适应度
    fitness = np.array([func(ind.reshape(1, -1))[0] for ind in population])
    
    # 记录历史最优
    history = []
    best_fitness = np.min(fitness)
    best_solution = population[np.argmin(fitness)].copy()
    
    # 主循环
    for iteration in range(max_iter):
        # 记录当前代的最优适应度
        history.append(best_fitness)
        
        # 对每个个体进行变异和交叉
        for i in range(pop_size):
            # 获取当前种群最优个体
            current_best_idx = np.argmin(fitness)
            current_best = population[current_best_idx]
            
            # 随机选择三个不同的个体
            available_indices = [x for x in range(pop_size) if x != i and x != current_best_idx]
            r1, r2, r3 = np.random.choice(available_indices, 3, replace=False)
            
            # DE/rand-to-best/1 变异
            mutant = (
                population[r1] + 
                F * (current_best - population[r1]) +
                F * (population[r2] - population[r3])
            )
            
            # 边界处理
            mutant = np.clip(mutant, bounds[0], bounds[1])
            
            # 二项式交叉
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            # 选择
            trial_fitness = func(trial.reshape(1, -1))[0]
            if trial_fitness <= fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                # 更新最优解
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial.copy()
        
        # 可选：打印当前进度
        # if (iteration + 1) % 100 == 0:
            # print(f"Iteration {iteration + 1}/{max_iter}, Best fitness: {best_fitness}")
    
    return best_solution, best_fitness, history

# 测试代码
if __name__ == "__main__":
    from cec2017.functions import f1
    
    # 测试参数
    dim = 10
    pop_size = 50
    max_iter = 1000
    bounds = (-100, 100)
    
    # 运行算法
    best_sol, best_fit, hist = rand_to_best_de(
        func=f1,
        dim=dim,
        pop_size=pop_size,
        max_iter=max_iter,
        bounds=bounds
    )
    
    print("\nOptimization finished!")
    print(f"Best solution: {best_sol}")
    print(f"Best fitness: {best_fit}")
    
    # 绘制收敛曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(hist)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness (log scale)')
    plt.title('Convergence Curve')
    plt.grid(True)
    plt.show()