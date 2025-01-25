import numpy as np

def de_rand_2(func, dim, pop_size, max_iter, arange, F=0.5, CR=0.7):
    """
    DE/rand/2 变体的向量化实现
    """
    # 初始化种群
    population = np.random.uniform(arange[0], arange[1], (pop_size, dim))
    fitness = func(population)
    
    # 记录最优解
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    history = []
    # 主循环
    for i in range(max_iter):
        # 为所有个体一次性生成随机索引
        r_matrix = np.array([
            [i] + list(np.random.choice(
                [x for x in range(pop_size) if x != i], 
                5, 
                replace=False
            )) for i in range(pop_size)
        ])
        
        # 生成所有变异向量
        mutants = (population[r_matrix[:, 1]] + 
                  F * (population[r_matrix[:, 2]] - population[r_matrix[:, 3]]) +
                  F * (population[r_matrix[:, 4]] - population[r_matrix[:, 5]]))
        
        # 生成交叉掩码
        cross_points = np.random.rand(pop_size, dim) < CR
        # 确保每个试验向量至少有一个维度来自变异向量
        rows_without_cross = ~np.any(cross_points, axis=1)
        random_dims = np.random.randint(0, dim, size=np.sum(rows_without_cross))
        cross_points[rows_without_cross, random_dims] = True
        
        # 生成试验向量
        trials = np.where(cross_points, mutants, population)
        trials = np.clip(trials, arange[0], arange[1])
        
        # 评估所有试验向量
        trial_fitness = func(trials)
        
        # 选择
        improvements = trial_fitness <= fitness
        population[improvements] = trials[improvements]
        fitness[improvements] = trial_fitness[improvements]
        
        # 更新最优解
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_solution = population[best_idx].copy()
            best_fitness = fitness[best_idx]
        history.append(best_fitness)
        if i % 50 == 0:
            print("iter:", i, "best_fitness:", best_fitness)
    return best_solution, best_fitness, history

def de_rand_to_best_1(func, dim, pop_size, max_iter, arange, F=0.5, CR=0.7):
    """
    DE/rand-to-best/1 变体的向量化实现
    """
    # 初始化种群
    population = np.random.uniform(arange[0], arange[1], (pop_size, dim))
    fitness = func(population)
    
    # 记录最优解
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    history = []
    # 主循环
    for i in range(max_iter):
        # 获取当前种群最优个体
        current_best_idx = np.argmin(fitness)
        current_best = population[current_best_idx]
        
        # 为所有个体一次性生成随机索引
        r_matrix = np.array([
            [i] + list(np.random.choice(
                [x for x in range(pop_size) if x != i and x != current_best_idx], 
                3, 
                replace=False
            )) for i in range(pop_size)
        ])
        
        # 生成所有变异向量
        mutants = (population[r_matrix[:, 1]] + 
                  F * (current_best - population[r_matrix[:, 1]]) +
                  F * (population[r_matrix[:, 2]] - population[r_matrix[:, 3]]))
        
        # 生成交叉掩码
        cross_points = np.random.rand(pop_size, dim) < CR
        # 确保每个试验向量至少有一个维度来自变异向量
        rows_without_cross = ~np.any(cross_points, axis=1)
        random_dims = np.random.randint(0, dim, size=np.sum(rows_without_cross))
        cross_points[rows_without_cross, random_dims] = True
        
        # 生成试验向量
        trials = np.where(cross_points, mutants, population)
        trials = np.clip(trials, arange[0], arange[1])
        
        # 评估所有试验向量
        trial_fitness = func(trials)
        
        # 选择
        improvements = trial_fitness <= fitness
        population[improvements] = trials[improvements]
        fitness[improvements] = trial_fitness[improvements]
        
        # 更新最优解
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fitness:
            best_solution = population[best_idx].copy()
            best_fitness = fitness[best_idx]
        history.append(best_fitness)
        # if i % 50 == 0:
            # print("iter:", i, "best_fitness:", best_fitness)
    return best_solution, best_fitness, history

# 测试代码
if __name__ == "__main__":
    from cec2017.functions import all_functions
    import matplotlib.pyplot as plt
    # 测试参数
    dim = 30
    pop_size = 30
    max_iter = 10000
    bounds = (-100, 100)
    func = all_functions[0]
    # 测试两个变体
    best_sol_rand2, best_fit_rand2, history_rand2 = de_rand_2(
        func=func,
        dim=dim,
        pop_size=pop_size,
        max_iter=max_iter,
        arange=bounds
    )
    
    best_sol_randtobest, best_fit_randtobest, history_randtobest = de_rand_to_best_1(
        func=func,
        dim=dim,
        pop_size=pop_size,
        max_iter=max_iter,
        arange=bounds
    )
    
    # 运行SHADE
    from SHADE import SHADE
    shade = SHADE(func, dim, pop_size, memory_size=100, max_evaluations=max_iter*pop_size, arange=bounds)
    best_sol_shade, best_fit_shade, history_shade = shade.run()
    
    print("\nDE/rand/2 Results:")
    print(f"Best fitness: {best_fit_rand2}")
    
    print("\nDE/rand-to-best/1 Results:")
    print(f"Best fitness: {best_fit_randtobest}") 
    
    print("\nSHADE Results:")
    print(f"Best fitness: {best_fit_shade}")

    plt.plot(np.log10(history_rand2), label='DE/rand/2')
    plt.plot(np.log10(history_randtobest), label='DE/rand-to-best/1')
    plt.plot(np.log10(history_shade), label='SHADE')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title('Optimization Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
