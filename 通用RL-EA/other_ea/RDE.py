import numpy as np
def rde(population, pop_size, dim, max_iter, obj_func, bounds, p=0.11, archive_rate=2.0, memory_size=6):
    """
    重构差分进化算法 (Reconstructed Differential Evolution)
    
    参数:
        population: 初始种群 shape=(pop_size, dim)
        pop_size: 种群大小
        dim: 问题维度
        max_iter: 最大迭代次数
        obj_func: 目标函数
        bounds: 边界约束 shape=(2,) [lower, upper]
        p: DE/current-to-pbest/1中的p值，默认0.11
        archive_rate: 外部档案大小比率，默认2.0
        memory_size: 参数记忆库大小，默认6
    """
    import numpy as np
    
    # 初始化
    X = population.copy()
    fitness = obj_func(X)
    
    # 初始化外部档案
    archive = []
    archive_size = int(archive_rate * pop_size)
    
    # 初始化最优解
    gbest_idx = np.argmin(fitness)
    gbest = X[gbest_idx].copy()
    gbest_fitness = fitness[gbest_idx]
    gbest_history = [gbest_fitness]
    
    # 参数自适应相关
    memory_pos = 0
    memory_F = np.ones(memory_size) * 0.5
    memory_Cr = np.ones(memory_size) * 0.5
    
    # 主循环
    for t in range(max_iter):
        # 生成控制参数
        r = np.random.randint(0, memory_size)
        F = np.random.standard_cauchy(pop_size) * 0.1 + memory_F[r]
        F = np.clip(F, 0, 1)
        Cr = np.random.normal(memory_Cr[r], 0.1, pop_size)
        Cr = np.clip(Cr, 0, 1)
        
        # DE/current-to-order-pbest/1变异
        mutants = np.zeros_like(X)
        num_pbest = max(1, int(p * pop_size))
        pbest_idx = np.argsort(fitness)[:num_pbest]
        
        for i in range(pop_size):
            # 选择pbest个体
            pbest_i = np.random.choice(pbest_idx)
            
            # 随机选择两个不同的个体
            available_idx = list(range(pop_size))
            available_idx.remove(i)
            
            # 如果有外部档案，将其加入选择池
            if archive:
                archive_array = np.array(archive)
                all_solutions = np.vstack([X[available_idx], archive_array])
                r1, r2 = np.random.choice(len(all_solutions), 2, replace=False)
                sol_r1, sol_r2 = all_solutions[r1], all_solutions[r2]
            else:
                r1, r2 = np.random.choice(available_idx, 2, replace=False)
                sol_r1, sol_r2 = X[r1], X[r2]
            
            # 对三个解进行排序
            solutions = [X[pbest_i], sol_r1, sol_r2]
            fits = [fitness[pbest_i]]
            fits.extend([obj_func(sol.reshape(1, -1))[0] for sol in [sol_r1, sol_r2]])
            sorted_idx = np.argsort(fits)
            
            pbest = solutions[sorted_idx[0]]
            median = solutions[sorted_idx[1]]
            worst = solutions[sorted_idx[2]]
            
            # 执行变异
            mutants[i] = (X[i] + F[i] * (pbest - X[i]) + 
                         F[i] * (median - worst))
        
        # 边界处理
        mutants = np.clip(mutants, bounds[0], bounds[1])
        
        # 交叉
        trials = np.zeros_like(X)
        for i in range(pop_size):
            j_rand = np.random.randint(0, dim)
            mask = np.random.rand(dim) < Cr[i]
            mask[j_rand] = True
            trials[i] = np.where(mask, mutants[i], X[i])
        
        # 选择
        trial_fitness = obj_func(trials)
        improved = trial_fitness < fitness
        
        # 更新外部档案
        if np.any(improved):
            archive.extend(X[improved].tolist())  # 转换为列表后添加
            if len(archive) > archive_size:
                idx = np.random.choice(len(archive), size=archive_size, replace=False)
                archive = [archive[i] for i in idx]
        
        # 更新种群
        X[improved] = trials[improved]
        fitness[improved] = trial_fitness[improved]
        
        # 更新全局最优
        current_best = np.argmin(fitness)
        if fitness[current_best] < gbest_fitness:
            gbest = X[current_best].copy()
            gbest_fitness = fitness[current_best]
            
        gbest_history.append(gbest_fitness)
        
        # 更新参数记忆库
        if np.any(improved):
            memory_F[memory_pos] = np.mean(F[improved])
            memory_Cr[memory_pos] = np.mean(Cr[improved])
            memory_pos = (memory_pos + 1) % memory_size
    
    return gbest, gbest_fitness, gbest_history

# 使用示例
if __name__ == "__main__":
    import os
    import sys
    import matplotlib.pyplot as plt
    from jade_de import jade_de 
     # 获取当前文件所在目录的父目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # 将父目录添加到 Python 模块搜索路径中
    sys.path.append(parent_dir)

    # 导入 fitness_function 模块
    from fitness_function.cec2017.functions import all_functions    
    # 问题设置
    dim = 100
    pop_size = 100
    arange = np.array([-100, 100])
    max_iter = 3000
    func = all_functions[2]

    # 初始化种群
    population = np.random.uniform(arange[0], arange[1], (pop_size, dim))

    # 运行优化
    #测试depso
    best_depso, fitness_depso, history_depso = jade_de(
        np.copy(population),
        pop_size,
        dim,
        max_iter,
        func,
        arange
    )
    # 测试rde
    best_rde, fitness_rde, history_rde = rde(
        np.copy(population),
        pop_size,
        dim,
        max_iter,
        func,
        arange
    )
    print("DEPSO最优解:", fitness_depso)
    print("RDE最优解:", fitness_rde)
    plt.plot(history_depso, label="DEPSO")
    plt.plot(history_rde, label="RDE")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("DEPSO vs RDE on F1")
    plt.show()



     