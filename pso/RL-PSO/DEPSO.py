def depso(num_particles, dim, max_iter, obj_func, arange):
    """
    基于搜索中心的自适应加权粒子群算法 (SDMS-PSO format)
    """
    import numpy as np
    
    # 初始化
    X = np.random.uniform(arange[0], arange[1], (num_particles, dim))
    V_max = 0.2 * (arange[1] - arange[0])
    V = np.random.uniform(-V_max, V_max, (num_particles, dim))
    
    # 评估初始适应度
    fitness = obj_func(X)
    pbest = X.copy()
    pbest_fitness = fitness.copy()
    gbest = pbest[np.argmin(pbest_fitness)]

                  
    gbest_fitness = np.min(pbest_fitness)
    
    # 历史记录
    gbest_history = [gbest_fitness] 
    
    # 主循环
    for t in range(max_iter):
        # 计算精英集
        k = max(10, 20 - int(t / 1000))  # 精英粒子数从20递减到10
        sorted_indices = np.argsort(pbest_fitness)
        elite_indices = sorted_indices[:k]
        Et = pbest[elite_indices]
        fitness_Et = pbest_fitness[elite_indices]
        
        # 计算权重
        fmax_t = np.max(fitness_Et)
        fmin_t = np.min(fitness_Et)
        denominator = fmax_t - fmin_t + 1e-8
        mi_t = np.exp((fmax_t - fitness_Et) / denominator)
        Wi_t = mi_t / np.sum(mi_t)
        
        # 计算搜索中心
        theta_t = np.sum(Wi_t[:, np.newaxis] * Et, axis=0)
        
        # 更新速度和位置
        w = 0.9 - (0.9 - 0.4) * (t / max_iter)
        c = 3.2
        
        # Generate random numbers for all particles at once
        r1 = np.random.rand(num_particles, dim)
        r2 = np.random.rand(num_particles, dim)

        # Update velocities for all particles
        V = w * r1 * V + c * r2 * (theta_t - X)
        V = np.clip(V, -V_max, V_max)

        # Update positions for all particles
        X = X + V
        X = np.clip(X, arange[0], arange[1])

        # Update fitness for all particles
        fitness = obj_func(X)

        # Update pbest and gbest
        improved = fitness < pbest_fitness
        pbest[improved] = X[improved]
        pbest_fitness[improved] = fitness[improved]

        # Update global best
        if np.min(fitness) < gbest_fitness:
            gbest_idx = np.argmin(fitness)
            gbest = X[gbest_idx].copy()
            gbest_fitness = fitness[gbest_idx]
        gbest_history.append(gbest_fitness )
    
    return gbest, gbest_fitness, gbest_history

if __name__ == "__main__": 

    from cec2017.functions import all_functions 
    import numpy as np

    # 重复100次， 取平均适应度
    best_fit_list = []
    for i in range(1):
        func = all_functions[21]
        num_particles = 100  # 粒子数设置为50
        dim = 100  # 维度设置为50
        max_iter = 10000  # 最大迭代次数设置为10,000
        arange = [-100, 100]

        best_sol, best_fit, best_history = depso(num_particles, dim, max_iter, func, arange)
        import matplotlib.pyplot as plt
        plt.plot(np.log10(best_history))
        plt.title("DEPSO Convergence")
        plt.xlabel("Iterations")
        plt.ylabel("Fitness")
        plt.legend()
        plt.savefig("results_pso/RL-PSO/DEPSO.png")
        plt.show()
        best_fit_list.append(best_fit)

    print("max:", np.max(best_fit_list))
    print("mean:", np.mean(best_fit_list))
    print("std:", np.std(best_fit_list))
    print("min:", np.min(best_fit_list))