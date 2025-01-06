def jade_de(population, num_particles, dim, max_iter, obj_func, arange):
    import numpy as np
    from scipy import stats
    
    # JADE参数
    c = 0.1      # 自适应参数
    p = 0.05     # 优秀解比例
    mu_cr = 0.5  # CR均值初始化
    mu_f = 0.5   # F均值初始化
    
    # 初始化
    X = population
    fitness = obj_func(X)
    
    # 记录最优解
    best_idx = np.argmin(fitness)
    best_solution = X[best_idx].copy()
    best_fitness = fitness[best_idx]
    best_history = []
    
    for t in range(max_iter):
        # 生成CR和F值
        CR = np.random.normal(mu_cr, 0.1, num_particles)
        CR = np.clip(CR, 0, 1)
        F = stats.cauchy.rvs(loc=mu_f, scale=0.1, size=num_particles)
        F[F <= 0] = 0.1
        F[F > 1] = 1
        
        # 存储成功的CR和F值
        S_CR = []
        S_F = []
        
        # 生成变异向量
        for i in range(num_particles):
            # 选择最优解集
            p_best_size = int(num_particles * p)
            p_best_indices = np.argsort(fitness)[:p_best_size]
            p_best_idx = np.random.choice(p_best_indices)
            
            # 随机选择两个不同的解
            idxs = [idx for idx in range(num_particles) if idx != i]
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            
            # 变异
            v = X[i] + F[i] * (X[p_best_idx] - X[i]) + F[i] * (X[r1] - X[r2])
            v = np.clip(v, arange[0], arange[1])
            
            # 交叉
            j_rand = np.random.randint(dim)
            mask = np.random.rand(dim) < CR[i]
            mask[j_rand] = True
            u = np.where(mask, v, X[i])
            
            # 选择
            u_fitness = obj_func(u.reshape(1, -1))[0]
            if u_fitness <= fitness[i]:
                X[i] = u
                fitness[i] = u_fitness
                S_CR.append(CR[i])
                S_F.append(F[i])
        
        # 更新mu_CR和mu_F
        if len(S_CR) > 0:
            mu_cr = (1 - c) * mu_cr + c * np.mean(S_CR)
            mu_f = (1 - c) * mu_f + c * (np.sum(np.array(S_F)**2) / np.sum(S_F))
        
        # 更新最优解
        current_best = np.min(fitness)
        if current_best < best_fitness:
            best_idx = np.argmin(fitness)
            best_solution = X[best_idx].copy()
            best_fitness = current_best
            
        best_history.append(best_fitness)
    
    return best_solution, best_fitness, best_history

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    # 测试函数
    def sphere(x):
        return np.sum(x**2, axis=1)

    # 参数设置
    dim = 30
    pop_size = 50
    max_iter = 1000
    arange = [-100, 100]

    # 初始化种群
    population = np.random.uniform(arange[0], arange[1], (pop_size, dim))

    # 运行JADE-DE
    best_solution, best_fitness, history = jade_de(population, pop_size, dim, max_iter, sphere, arange)
    plt.plot(history)
    plt.legend(["JADE-DE"])
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("JADE-DE on Sphere Function")
    plt.show()
    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)
