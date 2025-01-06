def shade(population, pop_size, dim, max_iter, obj_func, bounds, memory_size=6, p=0.11):
    """
    Success-History based Adaptive DE (SHADE)
    
    Args:
        population: 初始种群 shape=(pop_size, dim)
        pop_size: 种群大小
        dim: 维度
        max_iter: 最大迭代次数
        obj_func: 目标函数
        bounds: 边界约束 [lower, upper]
        memory_size: 历史记忆大小(H in paper)，默认6
        p: pbest的比例，默认0.11
    """
    import numpy as np
    
    # 初始化种群
    X = population.copy()
    fitness = obj_func(X)
    
    # 初始化历史记忆
    M_CR = np.ones(memory_size) * 0.5  # 历史交叉率记忆
    M_F = np.ones(memory_size) * 0.5   # 历史缩放因子记忆
    k = 0  # 历史记忆指针
    
    # 初始化最优解
    best_idx = np.argmin(fitness)
    gbest = X[best_idx].copy()
    gbest_fitness = fitness[best_idx]
    gbest_history = [gbest_fitness]
    
    # 初始化外部档案
    archive = []
    
    # 主循环
    for t in range(max_iter):
        # 存储成功的参数
        S_CR = []
        S_F = []
        S_diff = []  # 存储适应度改进值
        
        # 生成当代的CR和F
        CR = np.random.normal(M_CR[k], 0.1, size=pop_size)
        CR = np.clip(CR, 0, 1)
        F = np.random.standard_cauchy(size=pop_size) * 0.1 + M_F[k]
        F = F[F > 0]
        while len(F) < pop_size:  # 确保生成足够的F值
            f = np.random.standard_cauchy() * 0.1 + M_F[k]
            if f > 0:
                F = np.append(F, f)
        F = np.clip(F, 0, 1)
        
        # 临时数组存储新解
        X_new = X.copy()
        fitness_new = fitness.copy()
        
        # 对每个个体进行进化
        for i in range(pop_size):
            # 选择pbest
            p_num = max(1, int(pop_size * p))
            pbest_idx = np.argsort(fitness)[:p_num]
            pbest = X[np.random.choice(pbest_idx)]
            
            # 选择r1≠i
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            
            # 选择r2≠r1≠i
            if len(archive) > 0:
                # 从种群和档案中选择r2
                all_candidates = np.vstack((X, np.array(archive)))
                r2_cand = np.arange(len(all_candidates))
                r2_cand = r2_cand[r2_cand != i]
                if r1 < pop_size:  # 如果r1来自当前种群
                    r2_cand = r2_cand[r2_cand != r1]
                r2 = np.random.choice(r2_cand)
                xr2 = all_candidates[r2]
            else:
                # 从当前种群中选择r2
                idxs.remove(r1)
                r2 = np.random.choice(idxs)
                xr2 = X[r2]
            
            # 变异
            v = X[i] + F[i] * (pbest - X[i]) + F[i] * (X[r1] - xr2)
            
            # 边界处理
            v = np.clip(v, bounds[0], bounds[1])
            
            # 交叉
            j_rand = np.random.randint(dim)
            mask = np.random.random(dim) <= CR[i]
            mask[j_rand] = True
            u = np.where(mask, v, X[i])
            
            # 选择
            u_fitness = obj_func(u.reshape(1, -1))[0]
            
            if u_fitness <= fitness[i]:
                if u_fitness < fitness[i]:  # 严格改进时记录参数
                    diff = abs(fitness[i] - u_fitness)
                    archive.append(X[i].copy())
                    S_CR.append(CR[i])
                    S_F.append(F[i])
                    S_diff.append(diff)
                X_new[i] = u
                fitness_new[i] = u_fitness
                
                if u_fitness < gbest_fitness:
                    gbest = u.copy()
                    gbest_fitness = u_fitness
        
        # 更新种群
        X = X_new
        fitness = fitness_new
        
        # 更新历史记忆
        if len(S_CR) > 0:
            # Lehmer平均计算F
            mean_F = np.sum(np.array(S_F)**2) / np.sum(np.array(S_F))
            mean_CR = np.mean(S_CR) if sum(S_CR) > 0 else M_CR[k]
            
            M_F[k] = mean_F
            M_CR[k] = mean_CR
            k = (k + 1) % memory_size
        
        # 控制档案大小
        if len(archive) > pop_size:
            archive = list(np.array(archive)[
                np.random.choice(len(archive), pop_size, replace=False)])
        
        gbest_history.append(gbest_fitness)
    
    return gbest, gbest_fitness, gbest_history
# 使用示例
if __name__ == "__main__":
    import numpy as np 
    import os
    import sys
    import matplotlib.pyplot as plt
    from jade_de import jade_de 
    
    # 获取当前文件所在目录的父目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
    from fitness_function.cec2017.functions import all_functions
    
    # 问题设置
    dim = 10
    pop_size = 50
    arange = np.array([-100, 100])
    max_iter = 1000
    func = all_functions[22]

    # 初始化种群
    population = np.random.uniform(arange[0], arange[1], (pop_size, dim))

    # 运行对比实验
    best_jade, fitness_jade, history_jade = jade_de(
        np.copy(population),
        pop_size,
        dim,
        max_iter,
        func,
        arange
    )
    
    best_shade, fitness_shade, history_shade = shade(
        np.copy(population),
        pop_size,
        dim,
        max_iter,
        func,
        arange
    )
    
    print("JADE最优解:", fitness_jade)
    print("SHADE最优解:", fitness_shade)
    
    plt.plot(history_jade, label="JADE")
    plt.plot(history_shade, label="SHADE")
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("JADE vs SHADE on F1")
    plt.yscale('log')
    plt.grid(True)
    plt.show()