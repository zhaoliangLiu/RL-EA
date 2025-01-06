import numpy as np
def de(population, pop_size, dim, max_iter, obj_func, bounds, F=0.5, Cr=0.9):
    """
    标准差分进化算法 (Classical Differential Evolution - DE/rand/1/bin)
    
    参数:
        population: 初始种群 shape=(pop_size, dim)
        pop_size: 种群大小
        dim: 问题维度
        max_iter: 最大迭代次数
        obj_func: 目标函数
        bounds: 边界约束 shape=(2,) [lower, upper]
        F: 缩放因子，默认0.5
        Cr: 交叉概率，默认0.9
    
    返回:
        gbest: 最优解
        gbest_fitness: 最优解的适应度值
        gbest_history: 优化历程
    """
    import numpy as np
    
    # 初始化
    X = population.copy()
    fitness = obj_func(X)
    
    # 初始化最优解
    gbest_idx = np.argmin(fitness)
    gbest = X[gbest_idx].copy()
    gbest_fitness = fitness[gbest_idx]
    gbest_history = [gbest_fitness]
    
    # 主循环
    for t in range(max_iter):
        # 对每个个体进行DE操作
        for i in range(pop_size):
            # 随机选择三个不同的个体，且都不等于当前个体i
            idxs = list(range(pop_size))
            idxs.remove(i)
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            
            # DE/rand/1 变异操作
            v = X[r1] + F * (X[r2] - X[r3])
            
            # 边界处理
            v = np.clip(v, bounds[0], bounds[1])
            
            # 二项式交叉
            j_rand = np.random.randint(0, dim)
            u = np.zeros(dim)
            
            for j in range(dim):
                if j == j_rand or np.random.rand() < Cr:
                    u[j] = v[j]
                else:
                    u[j] = X[i, j]
            
            # 选择
            u_fitness = obj_func(u.reshape(1, -1))[0]
            if u_fitness < fitness[i]:
                X[i] = u
                fitness[i] = u_fitness
                
                # 更新全局最优
                if u_fitness < gbest_fitness:
                    gbest = u.copy()
                    gbest_fitness = u_fitness
        
        gbest_history.append(gbest_fitness)
    
    return gbest, gbest_fitness, gbest_history

# 使用示例
if __name__ == "__main__":
    def test_function(X):
        """测试函数（球函数）"""
        return np.sum(X**2, axis=1)

    # 问题设置
    dim = 30
    pop_size = 100
    bounds = np.array([-100, 100])
    max_iter = 1000

    # 初始化种群
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))

    # 运行优化
    best_solution, best_fitness, history = de(
        population=population,
        pop_size=pop_size,
        dim=dim,
        max_iter=max_iter,
        obj_func=test_function,
        bounds=bounds,
        F=0.5,    # 标准DE推荐值
        Cr=0.9    # 标准DE推荐值
    )

    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness}")