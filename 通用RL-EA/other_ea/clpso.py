def clpso(population, num_particles, dim, max_iter, obj_func, arange):
    import numpy as np
    
    w = 0.729    # 惯性权重
    c = 1.49445  # 学习因子
    
    # 初始化
    X = population
    V = np.random.uniform(arange[0]*0.2, arange[1]*0.2, (num_particles, dim))
    pbest = X.copy()
    pbest_values = obj_func(X)
    
    # 初始化学习概率
    Pc = 0.05 + 0.45 * (np.exp(10*(np.arange(num_particles)/num_particles-1)) 
                        / (np.exp(10)-1))
    
    # 学习示例矩阵
    fi = np.zeros((num_particles, dim))
    
    # 找到全局最佳
    gbest_index = np.argmin(pbest_values)
    gbest = pbest[gbest_index].copy()
    gbest_value = pbest_values[gbest_index]
    gbest_history = []
    
    for t in range(max_iter):
        # 更新学习示例
        for i in range(num_particles):
            for d in range(dim):
                if np.random.rand() < Pc[i]:
                    # 选择两个不同的粒子
                    candidates = np.random.choice(num_particles, 2, replace=False)
                    better_one = candidates[np.argmin([pbest_values[candidates[0]], 
                                                     pbest_values[candidates[1]]])]
                    fi[i,d] = better_one
                else:
                    fi[i,d] = i
        
        # 更新速度和位置
        for i in range(num_particles):
            # 修复这里：为每个维度分别获取示例
            for d in range(dim):
                exemplar_idx = int(fi[i,d])
                V[i,d] = w * V[i,d] + c * np.random.rand() * (pbest[exemplar_idx,d] - X[i,d])
            
            # 限制速度和位置
            V[i] = np.clip(V[i], arange[0]*0.2, arange[1]*0.2)
            X[i] = X[i] + V[i]
            X[i] = np.clip(X[i], arange[0], arange[1])
        
        # 评估和更新
        values = obj_func(X)
        better_mask = values < pbest_values
        pbest[better_mask] = X[better_mask]
        pbest_values[better_mask] = values[better_mask]
        
        current_best = np.min(pbest_values)
        if current_best < gbest_value:
            gbest_index = np.argmin(pbest_values)
            gbest = pbest[gbest_index].copy()
            gbest_value = current_best
            
        gbest_history.append(gbest_value)
        
    return gbest, gbest_value, gbest_history



def test1():
    
    import numpy as np
    from jade_de import jade_de
    from clpso import clpso
    import matplotlib.pyplot as plt
    # 参数设置
    dim = 10          # 维度
    pop_size = 30     # 种群大小
    max_iter = 1000    # 最大迭代次数
    arange = [-100, 100]  # 搜索范围
    
    # 测试函数 (Sphere函数)
    def sphere(x):
        return np.sum(x**2, axis=1)
    
    # 初始化种群
    population = np.random.uniform(arange[0], arange[1], (pop_size, dim))
    
    # 测试JADE-DE
    best_jade, fitness_jade, history_jade = jade_de(
        np.copy(population), 
        pop_size, 
        dim, 
        max_iter, 
        sphere, 
        arange
    )
    
    # 测试CLPSO
    best_clpso, fitness_clpso, history_clpso = clpso(
        np.copy(population), 
        pop_size, 
        dim, 
        max_iter, 
        sphere, 
        arange
    )
    
    # 打印结果
    print("\nJADE-DE最优解:", fitness_jade)
    print("CLPSO最优解:", fitness_clpso)
    
    # 绘制收敛曲线
    plt.plot(history_jade, label='JADE-DE')
    plt.plot(history_clpso, label='CLPSO')
    plt.xlabel('iteration')
    plt.ylabel('fitness')
    plt.title('JADE-DE vs CLPSO on Sphere Function')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test1()
