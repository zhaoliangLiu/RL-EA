import numpy as np

def o1(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    LLH1: 基于遗传算法(GA)的算子，适用于连续变量优化
    基于公式: x = 0.5[(p1 + p2) + β(p1 - p2)]
    """
    dimension = population[individual_id].shape[0]
    eta = 20  # 可调参数
    
    # 随机选择两个父代
    available_indices = [i for i in range(len(population)) if i != individual_id]
    p1_idx, p2_idx = np.random.choice(available_indices, 2, replace=False)
    p1, p2 = population[p1_idx], population[p2_idx]
    
    # 计算β
    mu = np.random.random()
    if mu <= 0.5:
        beta = (2 * mu) ** (1.0 / (eta + 1))
    else:
        beta = (2 * (1 - mu)) ** (-1.0 / (eta + 1))
    
    # 生成新个体
    new_individual = 0.5 * ((p1 + p2) + beta * (p1 - p2))
    
    return new_individual, velocities

def o2(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    LLH2: 基于PSO的算子，具有强大的局部搜索能力和快速收敛速度
    基于公式: x' = x + wv + r1(pk - x) + r2(gk - x)
    """
    w = 0.7  # 惯性权重
    r1 = np.random.random()
    r2 = np.random.random()
    
    # 更新速度和位置
    new_velocities = velocities.copy()
    new_velocities = w * velocities + r1 * (pBest[individual_id] - population[individual_id]) + \
                    r2 * (gBest - population[individual_id])
    new_individual = population[individual_id] + new_velocities
    
    return new_individual, new_velocities

def o3(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    LLH3: 基于差分进化(DE)的算子
    使用DE/rand/1策略: x' = x + F(x1 - x2)
    """
    F = 0.5  # 缩放因子
    
    # 随机选择两个不同的个体
    available_indices = [i for i in range(len(population)) if i != individual_id]
    idx1, idx2 = np.random.choice(available_indices, 2, replace=False)
    
    # 生成新个体
    new_individual = population[individual_id] + F * (population[idx1] - population[idx2])
    
    return new_individual, velocities

def o4(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    LLH4: 基于自适应差分进化的算子
    使用DE/best/1策略: x' = xbest + F(x1 - x2)
    """
    F = 0.7  # 缩放因子
    
    # 随机选择两个不同的个体
    available_indices = [i for i in range(len(population)) if i != individual_id]
    idx1, idx2 = np.random.choice(available_indices, 2, replace=False)
    
    # 使用全局最优个体
    new_individual = gBest + F * (population[idx1] - population[idx2])
    
    return new_individual, velocities

def o5(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    LLH5: 基于局部搜索的算子
    使用高斯扰动: x' = x + N(0, σ)
    """
    sigma = 0.1  # 标准差
    
    # 添加高斯噪声
    noise = np.random.normal(0, sigma, size=population[individual_id].shape)
    new_individual = population[individual_id] + noise
    
    return new_individual, velocities

def o6(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    LLH6: 基于PSO和DE混合的算子
    结合PSO的速度更新和DE的变异操作
    """
    w = 0.5  # 惯性权重
    F = 0.5  # DE缩放因子
    
    # PSO部分
    r1 = np.random.random()
    new_velocities = velocities.copy()
    new_velocities = w * velocities + r1 * (pBest[individual_id] - population[individual_id])
    
    # DE部分
    available_indices = [i for i in range(len(population)) if i != individual_id]
    idx1, idx2 = np.random.choice(available_indices, 2, replace=False)
    mutation = F * (population[idx1] - population[idx2])
    
    # 组合更新
    new_individual = population[individual_id] + new_velocities + mutation
    
    return new_individual, new_velocities

def o7(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    LLH7: 基于自适应权重的算子
    权重随迭代次数动态调整
    """
    w_max = 0.9
    w_min = 0.4
    w = w_max - (w_max - w_min) * t / max_iter
    
    r1 = np.random.random()
    r2 = np.random.random()
    
    new_velocities = velocities.copy()
    new_velocities = w * velocities + r1 * (pBest[individual_id] - population[individual_id]) + \
                    r2 * (gBest - population[individual_id])
    new_individual = population[individual_id] + new_velocities
    
    return new_individual, new_velocities

def o8(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    LLH8: 基于精英策略的算子
    使用最优个体引导搜索
    """
    alpha = 0.3  # 精英影响因子
    
    # 计算种群中最优个体
    fitness_values = fitness_function(population)
    best_idx = np.argmin(fitness_values)
    
    # 向最优个体移动
    new_individual = population[individual_id] + \
                    alpha * (population[best_idx] - population[individual_id])
    
    return new_individual, velocities

def o9(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    LLH9: 基于随机游走的算子
    在当前解附近随机探索
    """
    step_size = 0.1
    
    # 生成随机方向
    direction = np.random.randn(*population[individual_id].shape)
    direction = direction / np.linalg.norm(direction)
    
    # 随机游走
    new_individual = population[individual_id] + step_size * direction
    
    return new_individual, velocities

def o10(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    LLH10: 基于交叉操作的算子
    结合多个父代信息
    """
    CR = 0.8  # 交叉率
    
    # 随机选择两个父代
    available_indices = [i for i in range(len(population)) if i != individual_id]
    p1_idx, p2_idx = np.random.choice(available_indices, 2, replace=False)
    
    # 生成交叉掩码
    mask = np.random.random(population[individual_id].shape) < CR
    
    # 交叉操作
    new_individual = np.where(mask, 
                            population[p1_idx], 
                            population[p2_idx])
    
    return new_individual, velocities

  
operators = [o1, o2, o3, o4, o5, o6, o7, o8, o9, o10] 

# operators = [o7, o12]


if __name__ == "__main__":
    dimension = 2
    population_size = 5
    population = np.random.rand(population_size, dimension)
    pBest = np.random.rand(population_size, dimension)
    gBest = np.min(pBest, axis=0)
    t = 0
    max_iter = 100
    velocities = np.zeros((population_size, dimension))

    def fitness_function(x):
        return np.sum(x ** 2, axis=1)

    individual_id = 0
    print('原始个体：')
    print(population[individual_id])
    print('应用算子 o7 后的个体：')
    new_individual, new_velocities = o1(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities)
    print(new_individual)