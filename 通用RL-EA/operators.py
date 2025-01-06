import numpy as np

def o1(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    PSO 算子 o1: w = 0.7, c1 = 2.0, c2 = 2.0
    """
    w = 0.7
    c1 = 2.0
    c2 = 2.0
    r1 = np.random.rand(*population[individual_id].shape)
    r2 = np.random.rand(*population[individual_id].shape)
    velocities = w * velocities + c1 * r1 * (pBest[individual_id] - population[individual_id]) + c2 * r2 * (gBest - population[individual_id])
    new_individual = population[individual_id] + velocities
    return new_individual, velocities

def o2(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    PSO 算子 o2: w = 1, c1 = 2.5, c2 = 2
    """
    w = 1
    c1 = 2.5
    c2 = 2
    r1 = np.random.rand(*population[individual_id].shape)
    r2 = np.random.rand(*population[individual_id].shape)
    new_velocities = velocities.copy()
    new_velocities = w * velocities + c1 * r1 * (pBest[individual_id] - population[individual_id]) + c2 * r2 * (gBest - population[individual_id])
    new_individual = population[individual_id] + new_velocities
    return new_individual, new_velocities

def o3(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    PSO 算子 o3: w = 0.5, c1 = 2, c2 = 2.5
    """
    w = 0.5
    c1 = 2.0
    c2 = 2.5
    r1 = np.random.rand(*population[individual_id].shape)
    r2 = np.random.rand(*population[individual_id].shape)
    new_velocities = velocities.copy()
    new_velocities = w * velocities + c1 * r1 * (pBest[individual_id] - population[individual_id]) + c2 * r2 * (gBest - population[individual_id])
    new_individual = population[individual_id] + new_velocities
    return new_individual, new_velocities

def o4(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    DE/rand/1: F = 0.5, CR = 0.5
    """
    F = 0.5
    CR = 0.5
    
    available_indices = [i for i in range(len(pBest)) if i != individual_id]
    idx1, idx2 = np.random.choice(available_indices, 2, replace=False)
    
    mutant = population[individual_id] + F * (pBest[idx1] - pBest[idx2])
    cross_points = np.random.rand(*population[individual_id].shape) < CR
    new_individual = np.where(cross_points, mutant, population[individual_id])
    return new_individual, velocities

def o5(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    DE/best/1: F = 0.5, CR = 0.5
    """
    F = 0.5
    CR = 0.5
    
    available_indices = [i for i in range(len(pBest)) if i != individual_id]
    idx1, idx2 = np.random.choice(available_indices, 2, replace=False)
    
    mutant = gBest + F * (pBest[idx1] - pBest[idx2])
    cross_points = np.random.rand(*population[individual_id].shape) < CR
    new_individual = np.where(cross_points, mutant, population[individual_id])
    return new_individual, velocities

def o6(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    DE/current-to-best/1: F = 0.5, CR = 0.5
    """
    F = 0.5
    CR = 0.5
    
    available_indices = [i for i in range(len(pBest)) if i != individual_id]
    idx1, idx2 = np.random.choice(available_indices, 2, replace=False)
    
    mutant = population[individual_id] + F * (gBest - population[individual_id]) + F * (pBest[idx1] - pBest[idx2])
    cross_points = np.random.rand(*population[individual_id].shape) < CR
    new_individual = np.where(cross_points, mutant, population[individual_id])
    return new_individual, velocities

def o7(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    新算子 o7，基于搜索中心的自适应加权粒子群算法
    """
    dimension = population[individual_id].shape[0]
    
    k = max(int(len(pBest) * 0.05), int(len(pBest) * 0.2 - np.floor(t / 1000)))
    fitness_values = fitness_function(pBest.reshape(-1, dimension))
    sorted_indices = np.argsort(fitness_values)
    elite_indices = sorted_indices[:k]
    Et = pBest[elite_indices]
    fitness_Et = fitness_values[elite_indices]

    fmax_t = np.max(fitness_Et)
    fmin_t = np.min(fitness_Et)
    denominator = fmax_t - fmin_t + 1e-8
    mi_t = np.exp((fmax_t - fitness_Et) / denominator)
    Wi_t = mi_t / np.sum(mi_t)

    theta_t = np.sum(Wi_t[:, np.newaxis] * Et, axis=0)

    w = 0.9 - (0.9 - 0.4) * (t / max_iter)
    c = 3.2

    r1 = np.random.rand(dimension)
    r2 = np.random.rand(dimension)
    new_velocities = velocities.copy()
    new_velocities = w * r1 * velocities + c * r2 * (theta_t - population[individual_id])
    new_individual = population[individual_id] + new_velocities
    return new_individual, new_velocities

def o8(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    FIPS 算子 o8：全信息粒子群优化算法
    """
    dimension = population[individual_id].shape[0]
    phi = 1
    
    w = 0.9 - (0.9 - 0.4) * (t / max_iter)

    # 计算适应度值并归一化
    fitness_values = fitness_function(pBest.reshape(-1, dimension))
    fmax = np.max(fitness_values)
    fmin = np.min(fitness_values)
    fitness_normalized = (fmax - fitness_values) / (fmax - fmin + 1e-8)
    phi_weights = phi * (fitness_normalized / (np.sum(fitness_normalized) + 1e-8))
    
    # 计算加权影响
    weighted_influence = np.zeros_like(population[individual_id])  # 确保维度匹配
    for j in range(len(pBest)):
        r_j = np.random.rand(*population[individual_id].shape)  # 修改随机数生成维度
        weighted_influence += phi_weights[j] * r_j * (pBest[j] - population[individual_id])

    # 更新速度和位置
    new_velocities = velocities.copy()
    new_velocities = w * velocities + weighted_influence
    new_individual = population[individual_id] + new_velocities
    return new_individual, new_velocities

def o9(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    简化的 CLPSO 算子 o9：综合学习粒子群优化算法的简化版本
    """
    dimension = population[individual_id].shape[0]
    
    fi = np.random.randint(0, len(pBest), dimension)
    
    w = 0.9 - (0.9 - 0.4) * t / max_iter
    Pc = 0.2
    c = 1.49445
    
    rand_matrix = np.random.rand(dimension)
    r = np.random.rand(dimension)
    
    exemplar = np.array([pBest[fi[d]][d] for d in range(dimension)])
    compare = r < Pc
    pbest_update = np.where(compare, exemplar, pBest[individual_id])
    
    new_velocities = velocities.copy()
    new_velocities = w * velocities + c * rand_matrix * (pbest_update - population[individual_id])
    new_individual = population[individual_id] + new_velocities
    return new_individual, new_velocities

def o10(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    简化的jSO算子
    """
    dimension = population[individual_id].shape[0]
    GMAX = max_iter

    if t <= 0.2 * GMAX:
        p = 0.11
    elif t <= 0.4 * GMAX:
        p = 0.13
    elif t <= 0.6 * GMAX:
        p = 0.15
    elif t <= 0.9 * GMAX:
        p = 0.2
    else:
        p = 0.1

    F = 0.7
    CR = 0.9 if t < 0.25 * GMAX else 0.6
    
    p_num = max(int(p * len(pBest)), 2)
    fitness_values = fitness_function(pBest.reshape(-1, dimension))
    best_indices = np.argsort(fitness_values)[:p_num]
    p_best = pBest[np.random.choice(best_indices)]
    
    available_indices = [i for i in range(len(pBest)) if i != individual_id]
    r1, r2 = np.random.choice(available_indices, 2, replace=False)
    
    mutant = population[individual_id] + F * (p_best - population[individual_id]) + F * (pBest[r1] - pBest[r2])
    
    jrand = np.random.randint(dimension)
    cross_points = np.random.rand(dimension) < CR
    cross_points[jrand] = True
    new_individual = np.where(cross_points, mutant, population[individual_id])
    return new_individual, velocities

def o11(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    DE/rand/2: F = 0.5, CR = 0.5
    """
    F = 0.5
    CR = 0.5
    
    available_indices = [i for i in range(len(pBest)) if i != individual_id]
    r1, r2, r3, r4, r5 = np.random.choice(available_indices, 5, replace=False)
    
    mutant = pBest[r1] + F * (pBest[r2] - pBest[r3]) + F * (pBest[r4] - pBest[r5])
    cross_points = np.random.rand(*population[individual_id].shape) < CR
    new_individual = np.where(cross_points, mutant, population[individual_id])
    return new_individual, velocities

def o12(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    DE/current-to-rand/1: F = 0.5, K = 0.5
    """
    F = 0.5
    K = 0.5
    
    available_indices = [i for i in range(len(pBest)) if i != individual_id]
    r1, r2, r3 = np.random.choice(available_indices, 3, replace=False)
    
    new_individual = population[individual_id] + K * (pBest[r1] - population[individual_id]) + F * (pBest[r2] - pBest[r3])
    return new_individual, velocities
import numpy as np
from scipy import stats

def  o13(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    JADE-DE 算子: 自适应差分进化算法
    """
    dimension = population[individual_id].shape[0]
    
    # JADE参数
    c = 0.1      # 自适应参数
    p = 0.05     # 优秀解比例
    mu_cr = 0.5  # CR均值
    mu_f = 0.5   # F均值
    
    # 生成CR和F值
    CR = np.clip(np.random.normal(mu_cr, 0.1, 1)[0], 0, 1)
    F = np.clip(stats.cauchy.rvs(loc=mu_f, scale=0.1, size=1)[0], 0.1, 1)
    
    # 选择最优解集
    p_best_size = max(int(len(pBest) * p), 2)
    fitness_values = fitness_function(pBest.reshape(-1, dimension))
    p_best_indices = np.argsort(fitness_values)[:p_best_size]
    p_best_idx = np.random.choice(p_best_indices)
    
    # 随机选择两个不同的解
    available_indices = [i for i in range(len(pBest)) if i != individual_id]
    r1, r2 = np.random.choice(available_indices, 2, replace=False)
    
    # 变异
    v = population[individual_id] + F * (pBest[p_best_idx] - population[individual_id]) + F * (pBest[r1] - pBest[r2])
    
    # 交叉
    j_rand = np.random.randint(dimension)
    mask = np.random.rand(dimension) < CR
    mask[j_rand] = True
    new_individual = np.where(mask, v, population[individual_id])
    
    return new_individual, velocities

def o14(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    SHADE算子，基于成功历史的差分进化
    """
    dimension = population[individual_id].shape[0]
    pop_size = len(population)
    
    # 简化的参数生成
    CR = np.random.normal(0.5, 0.1)
    CR = np.clip(CR, 0, 1)
    F = np.clip(np.random.standard_cauchy() * 0.1 + 0.5, 0, 1)
    
    # 选择pbest（使用前5%的优秀个体）
    p = 0.05
    p_num = max(1, int(pop_size * p))
    fitness_values = fitness_function(population.reshape(-1, dimension))
    sorted_indices = np.argsort(fitness_values)[:p_num]
    pbest = population[np.random.choice(sorted_indices)]
    
    # 选择r1, r2
    available_idx = list(range(pop_size))
    available_idx.remove(individual_id)
    r1, r2 = np.random.choice(available_idx, 2, replace=False)
    
    # 变异
    v = population[individual_id] + F * (pbest - population[individual_id]) + \
        F * (population[r1] - population[r2])
    
    # 交叉
    j_rand = np.random.randint(dimension)
    mask = np.random.random(dimension) <= CR
    mask[j_rand] = True
    new_individual = np.where(mask, v, population[individual_id])
    
    return new_individual, velocities

def o15(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities):
    """
    RDE算子，随机或最优个体的差分进化
    """
    dimension = population[individual_id].shape[0]
    pop_size = len(population)
    
    # 固定参数
    F = 0.5
    CR = 0.9
    
    # 基于迭代轮次自适应选择策略
    p_best = t / max_iter
    
    if np.random.rand() < p_best:
        # best/1 策略
        fitness_values = fitness_function(population.reshape(-1, dimension))
        best_idx = np.argmin(fitness_values)
        base = population[best_idx]
    else:
        # rand/1 策略
        available_idx = list(range(pop_size))
        available_idx.remove(individual_id)
        r_base = np.random.choice(available_idx)
        base = population[r_base]
        available_idx.remove(r_base)
    
    # 选择两个不同的个体进行差分
    available_idx = list(range(pop_size))
    available_idx.remove(individual_id)
    r1, r2 = np.random.choice(available_idx, 2, replace=False)
    
    # 变异
    v = base + F * (population[r1] - population[r2])
    
    # 交叉
    j_rand = np.random.randint(dimension)
    mask = np.random.random(dimension) <= CR
    mask[j_rand] = True
    new_individual = np.where(mask, v, population[individual_id])
    
    return new_individual, velocities
  
operators = [o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15]
# operators = [o7, o8, o9, o10, o11, o12, o13, o14, o15]

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
    new_individual, new_velocities = o15(individual_id, population, pBest, gBest, fitness_function, t, max_iter, velocities)
    print(new_individual)