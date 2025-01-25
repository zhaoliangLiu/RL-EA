import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

# 差分进化算法函数
def differential_evolution(population, fitness_function, bounds, F=0.5, CR=0.9, max_generations=100, 
                           use_crossover=True, accept_all=False, mutation_strategy='current-to-pBest-w/1', 
                           global_optimum=None, archive_size=50, e=0.5, p=0.1):
    pop_size, dim = population.shape
    fitness = fitness_function(population)
    best_idx = np.argmin(fitness)
    best_individual = population[best_idx]
    best_fitness = fitness[best_idx]

    # 外部档案
    archive = []

    # 记录历史
    history = []
    for gen in range(max_generations):
        new_population = np.zeros_like(population)
        for i in range(pop_size):
            # 选择变异个体
            if mutation_strategy == 'current-to-pBest-w/1':
                # DE/current-to-pBest-w/1 策略
                p_best_vectors = population[np.random.choice(pop_size, int(pop_size * p), replace=False)]  # 随机选择 p% 的个体作为 p-best
                F_w = adaptive_F_w(F, gen, max_generations)
                idxs = [idx for idx in range(pop_size) if idx != i]
                r1, r2 = np.random.choice(idxs, 2, replace=False)
                mutant = population[i] + F_w * (p_best_vectors[np.random.randint(0, len(p_best_vectors))] - population[i]) + F * (population[r1] - population[r2])
            elif mutation_strategy == 'current-to-Amean-w/I':
                # DE/current-to-Amean-w/I 策略
                F_w = adaptive_F_w(F, gen, max_generations)
                X_Amean = calculate_X_Amean(population, archive, e)
                idxs = [idx for idx in range(pop_size) if idx != i]
                r1, r2 = np.random.choice(idxs, 2, replace=False)
                mutant = population[i] + F_w * (X_Amean - population[i]) + F * (population[r1] - population[r2])
            else:
                raise ValueError("Invalid mutation strategy. Choose 'current-to-pBest-w/1' or 'current-to-Amean-w/I'.")

            # 边界处理
            mutant = np.clip(mutant, [b[0] for b in bounds], [b[1] for b in bounds])

            # 交叉
            if use_crossover:
                cross_points = np.random.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, population[i])
            else:
                trial = mutant

            # 适应度选择
            trial_fitness = fitness_function(np.array([trial]))
            if accept_all or trial_fitness < fitness[i]:
                new_population[i] = trial
                fitness[i] = trial_fitness
                # 将 trial 添加到 archive 中
                if len(archive) < archive_size:
                    archive.append(trial)
                else:
                    archive[np.random.randint(0, archive_size)] = trial  # 随机替换
            else:
                new_population[i] = population[i]

        # 更新种群
        population = new_population

        # 更新最佳个体
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        # 记录当前代的信息
        history.append({
            'population': population.copy(),  # 记录整个种群
            'best_individual': best_individual.copy(),
            'best_fitness': best_fitness
        })

    return history

# 自适应权重计算
def adaptive_F_w(F, gen, max_generations):
    """
    计算自适应权重 F_w
    """
    # 根据论文中的公式，F_w 随进化代数增加而减小
    F_w = (0.7 + (gen / max_generations) * (0.5 - 0.7)) * F
    return F_w

# 计算加权平均 X_Amean
def calculate_X_Amean(population, archive, e):
    """
    计算加权平均 X_Amean
    """
    if len(archive) == 0:
        # 如果 archive 为空，直接使用 population 计算 X_Amean
        combined_population = population
    else:
        # 将 archive 转换为 numpy 数组
        archive_array = np.array(archive)
        # 检查 archive 的维度是否与 population 一致
        if archive_array.shape[1] != population.shape[1]:
            raise ValueError("Archive and population dimensions do not match.")
        # 垂直堆叠 population 和 archive
        combined_population = np.vstack((population, archive_array))

    # 计算加权平均
    m = round(e * len(combined_population))
    if m == 0:
        m = 1  # 确保至少选择一个个体
    weights = np.array([(np.log(m + 0.5) - np.log(i + 1)) for i in range(m)])
    weights /= np.sum(weights)
    X_Amean = np.sum(weights[:, np.newaxis] * combined_population[:m], axis=0)
    return X_Amean

# 动态展示进化过程（子图）
def plot_dynamic_evolution_subplots(histories, bounds, titles, global_optimum=None):
    """
    动态展示进化过程，使用子图布局
    """
    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Dynamic Evolution of Two Mutation Strategies', fontsize=16)

    # 初始化子图
    scats = []
    best_points = []
    for i, ax in enumerate(axs):
        ax.set_title(titles[i])
        ax.set_xlim(bounds[0][0], bounds[0][1])
        ax.set_ylim(bounds[1][0], bounds[1][1])
        scat = ax.scatter([], [], c='blue', label='Population')
        best_point, = ax.plot([], [], 'ro', label='Best Individual')
        if global_optimum is not None:
            ax.plot(global_optimum[0], global_optimum[1], 'g*', markersize=10, label='Global Optimum')
        ax.legend()
        scats.append(scat)
        best_points.append(best_point)

    # 动态更新
    max_gen = max(len(history) for history in histories)
    for gen in range(max_gen):
        for i, history in enumerate(histories):
            if gen < len(history):
                record = history[gen]
                population = record['population']
                best_individual = record['best_individual']
                scats[i].set_offsets(population)
                best_points[i].set_data([best_individual[0]], [best_individual[1]])
                axs[i].set_title(f"{titles[i]} - Generation: {gen+1}, Best Fitness: {record['best_fitness']:.4f}")
        plt.pause(0.1)

    plt.ioff()
    plt.show()

def plot_all_particles_density(histories, bounds, global_optimum=None, grid_size=100, sigma=1.0):
    """
    用颜色深浅表示所有粒子在某个区域的访问频率，并加入高斯滤波平滑处理
    修改颜色映射：访问次数多的区域颜色更蓝，访问次数少的区域颜色更黄
    """
    strategies = ['current-to-pBest-w/1', 'current-to-Amean-w/I']
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('All Particles Density under Different Mutation Strategies (Smoothed)', fontsize=16)

    # 自定义颜色映射：从黄色到蓝色
    custom_cmap = plt.get_cmap('viridis')
    # custom_cmap = plt.get_cmap('plasma')
    min_color = custom_cmap(0)  # 自定义颜色映射的最小颜色
    for i, strategy in enumerate(strategies):
        # 提取所有粒子的轨迹
        all_trajectories = []
        for record in histories[i]:
            all_trajectories.extend(record['population'])  # 将所有粒子的位置合并
        all_trajectories = np.array(all_trajectories)

        # 初始化密度矩阵
        density = np.zeros((grid_size, grid_size))

        # 计算每个网格的访问次数
        x_bins = np.linspace(bounds[0][0], bounds[0][1], grid_size + 1)
        y_bins = np.linspace(bounds[1][0], bounds[1][1], grid_size + 1)

        for x, y in all_trajectories:
            # 找到粒子所在的网格索引
            x_idx = np.searchsorted(x_bins, x) - 1
            y_idx = np.searchsorted(y_bins, y) - 1
            if 0 <= x_idx < grid_size and 0 <= y_idx < grid_size:
                density[x_idx, y_idx] += 1

        # 对密度矩阵进行高斯滤波
        smoothed_density = gaussian_filter(density, sigma=sigma)

        # 绘制平滑后的密度图
        ax = axs[i]
        im = ax.imshow(
            smoothed_density.T, 
            origin='lower', 
            extent=[bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]],
            cmap=custom_cmap,  # 使用自定义颜色映射
            norm=LogNorm(vmin=1, vmax=smoothed_density.max()),  # 使用对数归一化
            alpha=0.7
        )
        plt.colorbar(im, ax=ax, label='Visit Frequency')
        # 绘制背景（热力图的最小颜色）
        ax = axs[i]
        ax.set_facecolor(min_color)  # 设置背景为热力图的最小颜色
        # 标记全局最优解
        if global_optimum is not None:
            ax.plot(global_optimum[0], global_optimum[1], 'r*', markersize=10, label='Global Optimum')  # 红色星号标记

        # 设置子图标题和范围
        ax.set_title(strategy)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.legend()

    plt.tight_layout()
    plt.show()
# 计算粒子到全局最优解和种群中心的距离
def calculate_particle_distances(history, global_optimum):
    """
    计算每代粒子到全局最优解的距离和到种群中心的距离
    """
    distances_to_optimum = []  # 每代粒子到全局最优解的平均距离
    distances_to_center = []   # 每代粒子到种群中心的平均距离

    for record in history:
        population = record['population']
        # 计算种群中心
        center = np.mean(population, axis=0)
        
        # 计算粒子到全局最优解的距离
        dist_to_optimum = np.linalg.norm(population - global_optimum, axis=1)
        avg_dist_to_optimum = np.mean(dist_to_optimum)
        distances_to_optimum.append(avg_dist_to_optimum)
        
        # 计算粒子到种群中心的距离
        dist_to_center = np.linalg.norm(population - center, axis=1)
        avg_dist_to_center = np.mean(dist_to_center)
        distances_to_center.append(avg_dist_to_center)

    return distances_to_optimum, distances_to_center

# 绘制粒子到全局最优解和种群中心的距离变化
def plot_particle_distances(histories, strategies, global_optimum):
    """
    绘制粒子到全局最优解的距离和到种群中心的距离的变化
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制粒子到全局最优解的距离 
    for i, history in enumerate(histories):
        distances_to_optimum, _ = calculate_particle_distances(history, global_optimum)
        plt.plot(distances_to_optimum, label=f'{strategies[i]}')
    plt.xlabel('Generation')
    plt.ylabel('Average Distance to Global Optimum')
    plt.title('Distance to Global Optimum')
    plt.legend()
    plt.grid()
 

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 定义适应度函数（示例：Rastrigin函数）
    def fitness_function(x):
        return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10, axis=1)

    # 参数设置
    pop_size = 100
    aramge = 50
    bounds = [(-aramge, aramge), (-aramge, aramge)]  # 变量范围
    population = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, 2))
    global_optimum = np.array([0, 0])  # 实际全局最优解
    max_generations = 100

    # 运行两种变异策略
    strategies = ['current-to-pBest-w/1', 'current-to-Amean-w/I']
    histories = []

    # DE/current-to-pBest-w/1（增强开发能力）
    history_pbest = differential_evolution(
        population=population.copy(),
        fitness_function=fitness_function,
        bounds=bounds,
        F=0.3,  # 较小的 F，增强开发
        CR=0.5,  # 较小的 CR，增强开发
        max_generations=max_generations,
        use_crossover=True,
        accept_all=False,
        mutation_strategy='current-to-pBest-w/1',
        global_optimum=global_optimum,
        archive_size=20,  # 较小的 archive_size，增强开发
        e=0.5,  # 较小的 e，增强开发
        p=0.6  # p% 的个体作为 p-best
    )
    histories.append(history_pbest)

    # DE/current-to-Amean-w/I（增强勘探能力）
    history_amean = differential_evolution(
        population=population.copy(),
        fitness_function=fitness_function,
        bounds=bounds,
        F=0.8,  # 较大的 F，增强勘探
        CR=0.9,  # 较大的 CR，增强勘探
        max_generations=max_generations,
        use_crossover=True,
        accept_all=False,
        mutation_strategy='current-to-Amean-w/I',
        global_optimum=global_optimum,
        archive_size=100,  # 较大的 archive_size，增强勘探
        e=0.8  # 较大的 e，增强勘探
    )
    histories.append(history_amean)

    # 动态展示进化过程（子图）
    titles = ['DE/current-to-pBest-w/1 (Enhanced Exploitation)', 'DE/current-to-Amean-w/I (Enhanced Exploration)']
    plot_dynamic_evolution_subplots(histories, bounds, titles, global_optimum=global_optimum)

    # 绘制所有粒子的密度图
    plot_all_particles_density(histories, bounds, global_optimum=global_optimum, grid_size=100, sigma=1.0)

    # 绘制粒子到全局最优解和种群中心的距离变化
    plot_particle_distances(histories, strategies, global_optimum)

    # 输出最终适应度
    for i, strategy in enumerate(strategies):
        print(f"Final fitness for {strategy}: {histories[i][-1]['best_fitness']:.4f}")