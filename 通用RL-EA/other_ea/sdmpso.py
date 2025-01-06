def sdms_pso(population, num_particles, dim, max_iter, obj_func, arange):
    import numpy as np

    # 参数设置
    m = num_particles          # 每个子群的粒子数量
    n = 5                      # 子群数量，根据论文设定
    R = 20                     # 重组周期
    LP = 5                     # 学习周期
    LA = 40                    # 参数档案长度
    L = 15                     # 局部优化周期
    L_FEs = 100                # 局部搜索的最大适应度评估次数
    Max_FEs = max_iter * m     # 最大适应度评估次数
    c1 = 1.49445
    c2 = 1.49445

    # 初始化
    FEs = 0                    # 已使用的适应度评估次数
    gen = 0                    # 迭代代数
    parameter_set = []         # 参数档案
    success_num = np.zeros(n)  # 成功次数统计

    # 初始化粒子位置和速度
    X = population
    V_max = 0.2 * (arange[1] - arange[0])
    V = np.random.uniform(-V_max, V_max, (num_particles, dim))
    pbest = X.copy()
    pbest_values = obj_func(X)
    FEs += num_particles
    gbest_index = np.argmin(pbest_values)
    gbest = pbest[gbest_index].copy()
    gbest_value = pbest_values[gbest_index]
    gbest_history = [gbest_value]

    # 将粒子随机分成n个子群
    indices = np.arange(num_particles)
    np.random.shuffle(indices)
    sub_swarms = np.array_split(indices, n)

    # while FEs < 0.95 * Max_FEs:
    for i in range(max_iter):
        if (i < 0.95 * max_iter):
            gen += 1

            # 生成每个子群的初始参数iwt
            if len(parameter_set) < LA or np.sum(success_num) <= LP:
                iwt = 0.5 + np.random.rand(n) * 0.4  # 在[0.5, 0.9]之间随机生成
            else:
                median_param = np.median(parameter_set)
                iwt = np.random.normal(median_param, 0.1, n)
            success_num = np.zeros(n)

            # LP个学习周期
            for _ in range(LP):
                for swarm_idx in range(n):
                    swarm = sub_swarms[swarm_idx]
                    w = iwt[swarm_idx]
                    for i in swarm:
                        # 找到粒子的局部最优lbest
                        lbest_value = np.min(pbest_values[swarm])
                        lbest_idx = swarm[np.argmin(pbest_values[swarm])]
                        lbest = pbest[lbest_idx]

                        # 更新速度和位置（根据论文中的公式(3)和(2)）
                        r1 = np.random.rand(dim)
                        r2 = np.random.rand(dim)
                        V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (lbest - X[i])
                        V[i] = np.clip(V[i], -V_max, V_max)
                        X[i] = X[i] + V[i]
                        X[i] = np.clip(X[i], arange[0], arange[1])

                        # 计算适应度值
                        fitness = obj_func(X[i].reshape(1, -1))[0]
                        

                        # 更新个体最优
                        if fitness < pbest_values[i]:
                            pbest[i] = X[i].copy()
                            pbest_values[i] = fitness

                            # 更新局部最优
                            if fitness < lbest_value:
                                lbest = X[i].copy()
                                lbest_value = fitness
                                success_num[swarm_idx] += 1

                                # 更新全局最优
                                if fitness < gbest_value:
                                    gbest = X[i].copy()
                                    gbest_value = fitness

            # 保存成功次数最多的子群的iwt参数
            max_success_idx = np.argmax(success_num)
            parameter_set.append(iwt[max_success_idx])

            # 每隔L代进行局部优化（如使用准牛顿法）
            if gen % L == 0:
                # 对最好的25%子群的lbest进行优化
                num_refine = max(1, int(0.25 * n))
                swarm_fitness = [np.min(pbest_values[swarm]) for swarm in sub_swarms]
                best_swarms = np.argsort(swarm_fitness)[:num_refine]
                for swarm_idx in best_swarms:
                    swarm = sub_swarms[swarm_idx]
                    lbest_idx = swarm[np.argmin(pbest_values[swarm])]
                    # 进行局部优化（示例使用微小扰动，可替换为其他优化方法）
                    for _ in range(L_FEs):
                        perturbation = np.random.uniform(-0.01, 0.01, dim)
                        new_position = pbest[lbest_idx] + perturbation
                        new_position = np.clip(new_position, arange[0], arange[1])
                        fitness = obj_func(new_position.reshape(1, -1))[0]
                        
                        if fitness < pbest_values[lbest_idx]:
                            pbest[lbest_idx] = new_position
                            pbest_values[lbest_idx] = fitness
                            if fitness < gbest_value:
                                gbest = new_position.copy()
                                gbest_value = fitness

            # 每隔R代重新分组
            if gen % R == 0:
                np.random.shuffle(indices)
                sub_swarms = np.array_split(indices, n)

            gbest_history.append(gbest_value)

        # 剩余FEs种群整体进化
        else:
            w = 0.5 + np.random.rand() * 0.4  # 参数行为相同
            r1 = np.random.rand(num_particles, dim)
            r2 = np.random.rand(num_particles, dim)
            V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
            V = np.clip(V, -V_max, V_max)
            X = X + V
            X = np.clip(X, arange[0], arange[1])

            # 计算适应度值
            fitness_values = obj_func(X)
            FEs += num_particles

            # 更新个体最优和全局最优
            better_mask = fitness_values < pbest_values
            pbest[better_mask] = X[better_mask]
            pbest_values[better_mask] = fitness_values[better_mask]

            current_best_value = np.min(pbest_values)
            if current_best_value < gbest_value:
                gbest_index = np.argmin(pbest_values)
                gbest = pbest[gbest_index].copy()
                gbest_value = current_best_value

            gbest_history.append(gbest_value)

    return gbest, gbest_value, gbest_history

def test1():
    import numpy as np 
    from DEPSO import depso
    from jade_de import jade_de
    from standard_pso import standard_pso
    import matplotlib.pyplot as plt
    import sys
    import os

    # 获取当前文件所在目录的父目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # 将父目录添加到 Python 模块搜索路径中
    sys.path.append(parent_dir)

    # 导入 fitness_function 模块
    from fitness_function.cec2017.functions import all_functions

    # 参数设置
    dim = 10          # 维度
    pop_size = 30     # 种群大小
    max_iter = 1000   # 最大迭代次数
    arange = [-100, 100]  # 搜索范围

    # 测试函数
    func = all_functions[3]

    # 初始化种群
    population = np.random.uniform(arange[0], arange[1], (pop_size, dim))

    # 测试SDMS-PSO
    best_sdms, fitness_sdms, history_sdms = sdms_pso(
        np.copy(population),
        pop_size,
        dim,
        max_iter,
        func,
        arange
    )

    #测试depso
    best_depso, fitness_depso, history_depso = depso(
        np.copy(population),
        pop_size,
        dim,
        max_iter,
        func,
        arange
    )

    # 测试jade_de
    best_jade, fitness_jade, history_jade = jade_de(
        np.copy(population),
        pop_size,
        dim,
        max_iter,
        func,
        arange)

    # 测试standard_pso
    best_standard, fitness_standard, history_standard = standard_pso(
        np.copy(population),
        pop_size,
        dim,
        max_iter,
        func,
        arange
    )
    # 打印结果
    print("\nSDMS-PSO最优解:", fitness_sdms)
    print("\ndepso最优解:", fitness_depso)
    print("\nStandard Pso最优解:", fitness_standard)
    print("\njade-de最优解:", fitness_jade)

    

    # y轴以log缩放
    # plt.yscale("log")
    # 绘制收敛曲线
    plt.plot(history_sdms, label='SDMS-PSO')
    plt.plot(history_depso, label='CLPSO')
    plt.plot(history_standard, label='Standard PSO')
    plt.plot(history_jade, label='JADE-DE')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title('SDMS-PSO vs CLPSO on Sphere Function')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test1()