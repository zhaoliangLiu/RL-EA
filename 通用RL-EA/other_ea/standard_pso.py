def standard_pso(population, num_particles, dim, max_iter, obj_func, arange):
    import numpy as np

    w = 0.729    # 惯性权重
    c1 = 1.49445  # 认知权重
    c2 = 1.49445  # 社会权重

    # 初始化粒子的位置和速度
    V_arange = (arange[0] * 0.2, arange[1] * 0.2)
    X = population
    V = np.random.uniform(V_arange[0], V_arange[1], (num_particles, dim))  # 速度

    # 初始化个体最佳位置和值
    pbest = X.copy()
    pbest_values = obj_func(X)

    # 初始化全局最佳位置和值
    gbest_index = np.argmin(pbest_values)
    gbest = pbest[gbest_index]
    gbest_value = pbest_values[gbest_index]

    # 历史全局最佳值
    gbest_history = []

    for t in range(max_iter):
        # 更新速度
        r1 = np.random.rand(num_particles, dim)
        r2 = np.random.rand(num_particles, dim)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, V_arange[0], V_arange[1])

        # 更新位置
        X = X + V
        X = np.clip(X, arange[0], arange[1])

        # 评估目标函数
        values = obj_func(X)

        # 更新个体最佳
        better_mask = values < pbest_values
        pbest[better_mask] = X[better_mask]
        pbest_values[better_mask] = values[better_mask]

        # 更新全局最佳
        current_best_index = np.argmin(pbest_values)
        current_best_value = pbest_values[current_best_index]
        if current_best_value < gbest_value:
            gbest_value = current_best_value
            gbest = pbest[current_best_index]

        # 记录当前全局最佳值
        gbest_history.append(gbest_value)

    return gbest, gbest_value, gbest_history

# 在主函数测试

if __name__ == "__main__":
    # 测试该函数
    import os
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    # 获取当前文件所在目录的父目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # 将父目录添加到 Python 模块搜索路径中
    sys.path.append(parent_dir)

    # 导入 fitness_function 模块
    from fitness_function.cec2017.functions import all_functions
    population = np.random.rand(50, 10) * 200 - 100
    for i in range(1, len(all_functions)):
        func = all_functions[i]
        gbest, gbest_value, gbest_history = standard_pso(population, 50, 10, 1000, func, (-100, 100))
        print(gbest_value)
        plt.plot(gbest_history)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.legend(["PSO"])
        plt.title(f"PSO on Sphere Function{i}")
        plt.show()

