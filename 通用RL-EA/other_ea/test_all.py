def test1():
    import numpy as np 
    from DEPSO import depso
    from jade_de import jade_de
    from standard_pso import standard_pso
    from sdmpso import sdms_pso
    from DE import de 
    from RDE import rde
    from SHDE import shade
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
    func_id = 7                                         
    func = all_functions[func_id]

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
    print("已完成SDMS-PSO")
    #测试depso
    best_depso, fitness_depso, history_depso = depso(
        np.copy(population),
        pop_size,
        dim,
        max_iter,
        func,
        arange
    )
    print("已完成DEPSO")
    # 测试jade_de
    best_jade, fitness_jade, history_jade = jade_de(
        np.copy(population),
        pop_size,
        dim,
        max_iter,
        func,
        arange)
    print("已完成JADE-DE")
    # 测试standard_pso
    best_standard, fitness_standard, history_standard = standard_pso(
        np.copy(population),
        pop_size,
        dim,
        max_iter,
        func,
        arange
    )
    print("已完成Standard PSO")
    # 测试rde
    best_rde, fitness_rde, history_rde = rde(
        np.copy(population),
        pop_size,
        dim,
        max_iter,
        func,
        arange
    )
    print("已完成RDE")
    # 测试DE
    best_de, fitness_de, history_de = de(
        np.copy(population),
        pop_size,
        dim,
        max_iter,
        func,
        arange
    )
    print("已完成DE")
    # 测试shade
    best_shade, fitness_shade, history_shade = shade(
        np.copy(population),
        pop_size,
        dim,
        max_iter,
        func,
        arange
    )

    # 打印结果
    print("\nSDMS-PSO最优解:", fitness_sdms)
    print("depso最优解:", fitness_depso)
    print("Standard Pso最优解:", fitness_standard)
    print("jade-de最优解:", fitness_jade)
    print("RDE最优解:", fitness_rde)
    print("DE最优解:", fitness_de)
    print("SHADE最优解:", fitness_shade)
    

    

    # y轴以log缩放
    plt.yscale("log")
    # 绘制收敛曲线
    plt.plot(history_sdms, label='SDMS-PSO')
    plt.plot(history_depso, label='DEPSO')
    plt.plot(history_standard, label='Standard PSO')
    plt.plot(history_jade, label='JADE-DE')
    plt.plot(history_rde, label='RDE')
    plt.plot(history_de, label='DE')
    plt.plot(history_shade, label='SHADE')
    
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title(f'all algorithms on Function{func_id}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    test1()