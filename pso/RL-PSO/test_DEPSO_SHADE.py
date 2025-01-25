import numpy as np
import pandas as pd
from SHADE import SHADE
from DEPSO import depso
from cec2017.functions import all_functions
import os

def test_and_record():
    # 创建结果目录
    results_dir = "results_pso/RL-PSO"
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化结果列表
    results = []
    
    # 测试参数
    dim = 30
    pop_size = 100
    memory_size = 100
    max_evaluations = 300000
    max_iter = int(max_evaluations / pop_size)
    arange = [-100, 100]
    
    # 对每个函数进行测试
    for i in range(30):  # CEC2017有30个测试函数
        func = all_functions[i + 1]
        
        # 运行SHADE
        shade = SHADE(func, dim, pop_size, memory_size, max_evaluations, arange)
        best_solution_shade, best_fitness_shade, _ = shade.run()
        
        # 运行DEPSO
        best_solution_depso, best_fitness_depso, _ = depso(pop_size, dim, max_iter, func, arange)
        
        # 记录结果
        results.append({
            'Function_ID': i + 1,
            'SHADE_Best': best_fitness_shade,
            'DEPSO_Best': best_fitness_depso
        })
        
        print(f"Function {i+1} completed:")
        print(f"SHADE Best: {best_fitness_shade}")
        print(f"DEPSO Best: {best_fitness_depso}")
        print("-" * 50)
    
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, 'optimization_results.csv')
    df.to_csv(csv_path, index=False)
    
    # 计算并打印统计信息
    stats = df.agg({
        'SHADE_Best': ['mean', 'std', 'min', 'max'],
        'DEPSO_Best': ['mean', 'std', 'min', 'max']
    })
    
    stats_path = os.path.join(results_dir, 'statistics.csv')
    stats.to_csv(stats_path)
    
    print("\nResults have been saved to:", csv_path)
    print("Statistics have been saved to:", stats_path)
    
    return df, stats

if __name__ == "__main__":
    results_df, stats_df = test_and_record()
    
    # 打印总体统计信息
    print("\nOverall Statistics:")
    print("\nSHADE Statistics:")
    print(f"Mean: {stats_df['SHADE_Best']['mean']}")
    print(f"Std: {stats_df['SHADE_Best']['std']}")
    print(f"Min: {stats_df['SHADE_Best']['min']}")
    print(f"Max: {stats_df['SHADE_Best']['max']}")
    
    print("\nDEPSO Statistics:")
    print(f"Mean: {stats_df['DEPSO_Best']['mean']}")
    print(f"Std: {stats_df['DEPSO_Best']['std']}")
    print(f"Min: {stats_df['DEPSO_Best']['min']}")
    print(f"Max: {stats_df['DEPSO_Best']['max']}")