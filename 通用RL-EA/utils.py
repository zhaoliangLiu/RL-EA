# utils.py
import numpy as np

import numpy as np

def compute_state(pre_population, cur_population, pBest, gBest, fitness_function, max_iter, cur_iter, count_k):
    pop_size = len(cur_population)
    
    # 1. 相对位置特征排名
    population_center = np.mean(cur_population, axis=0)
    distances_to_center = np.linalg.norm(cur_population - population_center, axis=1)
    relative_position_rank = distances_to_center.argsort().argsort()
    
    # 2. 速度特征排名
    velocity = np.linalg.norm(cur_population - pre_population, axis=1)
    relative_velocity_rank = velocity.argsort().argsort()
    
    # 3. 探索开发指标排名
    distance_to_gbest = np.linalg.norm(cur_population - gBest, axis=1)
    distance_to_pbest = np.linalg.norm(cur_population - pBest, axis=1)
    # 添加数值稳定性检查
    distance_to_pbest = np.clip(distance_to_pbest, 1e-10, None)  # 确保分母不会太小
    exploration_ratio = np.clip(distance_to_gbest / distance_to_pbest, 0, 100)  # 限制比值范围
    exploration_ratio_rank = exploration_ratio.argsort().argsort()
    
    # 4. 种群多样性排名
    diversity = np.std(cur_population, axis=0)
    diversity_sum = np.sum(diversity)
    diversity_array = np.full(pop_size, diversity_sum)
    diversity_rank = diversity_array.argsort().argsort()
    
    # 5. 时间特征
    time_remaining = max(0.01, (max_iter - cur_iter) / max_iter)
    time_feature = np.full(pop_size, time_remaining)
    count_k_array = np.full(pop_size, count_k)
    
    # 将所有特征组合
    state_features = np.column_stack((
        relative_position_rank,
        relative_velocity_rank,
        exploration_ratio_rank,
        diversity_rank,
        time_feature,
        count_k_array
    ))
    
    return state_features

if __name__ == "__main__":
    from operators import operators
    from simple_function.simple_functions import all_functions
    import numpy as np

    pop = np.random.rand(50, 10) * 200 - 100
    pre_pop = np.random.rand(50, 10) * 200 - 100
    pBest = np.random.rand(50, 10) * 200 - 100
    gBest = np.random.rand(10) * 200 - 100
    fitness_function = all_functions[0]
    max_iter = 1000
    cur_iter = 100
    count_k = [0] * 50
    state_features =  compute_state(pre_pop, pop, pBest, gBest, fitness_function, max_iter, cur_iter, count_k)                                
    print(state_features)

