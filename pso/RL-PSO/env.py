import gym
import numpy as np
from gym import spaces
from cec2017.functions import all_functions
import scipy.stats as stats

class PSO_SHADE_Env(gym.Env):
    def __init__(
        self, 
        population_size=50, 
        dim=10, 
        max_iter=1000,
        memory_size=100,
        x_min=-100.0, 
        x_max=100.0,
        p_min=0.05, 
        num_function=1,
        start_function_id=0
    ):
        super(PSO_SHADE_Env, self).__init__()
        
        # 基础参数
        self.population_size = population_size
        self.dim = dim
        self.max_iter = max_iter
        self.x_min = x_min
        self.x_max = x_max
        self.p_min = p_min
        
        # SHADE参数
        self.memory_size = memory_size
        self.M_CR = np.ones(memory_size) * 0.5
        self.M_F = np.ones(memory_size) * 0.5
        self.k = 0
        self.archive = []
        
        # 动作和观察空间
        self.action_space = spaces.Discrete(16)
        self.observation_dim = 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )
        
        self.fitness_function_id = start_function_id
        self.num_function = num_function
        self.reset()

    def reset(self):
        self.fitness_function = all_functions[self.fitness_function_id]
        self.fitness_function_id = (self.fitness_function_id + 1) % self.num_function
        
        # 初始化种群
        self.population = np.random.uniform(
            self.x_min, self.x_max,
            (self.population_size, self.dim)
        )
        self.velocity = np.random.uniform(
            -0.2*(self.x_max - self.x_min),
            0.2*(self.x_max - self.x_min),
            (self.population_size, self.dim)
        )
        self.fitness = self.fitness_function(self.population)
        
        # 最优记录
        self.pbest_positions = self.population.copy()
        self.pbest_fitness = self.fitness.copy()
        self.gbest_idx = np.argmin(self.fitness)
        self.gbest_position = self.population[self.gbest_idx].copy()
        self.gbest_fitness = self.fitness[self.gbest_idx]
        
        # SHADE参数
        self.M_CR[:] = 0.5
        self.M_F[:] = 0.5
        self.k = 0
        self.archive = []
        
        # 进化状态
        self.group()
        self.cur_iter = 0
        self.gbest_fitness_old = self.gbest_fitness
        self.not_update_count = 0
        self.survival = np.ones(self.population_size, dtype=int)
        self.action_count = np.zeros(16, dtype=int)
        
        return self._get_full_state()

    def group(self):
        # 保持原有分组逻辑
        fitness_ranks = np.argsort(self.fitness)
        median = self.population_size // 2
        
        # 适应度前50%
        fitness_good = np.zeros(self.population_size, dtype=bool)
        fitness_good[fitness_ranks[:median]] = True
        
        # 距离中心前50%
        center = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - center, axis=1)
        distance_ranks = np.argsort(distances)
        distance_good = np.zeros(self.population_size, dtype=bool)
        distance_good[distance_ranks[:median]] = True
        
        # 四分组逻辑
        self.group_indices = np.select(
            condlist=[
                fitness_good & distance_good,
                fitness_good & ~distance_good,
                ~fitness_good & distance_good
            ],
            choicelist=[0, 1, 2],
            default=3
        )

    def _get_full_state(self):
        # 保持原有状态计算
        center = np.mean(self.population, axis=0)
        dist_center = np.linalg.norm(self.population - center, axis=1)
        dist_gbest = np.linalg.norm(self.population - self.gbest_position, axis=1)
        
        features = np.array([
            np.log(1 + self.gbest_fitness),
            np.log(1 + np.mean(self.fitness)),
            np.log(1 + np.std(self.fitness)),
            np.log(1 + abs(self.gbest_fitness_old - self.gbest_fitness)),
            np.log(1 + np.sum(dist_center)),
            np.log(1 + np.sum(dist_gbest)),
            self.not_update_count / 10.0,
            self.cur_iter / self.max_iter
        ], dtype=np.float32)
        
        # 前6项归一化
        features[:6] = (features[:6] - np.min(features[:6])) / (np.ptp(features[:6]) + 1e-8)
        return features

    def _vectorized_mutation(self, action):
        # 解析动作到每个粒子
        binary_action = np.unpackbits(np.array([action], dtype=np.uint8))[-4:]
        strategy_mask = binary_action[self.group_indices]  # [pop_size]
        
        # 批量生成参数
        r = np.random.randint(0, self.memory_size, self.population_size)
        CR = np.clip(stats.norm(loc=self.M_CR[r], scale=0.1).rvs(), 0, 1)
        F = np.clip(stats.cauchy(loc=self.M_F[r], scale=0.1).rvs(), 0, 1)
        
        # 预生成随机索引
        rand_idx = np.array([self._get_rand_indices(i) for i in range(self.population_size)])
        
        # DE/rand/1变异
        mask_rand = (strategy_mask == 0)
        r1, r2, r3 = rand_idx[mask_rand, 0], rand_idx[mask_rand, 1], rand_idx[mask_rand, 2]
        mutants_rand = self.population[r1] + F[mask_rand, None] * (self.population[r2] - self.population[r3])
        
        # DE/best/1变异
        mask_best = (strategy_mask == 1)
        r1, r2 = rand_idx[mask_best, 0], rand_idx[mask_best, 1]
        mutants_best = self.gbest_position + F[mask_best, None] * (self.population[r1] - self.population[r2])
        
        # 合并变异体
        mutants = np.empty_like(self.population)
        mutants[mask_rand] = mutants_rand
        mutants[mask_best] = mutants_best
        
        # 向量化交叉
        cross_mask = np.random.rand(*mutants.shape) < CR[:, None]
        cross_mask |= (np.arange(self.dim) == np.random.randint(self.dim))[None, :]  # 确保至少一个维度交叉
        trials = np.where(cross_mask, mutants, self.population)
        
        return np.clip(trials, self.x_min, self.x_max), F, CR

    def _get_rand_indices(self, idx):
        candidates = np.setdiff1d(np.arange(self.population_size), idx)
        return np.random.choice(candidates, 3, replace=False)

    def update_particles(self, action):
        # 生成试验个体
        trials, F, CR = self._vectorized_mutation(action)
        
        # 批量评估
        trial_fitness = self.fitness_function(trials)
        
        # 选择改进个体
        improved = trial_fitness < self.fitness
        self.population[improved] = trials[improved]
        self.fitness[improved] = trial_fitness[improved]
        
        # 更新全局最优
        new_gbest_idx = np.argmin(self.fitness)
        if self.fitness[new_gbest_idx] < self.gbest_fitness:
            self.gbest_position = self.population[new_gbest_idx].copy()
            self.gbest_fitness = self.fitness[new_gbest_idx]
        
        # 更新SHADE参数记忆
        if np.any(improved):
            valid = improved
            self.M_CR[self.k] = np.mean(CR[valid])
            self.M_F[self.k] = np.mean(F[valid])
            self.k = (self.k + 1) % self.memory_size
            
            # 更新存档
            self.archive.extend(self.population[~improved].tolist())
            if len(self.archive) > self.population_size:
                del self.archive[:len(self.archive)-self.population_size]

    def step(self, action):
        self.action_count[action] += 1
        old_gbest = self.gbest_fitness
        old_fitness = self.fitness.copy()
        
        # 执行更新
        self.update_particles(action)
        
        # 计算奖励
        improved = self.fitness < old_fitness
        self.survival = np.where(improved, 1, self.survival + 1)

        # 修复cep计算逻辑
        if np.any(improved):
            cep = np.mean(1.0 / self.survival[improved])  # 直接对改进的个体计算均值
        else:
            cep = 0.0  # 若无改进，设为0避免除零错误

        entropy = stats.entropy(np.histogram(self.population, bins=20)[0])
        reward = 10 * cep + 0.1 * np.log1p(entropy)
                
        # 全局改进奖励
        if self.gbest_fitness < old_gbest:
            reward += np.log1p(old_gbest - self.gbest_fitness)
            self.not_update_count = 0
        else:
            self.not_update_count += 1
            
        # 最终奖励增强
        self.cur_iter += 1
        done = self.cur_iter >= self.max_iter
        if done:
            reward += 50 * np.log1p(1.0 / (self.gbest_fitness + 1e-8))
        
        return self._get_full_state(), reward, done, {}

    def render(self, mode='human'):
        pass

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 测试配置
    dim = 10
    max_iter = 1000
    func_id = 9
    
    # 初始化环境
    env = PSO_SHADE_Env(dim=dim, max_iter=max_iter, start_function_id=func_id)
    
    # 测试策略
    strategies = {
        '全勘探(0000)': 0, 
        '全开发(1111)': 15,
        '混合策略(0011)': 3
    }
    
    # 运行测试
    results = {}
    for name, action in strategies.items():
        print(f"Testing {name}...")
        env.reset()
        history = []
        for _ in range(max_iter):
            _, _, done, _ = env.step(action)
            history.append(env.gbest_fitness)
            if done: break
        results[name] = history
    
    # 可视化
    plt.figure(figsize=(12,6))
    for name, data in results.items():
        plt.plot(np.log10(data), label=name)
    plt.xlabel('Iterations')
    plt.ylabel('Log10(Fitness)')
    plt.title(f'CEC2017 f{func_id+1} Optimization Traces')
    plt.legend()
    plt.grid(True)
    plt.show()