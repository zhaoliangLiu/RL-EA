import gym
import numpy as np
from gym import spaces
from sklearn.preprocessing import StandardScaler
# 假设 cec2017.functions 模块已安装并可用
from cec2017.functions import all_functions
import scipy.stats as stats

from RAND_TOBEST import rand_to_best_de 
from SHADE import SHADE
class PSO_SHADE_Env(gym.Env):
    """
    将PSO+SHADE的进化过程同时封装入一个Gym环境。
    每个episode对应一次对目标函数的优化。
    """
    def __init__(
        self, 
        population_size=50, 
        dim=10, 
        max_iter=1000,
        memory_size=100,  # SHADE的记忆库大小
        x_min=-100.0, 
        x_max=100.0,
        
        num_function=1,
        start_function_id=0
    ):
        super(PSO_SHADE_Env, self).__init__()
        
        # 基础参数设置
        self.population_size = population_size
        self.dim = dim
        self.max_iter = max_iter
        self.x_min = x_min
        self.x_max = x_max
       
      
        # DEPSO参数
        self.V_max = 0.2 * (x_max - x_min)
        
        # 动作和观察空间保持不变
        self.action_space = spaces.Discrete(16)  # 2^4 = 16 种可能的组合
        self.n_features = 5
        self.observation_dim = self.population_size * self.n_features + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )
        # 适应度函数id
        self.fitness_function_id = start_function_id
        self.num_function = num_function

        self.reset()

    def reset(self):
        """重置环境，初始化所有必要的参数和状态"""
        # 初始化种群和速度
        self.population = np.random.uniform(
            self.x_min, self.x_max, 
            (self.population_size, self.dim)
        )
        self.velocity = np.random.uniform(
            -self.V_max, self.V_max,
            (self.population_size, self.dim)
        )
        
        # 选择适应度函数
        self.fitness_function = all_functions[self.fitness_function_id]
        self.fitness_function_id = self.fitness_function_id 
        
        # 计算初始适应度
        self.fitness = self.fitness_function(self.population)

        self.fitness_0 = np.min(self.fitness)
        
        # PSO相关
        self.pbest_positions = self.population.copy()
        self.pbest_fitness = self.fitness.copy()
        self.gbest_position = self.population[np.argmin(self.fitness)]
        self.gbest_fitness = np.min(self.fitness)
        
        # 未更新次数
        self.unupdate_count = 0
        self.improve_fitness = 0
        self.improve_fitness_rate = 0 

        # 分组
        self.group()
        
        self.cur_iter = 0
        return self._get_full_state()

    def _initialize_population(self):
        """
        初始化PSO种群、速度、pbest等
        """
        self.population = np.random.uniform(
            self.x_min, self.x_max, (self.population_size, self.dim)
        )
        self.velocity = np.random.uniform(
            -abs(self.x_min) * self.velocity_rate,
            abs(self.x_max) * self.velocity_rate,
            (self.population_size, self.dim)
        )

        self.fitness = self.fitness_function(self.population)
        self.pbest_positions = self.population.copy()
        self.pbest_fitness   = self.fitness.copy()

        self.gbest_position = self.population[np.argmin(self.fitness)]
        self.gbest_fitness  = np.min(self.fitness)

    def group(self):
        """
        根据适应度和距离种群中心的距离将种群分为4组：
        1. 适应度在前50%，距离在前50%
        2. 适应度在前50%，距离在后50%
        3. 适应度在后50%，距离在前50%
        4. 适应度在后50%，距离在后50%
        """
        # 计算适应度排名
        fitness_ranks = np.argsort(self.fitness)  # 从小到大排序的索引
        median_fitness_rank = self.population_size // 2
        
        # 计算到种群中心的距离
        population_center = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - population_center, axis=1)
        distance_ranks = np.argsort(distances)  # 从小到大排序的索引
        median_distance_rank = self.population_size // 2
        
        # 初始化分组标签
        self.group_indices = np.zeros(self.population_size, dtype=int)
        
        # 创建布尔掩码
        is_fitness_good = np.zeros(self.population_size, dtype=bool)
        is_distance_good = np.zeros(self.population_size, dtype=bool)
        
        # 设置前50%的适应度和距离掩码
        is_fitness_good[fitness_ranks[:median_fitness_rank]] = True
        is_distance_good[distance_ranks[:median_distance_rank]] = True
        
        # 分组
        self.group_indices = np.where(
            is_fitness_good & is_distance_good, 0,      # 组1: 好适应度，近距离
            np.where(
                is_fitness_good & ~is_distance_good, 1,  # 组2: 好适应度，远距离
                np.where(
                    ~is_fitness_good & is_distance_good, 2,  # 组3: 差适应度，近距离
                    3  # 组4: 差适应度，远距离
                )
            )
        )
        
        # # 验证分组结果
        # group_sizes = [np.sum(self.group_indices == i) for i in range(4)]
        # expected_size = self.population_size // 4
        # print(f"Group sizes: {group_sizes} (expected ~{expected_size} each)")

    def _get_full_state(self):
        """
        构建包含以下特征的状态向量：
        1. 当前最佳适应度
        2. 平均适应度
        3. 适应度标准差
        4. 适应度比上一代提升多少
        5. 历史最佳适应度变化率（EWMA）
        6. 种群分布半径（多样性）
        7. 全局最优解未更新的次数
        8. 当前代数/总进化代数
        """
        population_center = np.mean(self.population, axis=0)
        
        # 计算全局特征
        current_best_fitness = self.gbest_fitness
        mean_fitness = np.mean(self.fitness)
        fitness_std = np.std(self.fitness)
        fitness_improvement = self.gbest_fitness - self.previous_best_fitness
        
        # 计算EWMA变化率
        alpha = 0.9  # 衰减因子
        if self.cur_iter > 0:
            current_change = (self.gbest_fitness - self.previous_best_fitness) / (abs(self.previous_best_fitness) + 1e-8)
            self.ewma_change_rate = alpha * self.ewma_change_rate + (1 - alpha) * current_change
        
        # 计算种群分布半径
        distances_to_center = np.linalg.norm(self.population - population_center, axis=1)
        population_radius = np.max(distances_to_center)
        
        # 构建每个粒子的特征矩阵
        feature_matrix = []
        for i in range(self.population_size):
            particle_features = [
                current_best_fitness,
                mean_fitness,
                fitness_std,
                fitness_improvement,
                self.ewma_change_rate,
                population_radius,
                float(self.gbest_update_count),
                self.cur_iter / self.max_iter
            ]
            feature_matrix.append(particle_features)
        
        # 标准化特征
        feature_matrix = np.array(feature_matrix, dtype=np.float32)
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(feature_matrix)
        
        # 添加全局参数
        iter_ratio = self.cur_iter / self.max_iter
        global_params = np.array([iter_ratio, 0.0, 0.0, 0.0], dtype=np.float32)  # 最后三个参数根据实际需求调整
        
        # 合并特征
        state = np.concatenate([
            standardized_features.flatten(),
            global_params
        ]).astype(np.float32)
        
        # 验证形状
        expected_shape = (self.observation_dim,)
        if state.shape != expected_shape:
            raise ValueError(f"State shape mismatch: {state.shape} vs {expected_shape}")
            
        return state

    def update_parameters(self):
        """
        更新PSO参数
        """
        iter_ratio = self.cur_iter / self.max_iter
        self.w = self.w_max * (self.w_min / self.w_max) ** iter_ratio
        self.c1 = self.c1_max - (self.c1_max - self.c1_min) * (iter_ratio ** 2)
        self.c2 = self.c2_min + (self.c2_max - self.c2_min) * (iter_ratio ** 2)

    def shade_select_CR_F(self):
        """
        从SHADE的M_CR, M_F中抽取CR, F值
        (示例：对每个粒子从记忆库随机选一个k，再加一点随机扰动)
        """
        k_indices = np.random.randint(0, self.H, size=self.population_size)
        CR_values = np.random.normal(self.M_CR[k_indices], 0.1)
        F_values  = np.random.normal(self.M_F[k_indices], 0.1)

        CR_values = np.clip(CR_values, 0.0, 1.0)
        F_values  = np.clip(F_values, 0.0, 1.0)
        return CR_values, F_values, k_indices

    def shade_update_memory(self, 
                            successful_CR, 
                            successful_F, 
                            d_fitness):
        """
        利用成功进化的CR/F更新M_CR, M_F
        """
        if len(successful_CR) == 0:
            return
        # 计算权重
        weights = np.array(d_fitness) / np.sum(d_fitness)
        mean_CR = np.sum(weights * np.array(successful_CR))

        F_array = np.array(successful_F)
        mean_F = (np.sum(weights * (F_array**2)) /
                  (np.sum(weights * F_array) + 1e-10))

        self.M_CR[self.memory_index] = mean_CR
        self.M_F[self.memory_index] = mean_F
        # 环形更新
        self.memory_index = (self.memory_index + 1) % self.H

    def update_particles(self, action):
        """更新粒子位置，使用向量化的DE/rand/2进行勘探，DE/rand-to-best/1进行开发"""
        # 确保action是numpy数组
        action = np.array(action)
        self.pop_action = np.array([action[g_id] for g_id in self.group_indices])
        exploration_mask = self.pop_action == 0  # DE/rand/2
        exploitation_mask = self.pop_action == 1  # DE/rand-to-best/1
        
        new_population = self.population.copy()
        new_fitness = self.fitness.copy()
        
        # 处理勘探粒子（使用DE/rand/2）
        explore_indices = np.where(exploration_mask)[0]
        if len(explore_indices) > 0:
            # 一次性生成所有随机索引矩阵
            r_matrix = np.array([
                [i] + list(np.random.choice(
                    [x for x in range(self.population_size) if x != i], 
                    5, 
                    replace=False
                )) for i in explore_indices
            ])
            
            # 一次性生成所有F和CR值
            F = np.clip(np.random.normal(0.5, 0.1, size=len(explore_indices)), 0, 1)
            CR = np.clip(np.random.normal(0.7, 0.1, size=len(explore_indices)), 0, 1)
            
            # 向量化DE/rand/2变异
            mutants = (self.population[r_matrix[:, 1]] + 
                      F[:, np.newaxis] * (self.population[r_matrix[:, 2]] - self.population[r_matrix[:, 3]]) 
                    # + F[:, np.newaxis] * (self.population[r_matrix[:, 4]] - self.population[r_matrix[:, 5]]))
                    )
            
            # 向量化交叉
            cross_points = np.random.rand(len(explore_indices), self.dim) < CR[:, np.newaxis]
            # 确保每个试验向量至少有一个维度来自变异向量
            rows_without_cross = ~np.any(cross_points, axis=1)
            random_dims = np.random.randint(0, self.dim, size=np.sum(rows_without_cross))
            cross_points[rows_without_cross, random_dims] = True
            
            # 生成试验向量
            trials = np.where(cross_points, mutants, self.population[explore_indices])
            trials = np.clip(trials, self.x_min, self.x_max)
            
            # 向量化评估和选择
            trial_fitness = self.fitness_function(trials)
            improvements = trial_fitness <= self.fitness[explore_indices]
            new_population[explore_indices[improvements]] = trials[improvements]
            new_fitness[explore_indices[improvements]] = trial_fitness[improvements]
        
        # 处理开发粒子（使用DE/rand-to-best/1）
        exploit_indices = np.where(exploitation_mask)[0]
        if len(exploit_indices) > 0:
            current_best = self.population[np.argmin(self.fitness)]
            
            # 一次性生成所有随机索引矩阵
            r_matrix = np.array([
                [i] + list(np.random.choice(
                    [x for x in range(self.population_size) if x != i], 
                    3, 
                    replace=False
                )) for i in exploit_indices
            ])
            
            # 一次性生成所有F和CR值
            F = np.clip(np.random.normal(0.5, 0.1, size=len(exploit_indices)), 0, 1)
            CR = np.clip(np.random.normal(0.7, 0.1, size=len(exploit_indices)), 0, 1)
            
            # 向量化DE/rand-to-best/1变异
            mutants = (self.population[r_matrix[:, 1]] + 
                      F[:, np.newaxis] * (current_best - self.population[r_matrix[:, 1]]) +
                      F[:, np.newaxis] * (self.population[r_matrix[:, 2]] - self.population[r_matrix[:, 3]]))
            
            # 向量化交叉
            cross_points = np.random.rand(len(exploit_indices), self.dim) < CR[:, np.newaxis]
            # 确保每个试验向量至少有一个维度来自变异向量
            rows_without_cross = ~np.any(cross_points, axis=1)
            random_dims = np.random.randint(0, self.dim, size=np.sum(rows_without_cross))
            cross_points[rows_without_cross, random_dims] = True
            
            # 生成试验向量
            trials = np.where(cross_points, mutants, self.population[exploit_indices])
            trials = np.clip(trials, self.x_min, self.x_max)
            
            # 向量化评估和选择
            trial_fitness = self.fitness_function(trials)
            improvements = trial_fitness <= self.fitness[exploit_indices]
            new_population[exploit_indices[improvements]] = trials[improvements]
            new_fitness[exploit_indices[improvements]] = trial_fitness[improvements]
        
        # 更新种群和适应度
        self.population = new_population
        self.fitness = new_fitness
        
        # 更新全局最优
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.gbest_fitness:
            self.gbest_position = self.population[best_idx].copy()
            self.gbest_fitness = self.fitness[best_idx]

    def step(self, action):
        # 将 Discrete 动作转换回二进制形式
        binary_action = [int(x) for x in format(action, '04b')]
        
        old_gbest = self.gbest_fitness
        old_population_diversity = self.calculate_diversity()  # 计算更新前的种群多样性
        
        self.update_particles(binary_action)
        
        new_population_diversity = self.calculate_diversity()  # 计算更新后的种群多样性
        
        # 计算多个方面的奖励
        improvement_reward = 0
        if self.gbest_fitness < old_gbest:
            improvement_reward = (old_gbest - self.gbest_fitness) / self.fitness_0 # 对数奖励
            if self.cur_iter > 0.75 * self.max_iter:
                improvement_reward *= 1000  # 动态缩放
        
        # 多样性奖励：鼓励探索
        diversity_reward = 0
        # if np.any(binary_action == 1):  # 对于使用DE的粒子组
        #     diversity_change = (new_population_diversity - old_population_diversity)/ (old_population_diversity + 1e-8)
        #     diversity_reward = np.tanh(diversity_change)  # 使用tanh限制奖励范围
         
        
        
        # 组合奖励
        time_factor = self.cur_iter / self.max_iter
        total_reward = (
            improvement_reward * (1.0 - time_factor) +  # 早期更注重改进
            diversity_reward * time_factor 
        )
        
        self.cur_iter += 1
        done = self.cur_iter >= self.max_iter
        info = {
            'improvement_reward': improvement_reward,
            'diversity_reward': diversity_reward
        }
        self.group()
        return self._get_full_state(), total_reward, done, info

    def calculate_diversity(self):
        """计算种群多样性"""
        center = np.mean(self.population, axis=0)
        distances = np.linalg.norm(self.population - center, axis=1)
        return np.mean(distances)

    

    def seed(self, seed=None):
        """设置环境的随机种子"""
        if seed is not None:
            np.random.seed(seed)
            # 如果使用了PyTorch，也需要设置其随机种子
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
        return [seed]
    def render(self):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('TkAgg')  # 明确指定后端
        
        # 当维度是2维时，绘制种群的散点图
        if self.dim == 2:
            # 如果没有初始化图形对象，则创建
            if not hasattr(self, 'fig'):
                plt.ion()  # 打开交互模式
                self.fig, self.ax = plt.subplots(figsize=(8, 6))  # 设置更大的图形尺寸
                self.scatter = self.ax.scatter([], [])
                self.ax.set_xlim(self.x_min, self.x_max)
                self.ax.set_ylim(self.x_min, self.x_max)
                self.ax.set_title('PSO-SHADE Population')
                self.ax.set_xlabel('X')
                self.ax.set_ylabel('Y')
                plt.show()  # 显式调用show
                
            # 更新散点图数据
            self.scatter.set_offsets(self.population)
            self.ax.set_title(f'Iteration: {self.cur_iter}, Best Fitness: {self.gbest_fitness:.4f}')
            
            # 绘制全局最优点
            if hasattr(self, 'gbest_scatter'):
                self.gbest_scatter.remove()
            self.gbest_scatter = self.ax.scatter(
                self.gbest_position[0], 
                self.gbest_position[1], 
                color='red', 
                marker='*', 
                s=200, 
                label='Global Best'
            )
            
            # 添加图例
            if not hasattr(self, 'legend_added'):
                self.ax.legend()
                self.legend_added = True
            
            try:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(0.1)  # 短暂暂停以展示动画效果
            except Exception as e:
                print(f"渲染错误: {e}")
                plt.close('all')  # 关闭所有图形
                delattr(self, 'fig')  # 删除图形属性，下次重新创建

# 使用示例
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    from DE_variants import de_rand_2, de_rand_to_best_1
    dims = [10,30,40]
    for dim in dims:
        for i in range(0, 30):
            # 测试参数设置
            max_iter = 5000
            start_function_id = i
            population_size = 2 * dim
            dim = dim
            bounds = (-100, 100)
            
            # 记录所有算法的历史
            RL_PSO_history = []
            
            # 1. 运行RL-PSO-DE
            env = PSO_SHADE_Env(
                population_size=population_size, 
                dim=dim, 
                max_iter=max_iter, 
                start_function_id=start_function_id
            )
            
            obs = env.reset()
            for i in range(max_iter):
                action = env.action_space.sample()  # 在实际应用中，这里应该是RL智能体的输出
                obs, reward, done, info = env.step(action)
                RL_PSO_history.append(env.gbest_fitness)
                # if i % 100 == 0:
                    # explore_particles = np.sum(env.pop_action == 0)
                    # exploit_particles = np.sum(env.pop_action == 1)
                    # print(f"RL-PSO-DE iter: {i}, explore: {explore_particles}, exploit: {exploit_particles}, Best: {env.gbest_fitness}")
            
            # 2. 运行DE/rand/2
            best_solution_rand2, best_fitness_rand2, history_rand2 = de_rand_2(
                func=env.fitness_function,
                dim=dim,
                pop_size=population_size,
                max_iter=max_iter,
                arange=(env.x_min, env.x_max)
            )
            
            # 3. 运行DE/rand-to-best/1
            best_solution_randtobest, best_fitness_randtobest, history_randtobest = de_rand_to_best_1(
                func=env.fitness_function,
                dim=dim,
                pop_size=population_size,
                max_iter=max_iter,
                arange=(env.x_min, env.x_max)
            )
            
            # 4. 运行SHADE
            shade = SHADE(
                env.fitness_function, 
                dim, 
                population_size, 
                memory_size=100, 
                max_evaluations=max_iter*population_size, 
                arange=(env.x_min, env.x_max)
            )
            best_solution_shade, best_fitness_shade, history_shade = shade.run()
            
            # 打印最终结果
            print("\nFinal Results:")
            print(f"RL-PSO-DE Best Fitness: {env.gbest_fitness}")
            print(f"DE/rand/2 Best Fitness: {best_fitness_rand2}")
            print(f"DE/rand-to-best/1 Best Fitness: {best_fitness_randtobest}")
            print(f"SHADE Best Fitness: {best_fitness_shade}")
            
            # 绘制收敛曲线
            plt.figure(figsize=(10, 6))
            plt.plot(np.log10(RL_PSO_history), label='RL-PSO-DE')
            plt.plot(np.log10(history_rand2), label='DE/rand/2')
            plt.plot(np.log10(history_randtobest), label='DE/rand-to-best/1')
            plt.plot(np.log10(history_shade), label='SHADE')
            plt.xlabel('Iterations')
            plt.ylabel('log10(Fitness)')
            plt.title(f'Optimization Progress on F{start_function_id+1} with dim={dim}')
            plt.legend()
            plt.grid(True)
            # 如果文件夹不存在，则创建
            import os
            if not os.path.exists(f'pso/RL-PSO/DE-result/{dim}'):
                os.makedirs(f'pso/RL-PSO/DE-result/{dim}')
            plt.savefig(f'pso/RL-PSO/DE-result/{dim}/{start_function_id+1}.png')
            # plt.show()