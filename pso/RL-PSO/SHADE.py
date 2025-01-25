import numpy as np
from scipy.stats import norm, cauchy

class SHADE:
    def __init__(self, objective_func, dim, pop_size=100, memory_size=100, max_evaluations=300000, arange = [-100, 100], p_min=0.05):
        self.objective_func = objective_func  # 适应度函数，支持二维np.ndarray输入
        self.dim = dim  # 问题的维度
        self.pop_size = pop_size  # 种群大小
        self.memory_size = memory_size  # 历史记忆大小
        self.max_evaluations = max_evaluations  # 最大评估次数
        self.p_min = p_min  # p_min参数
        self.arange = arange  # 搜索范围
        # 初始化历史记忆
        self.M_CR = np.ones(memory_size) * 0.5  # 交叉率的历史记忆
        self.M_F = np.ones(memory_size) * 0.5  # 缩放因子的历史记忆
        self.k = 0  # 历史记忆的索引

        # 初始化种群
        self.population = np.random.uniform(self.arange[0], self.arange[1], (pop_size, dim))  # 随机初始化种群
        self.fitness = self.objective_func(self.population)  # 计算初始适应度
        self.evaluations = pop_size  # 评估次数

        # 外部存档
        self.archive = []

    def run(self):
        history = []
        max_iter = self.max_evaluations // self.pop_size
        
        for _ in range(max_iter):
            # 向量化生成所有个体的参数
            r = np.random.randint(0, self.memory_size, size=self.pop_size)
            CR = np.clip(norm.rvs(loc=self.M_CR[r], scale=0.1, size=self.pop_size), 0, 1)
            F = np.clip(cauchy.rvs(loc=self.M_F[r], scale=0.1, size=self.pop_size), 0, 1)
            p = np.random.uniform(self.p_min, 0.2, size=self.pop_size)
            
            # 生成所有试验向量
            trial_vectors = np.zeros_like(self.population)
            for i in range(self.pop_size):
                trial_vectors[i] = self.generate_trial_vector(i, CR[i], F[i], p[i])
            
            # 批量计算适应度
            trial_fitness = self.objective_func(trial_vectors)
            self.evaluations += self.pop_size
            
            # 选择操作
            improvement_mask = trial_fitness <= self.fitness
            
            # 记录成功的参数
            S_CR = CR[improvement_mask]
            S_F = F[improvement_mask]
            delta_f = np.abs(self.fitness[improvement_mask] - trial_fitness[improvement_mask])
            
            # 更新种群和适应度
            self.population[improvement_mask] = trial_vectors[improvement_mask]
            self.fitness[improvement_mask] = trial_fitness[improvement_mask]
            
            # 更新存档
            for improved_solution in self.population[improvement_mask]:
                self.archive.append(improved_solution)
                if len(self.archive) > self.pop_size:
                    self.archive.pop(0)
            
            history.append(self.fitness.min())
            
            # 更新历史记忆
            if len(S_CR) > 0:
                mean_CR = np.mean(S_CR)
                mean_F = np.sum(S_F ** 2) / (np.sum(S_F) + 1e-10)  # Lehmer均值
                self.M_CR[self.k] = mean_CR
                self.M_F[self.k] = mean_F
                self.k = (self.k + 1) % self.memory_size
                
            np.clip(self.population, self.arange[0], self.arange[1], out=self.population)
        
        # 返回最佳解
        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index], history

    def generate_trial_vector(self, i, CR_i, F_i, p_i):
        # 选择pbest个体
        pbest_size = max(int(self.pop_size * p_i), 2)
        pbest_indices = np.argsort(self.fitness)[:pbest_size]
        pbest_index = np.random.choice(pbest_indices)

        # 选择随机个体
        r1, r2 = np.random.choice([x for x in range(self.pop_size) if x != i], 2, replace=False)

        # 从种群或存档中选择r2
        if self.archive:
            r2 = np.random.choice(range(len(self.archive)))
            x_r2 = self.archive[r2]
        else:
            x_r2 = self.population[r2]

        # 变异和交叉
        mutant = self.population[i] + F_i * (self.population[pbest_index] - self.population[i]) + F_i * (self.population[r1] - x_r2)
        trial_vector = np.where(np.random.rand(self.dim) < CR_i, mutant, self.population[i])

        # 边界处理
        trial_vector = np.clip(trial_vector, -100, 100)
        return trial_vector

if __name__ == "__main__":
    from cec2017.functions import all_functions 
    import numpy as np
    func = all_functions[9]
    min_fitness = 1e10
    max_fitness = 0
    avg_fitness = 0
    for i in range(10):
        # 示例用法
        dim = 10
        pop_size = 50
        memory_size = 100
        max_evaluations = 100000 
        arange = [-100, 100] 
        shade = SHADE(func, dim, pop_size, memory_size, max_evaluations, arange)
        best_solution, best_fitness, history = shade.run()
        print("Best solution:", best_solution)
        print("Best fitness:", best_fitness)
        min_fitness = min(min_fitness, best_fitness)
        max_fitness = max(max_fitness, best_fitness)
        avg_fitness += best_fitness
    avg_fitness /= 10
    print("Min fitness:", min_fitness)
    print("Max fitness:", max_fitness)
    print("Avg fitness:", avg_fitness)
    import matplotlib.pyplot as plt
    plt.plot(np.log10(history))
    plt.title("SHADE Convergence")
    plt.xlabel("Evaluations")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig("results_pso/RL-PSO/SHADE.png")
    plt.show()