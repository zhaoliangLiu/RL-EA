
import numpy as np
import torch
import matplotlib.pyplot as plt
from dedqn import DEDQN  # 假设之前实现的DEDQN类保存在dedqn.py中

class ClassicDE:
    """经典单策略差分进化算法"""
    def __init__(self, dim, bounds, npop, strategy, F=0.5, CR=0.9):
        self.dim = dim
        self.bounds = np.array(bounds)
        self.npop = npop
        self.strategy = strategy
        self.F = F
        self.CR = CR
        
        # 初始化种群
        self.population = self.initialize_population()
        self.best_idx = 0
        self.fitness = None
        
    def initialize_population(self):
        pop = np.zeros((self.npop, self.dim))
        for d in range(self.dim):
            l, u = self.bounds[d]
            pop[:, d] = l + np.random.rand(self.npop) * (u - l)
        return pop
    
    def evaluate_fitness(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.best_idx = np.argmin(self.fitness)
        self.best = self.population[self.best_idx]
    
    def mutate(self):
        mutants = []
        for i in range(self.npop):
            indices = [idx for idx in np.random.choice(self.npop, 3, replace=False) if idx != i]
            if self.strategy == 'rand/1':
                r1, r2, r3 = indices[:3]
                mutant = self.population[r1] + self.F*(self.population[r2] - self.population[r3])
            elif self.strategy == 'best/1':
                r1, r2 = indices[:2]
                mutant = self.best + self.F*(self.population[r1] - self.population[r2])
            mutant = np.clip(mutant, self.bounds[:,0], self.bounds[:,1])
            mutants.append(mutant)
        return np.array(mutants)
    
    def crossover(self, mutants):
        trials = np.copy(self.population)
        cross_points = np.random.rand(self.npop, self.dim) < self.CR
        rand_idx = np.random.randint(0, self.dim, self.npop)
        for i in range(self.npop):
            cross_points[i, rand_idx[i]] = True
            trials[i] = np.where(cross_points[i], mutants[i], self.population[i])
        return trials
    
    def evolve(self, func, generations):
        self.evaluate_fitness(func)
        best_fitness = [np.min(self.fitness)]
        
        for _ in range(generations):
            mutants = self.mutate()
            trials = self.crossover(mutants)
            
            # 评估试验向量
            trial_fitness = np.array([func(ind) for ind in trials])
            
            # 选择
            for i in range(self.npop):
                if trial_fitness[i] < self.fitness[i]:
                    self.population[i] = trials[i]
                    self.fitness[i] = trial_fitness[i]
            
            self.best_idx = np.argmin(self.fitness)
            self.best = self.population[self.best_idx]
            best_fitness.append(self.fitness[self.best_idx])
        
        return best_fitness

def run_comparison(dim=30, npop=50, generations=100, runs=10):
    # 实验参数
    bounds = [(-100, 100)]*dim
    strategies = ['DEDQN', 'rand/1', 'best/1']
    
    # 存储结果
    results = {s: [] for s in strategies}
    
    for run in range(runs):
        print(f"Run {run+1}/{runs}")
        np.random.seed(run)
        torch.manual_seed(run)
        
        # 初始化算法
        dedqn = DEDQN(dim, bounds, npop, ['rand/1', 'best/1', 'current-to-best/1'])
        de_rand = ClassicDE(dim, bounds, npop, 'rand/1')
        de_best = ClassicDE(dim, bounds, npop, 'best/1')
        
        # 定义测试函数（以Sphere为例）
        def test_func(x):
            return np.sum(x**2)
        
        # 运行DEDQN
        dedqn.global_opt = np.zeros(dim)
        dedqn_fitness = dedqn.evolve(test_func, generations)
        results['DEDQN'].append(dedqn_fitness)
        
        # 运行DE/rand/1
        de_rand.evaluate_fitness(test_func)
        rand_fitness = de_rand.evolve(test_func, generations)
        results['rand/1'].append(rand_fitness)
        
        # 运行DE/best/1
        de_best.evaluate_fitness(test_func)
        best_fitness = de_best.evolve(test_func, generations)
        results['best/1'].append(best_fitness)
    
    # 分析结果
    plt.figure(figsize=(10, 6))
    for strategy in strategies:
        # 计算平均收敛曲线
        avg_curve = np.mean(results[strategy], axis=0)
        std = np.std(results[strategy], axis=0)
        generations = np.arange(len(avg_curve))
        
        plt.plot(generations, avg_curve, label=strategy)
        plt.fill_between(generations, avg_curve-std, avg_curve+std, alpha=0.2)
    
    plt.yscale('log')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Algorithm Comparison on Sphere Function')
    plt.legend()
    plt.grid()
    plt.show()
    
    # 输出统计结果
    final_fitness = {s: [run[-1] for run in results[s]] for s in strategies}
    print("\nFinal Fitness Statistics:")
    for s in strategies:
        print(f"{s}: Mean={np.mean(final_fitness[s]):.2e}, Std={np.std(final_fitness[s]):.2e}")

if __name__ == "__main__":
    run_comparison(dim=30, npop=50, generations=100, runs=5)