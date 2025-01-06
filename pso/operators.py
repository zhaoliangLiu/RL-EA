import numpy as np

def o1(population, pBest, gBest):
    """
    pso: w = 1, c1 = 1.5, c2 = 1.5
    """
    shape = population.shape 
    w = 1
    c1 = 1.5
    c2 = 1.5
    pop_new = w * population + c1 * np.random.rand() * (pBest - population) + c2 * np.random.rand() * (gBest - population)

    return pop_new

def o2(population, pBest, gBest):
    """
    pso: w = 1, c1 = 2.5, c2 = 1.5
    """
    shape = population.shape 
    w = 1
    c1 = 2.5
    c2 = 1.5
    pop_new = w * population + c1 * np.random.rand() * (pBest - population) + c2 * np.random.rand() * (gBest - population)

    return pop_new

def o3(population, pBest, gBest):
    """
    pso: w = 1, c1 = 1.5, c2 = 2.5
    """
    shape = population.shape 
    w = 1
    c1 = 1.5
    c2 = 2.5
    pop_new = w * population + c1 * np.random.rand() * (pBest - population) + c2 * np.random.rand() * (gBest - population)

    return pop_new


def o4(population, pBest, gBest):
    """
    DE/rand/1: F = 0.5, CR = 0.5
    """
    F = 0.5
    CR = 0.5

    shape = population.shape 
    
    # 变异
    pop_new = population + F * (population - population[np.random.randint(0, shape[0], shape[0])]) 
    # 交叉
    pop_new = [x if np.random.rand() < CR else pop_new[i] for i, x in enumerate(population)]
    pop_new = np.array(pop_new)

    return pop_new

def o5(population, pBest, gBest):
    """
    DE/best/1: F = 0.5, CR = 0.5
    """
    F = 0.5
    CR = 0.5
    shape = population.shape 
    pop_new = gBest + F * (population[np.random.randint(0, shape[0], shape[0])] - population[np.random.randint(0, shape[0], shape[0])])
    # 交叉
    pop_new = [x if np.random.rand() < CR else pop_new[i] for i, x in enumerate(population)]
    pop_new = np.array(pop_new)

    return pop_new

def o6(population, pBest, gBest):
    """
    DE/current-to-best/1: F = 0.5, CR = 0.5
    """
    F = 0.5
    CR = 0.5
    shape = population.shape 
    pop_new = population + F * (gBest - population) + F * (population[np.random.randint(0, shape[0], shape[0])] - population[np.random.randint(0, shape[0], shape[0])])
    # 交叉
    pop_new = [x if np.random.rand() < CR else pop_new[i] for i, x in enumerate(population)]
    pop_new = np.array(pop_new)

    return pop_new

operators = [o1, o2, o3, o4, o5, o6] 


if __name__ == "__main__":
    population = np.array([[1, 2], [3, 4], [5, 6]])
    pBest = np.array([[1, 2], [3, 4], [5, 6]])
    gBest = np.array([[1, 2]])
    print(f'population: {population}')
    print(f'pBest: {pBest}')
    print(f'gBest: {gBest}')
    print(f'test o1')
    print(o1(population, pBest, gBest))

