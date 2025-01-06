import numpy as np
def f1(points):
    points = np.array(points)
    return np.sum(points**2, axis=1)

def f2(points):
    points = np.array(points)
    n = points.shape[1]
    a = 20
    b = 0.2
    c = 2 * np.pi
    sum_sq_term = np.sum(points**2, axis=1) / n
    cos_term = np.sum(np.cos(c * points), axis=1) / n
    return -a * np.exp(-b * np.sqrt(sum_sq_term)) - np.exp(cos_term) + a + np.exp(1)

all_functions = [f1, f2]
