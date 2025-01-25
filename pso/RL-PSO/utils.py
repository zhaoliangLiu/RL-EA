import numpy as np
from sklearn_extra.cluster import KMedoids  # 支持曼哈顿距离的聚类算法

def manhattan_clustering(particle_history, max_clusters=10):
    """
    使用曼哈顿距离对粒子历史记录进行聚类，并自动确定最佳聚类数 K。
    
    参数:
    - particle_history: 列表，每个元素是一个 np.array，表示一个粒子的位置。
    - max_clusters: 最大聚类数，用于手肘法。
    
    返回:
    - cluster_centers: 聚类中心，形状为 (K, n_features)。
    """
    # 将粒子历史记录转换为二维数组
    X = np.array(particle_history)
    
    # 使用手肘法确定最佳聚类数 K
    distortions = []
    K_range = range(1, max_clusters + 1)
    
    for k in K_range:
        kmedoids = KMedoids(n_clusters=k, metric='manhattan', init='k-medoids++', random_state=42)
        kmedoids.fit(X)
        distortions.append(kmedoids.inertia_)  # 使用 inertia_ 作为 distortion
    
    # 找到拐点（手肘点）
    k = find_elbow_point(distortions)
    print(f"Optimal number of clusters (K): {k}")
    
    # 使用最佳 K 进行聚类
    kmedoids = KMedoids(n_clusters=k, metric='manhattan', init='k-medoids++', random_state=42)
    kmedoids.fit(X)
    
    # 返回聚类中心
    cluster_centers = kmedoids.cluster_centers_
    return cluster_centers

def find_elbow_point(distortions):
    """
    找到手肘图的拐点。
    
    参数:
    - distortions: 每个 K 对应的 distortion 值。
    
    返回:
    - elbow_point: 最佳聚类数 K。
    """
    # 计算每个点的二阶差分
    second_derivatives = np.diff(distortions, 2)
    
    # 找到二阶差分最大的点（拐点）
    elbow_point = np.argmax(second_derivatives) + 2  # 二阶差分从 K=3 开始
    return elbow_point

