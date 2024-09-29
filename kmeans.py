import numpy as np
import random

class KMeans:
    def __init__(self, n_clusters=3, init_method='random', max_iter=300):
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.max_iter = max_iter
        self.centroids = None

    def initialize_centroids(self, data):
        if self.init_method == 'random':
            # 随机选择数据点作为初始质心
            indices = random.sample(range(data.shape[0]), self.n_clusters)
            self.centroids = data[indices]
        elif self.init_method == 'farthest_first':
            # 最远距离初始化方法
            self.centroids = []
            # 随机选择第一个质心
            self.centroids.append(data[random.randint(0, data.shape[0] - 1)])
            for _ in range(1, self.n_clusters):
                # 找到距离已选择质心最远的数据点作为新的质心
                distances = np.min([np.linalg.norm(data - centroid, axis=1) for centroid in self.centroids], axis=0)
                next_centroid = data[np.argmax(distances)]
                self.centroids.append(next_centroid)
            self.centroids = np.array(self.centroids)
        elif self.init_method == 'kmeans++':
            # KMeans++ 初始化方法
            self.centroids = []
            # 随机选择第一个质心
            self.centroids.append(data[random.randint(0, data.shape[0] - 1)])
            for _ in range(1, self.n_clusters):
                distances = np.min([np.linalg.norm(data - centroid, axis=1) for centroid in self.centroids], axis=0)
                probabilities = distances / np.sum(distances)
                next_centroid = data[np.random.choice(range(data.shape[0]), p=probabilities)]
                self.centroids.append(next_centroid)
            self.centroids = np.array(self.centroids)
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")

    def assign_clusters(self, data):
        # 计算每个点到各个质心的距离，分配到最近的质心
        distances = np.array([np.linalg.norm(data - centroid, axis=1) for centroid in self.centroids])
        return np.argmin(distances, axis=0)

    def update_centroids(self, data, labels):
        # 重新计算每个簇的质心
        new_centroids = []
        for i in range(self.n_clusters):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                # 如果簇为空，保持原质心
                new_centroids.append(self.centroids[i])
        self.centroids = np.array(new_centroids)

    def fit(self, data):
        # 初始化质心
        self.initialize_centroids(data)

        for _ in range(self.max_iter):
            # 分配数据点到最近的质心
            labels = self.assign_clusters(data)
            # 更新质心
            previous_centroids = self.centroids.copy()
            self.update_centroids(data, labels)
            
            # 检查收敛
            if np.all(previous_centroids == self.centroids):
                break

        return labels, self.centroids

    def predict(self, data):
        # 为新数据分配簇
        return self.assign_clusters(data)





if __name__ == "__main__":
    # 创建随机数据集
    np.random.seed(42)
    data1 = np.random.randn(100, 2) + np.array([5, 5])
    data2 = np.random.randn(100, 2) + np.array([-5, -5])
    data3 = np.random.randn(100, 2) + np.array([5, -5])
    data = np.vstack((data1, data2, data3))

    # 初始化 KMeans 算法
    kmeans = KMeans(n_clusters=3, init_method='random', max_iter=300)
    
    # 运行 KMeans 聚类
    labels, centroids = kmeans.fit(data)

    # 输出结果
    print(f"Final centroids:\n{centroids}")
    print(f"Cluster labels:\n{labels}")

    # 可视化结果
    import matplotlib.pyplot as plt

    # 绘制数据点，使用不同的颜色表示不同的簇
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
    
    # 绘制质心
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
    
    plt.title('KMeans Clustering Result')
    plt.legend()
    plt.show()