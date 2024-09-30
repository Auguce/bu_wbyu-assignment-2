import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

class KMeans:
    def __init__(self, n_clusters=3, init_method='random', max_iter=300):
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.max_iter = max_iter
        self.centroids = None
        self.centroid_history = []
        self.empty_clusters_history = []
        self.current_iter = 0
        self.converged = False
        self.snapshots = []  # 用于存储快照的Base64编码图像

    def initialize_centroids(self, data):
        if self.init_method == 'random':
            indices = random.sample(range(data.shape[0]), self.n_clusters)
            self.centroids = data[indices]
        elif self.init_method == 'farthest_first':
            self.centroids = []
            self.centroids.append(data[random.randint(0, data.shape[0] - 1)])
            for _ in range(1, self.n_clusters):
                distances = np.min([np.linalg.norm(data - centroid, axis=1) for centroid in self.centroids], axis=0)
                next_centroid = data[np.argmax(distances)]
                self.centroids.append(next_centroid)
            self.centroids = np.array(self.centroids)
        elif self.init_method == 'kmeans++':
            self.centroids = []
            self.centroids.append(data[random.randint(0, data.shape[0] - 1)])
            for _ in range(1, self.n_clusters):
                distances = np.min([np.linalg.norm(data - centroid, axis=1) for centroid in self.centroids], axis=0)
                probabilities = distances / np.sum(distances)
                next_centroid = data[np.random.choice(range(data.shape[0]), p=probabilities)]
                self.centroids.append(next_centroid)
            self.centroids = np.array(self.centroids)
        elif self.init_method == 'manual':
            if self.centroids is None:
                raise ValueError("Centroids must be provided for manual initialization.")
            if len(self.centroids.shape) != 2 or self.centroids.shape[1] != 2:
                raise ValueError("Manual centroids must be a 2D array with shape (n_clusters, 2).")
            self.centroids = np.array(self.centroids)
        else:
            raise ValueError(f"Unknown initialization method: {self.init_method}")

        # Initialize history
        self.centroid_history = [self.centroids.copy()]
        self.empty_clusters_history = []
        self.current_iter = 0
        self.converged = False

        # Assign initial labels
        self.labels = self.assign_clusters(data)

        # Take initial snapshot
        self.snapshot(data, step=True)

    def assign_clusters(self, data):
        distances = np.array([np.linalg.norm(data - centroid, axis=1) for centroid in self.centroids])
        return np.argmin(distances, axis=0)

    def update_centroids(self, data, labels):
        new_centroids = []
        empty_clusters = []
        for i in range(self.n_clusters):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0))
            else:
                empty_clusters.append(i)
                new_centroids.append(self.centroids[i])
        self.centroids = np.array(new_centroids)
        return empty_clusters

    def snapshot(self, data, step=False):
        fig, ax = plt.subplots()
        colors = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
        for i in range(self.n_clusters):
            cluster_points = data[self.labels == i]
            if len(cluster_points) > 0:
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i % len(colors)], s=10, alpha=0.6)
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], c='black', marker='x', s=100)
        ax.set_title(f'KMeans Clustering - Iteration {self.current_iter}')
        ax.set_xlim(data[:,0].min()-1, data[:,0].max()+1)
        ax.set_ylim(data[:,1].min()-1, data[:,1].max()+1)
        ax.set_aspect('equal')
        plt.close(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img = img.convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        self.snapshots.append(img_str)

    def step(self, data):
        if self.converged or self.current_iter >= self.max_iter:
            return None

        self.labels = self.assign_clusters(data)
        empty_clusters = self.update_centroids(data, self.labels)
        self.centroid_history.append(self.centroids.copy())
        self.empty_clusters_history.append(empty_clusters)
        self.current_iter += 1

        # Take snapshot after updating centroids
        self.snapshot(data)

        # Check for convergence
        if self.current_iter > 0 and np.allclose(self.centroid_history[-2], self.centroids):
            self.converged = True

        return self.labels, self.centroids, self.centroid_history, empty_clusters


    def fit(self, data):
        self.initialize_centroids(data)
        while not self.converged and self.current_iter < self.max_iter:
            step_result = self.step(data)
            if step_result is None:
                break
        final_empty_clusters = self.empty_clusters_history[-1] if self.empty_clusters_history else []
        return self.assign_clusters(data), self.centroids, self.centroid_history, final_empty_clusters

    def predict(self, data):
        return self.assign_clusters(data)


if __name__ == "__main__":
    # 创建随机数据集
    # np.random.seed(42)  # 删除或注释此行
    data1 = np.random.randn(100, 2) + np.array([5, 5])
    data2 = np.random.randn(100, 2) + np.array([-5, -5])
    data3 = np.random.randn(100, 2) + np.array([5, -5])
    data = np.vstack((data1, data2, data3))
    
    # 初始化 KMeans 算法
    kmeans = KMeans(n_clusters=3, init_method='random', max_iter=300)
    
    # 运行 KMeans 聚类
    labels, centroids, centroid_history, final_empty_clusters = kmeans.fit(data)

    # 输出结果
    print(f"Final centroids:\n{centroids}")
    print(f"Cluster labels:\n{labels}")
    print(f"Final empty clusters: {final_empty_clusters}")

    # 可视化结果
    import matplotlib.pyplot as plt

    # 绘制数据点，使用不同的颜色表示不同的簇
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
    
    # 绘制质心
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100, label='Centroids')
    
    plt.title('KMeans Clustering Result')
    plt.legend()
    plt.show()

