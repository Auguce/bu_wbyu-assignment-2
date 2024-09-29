from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from kmeans import KMeans

app = Flask(__name__)

# 全局变量用于存储数据
data = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_data', methods=['POST'])
def generate_data():
    global data
    # 创建随机数据集
    np.random.seed(42)
    data1 = np.random.randn(100, 2) + np.array([5, 5])
    data2 = np.random.randn(100, 2) + np.array([-5, -5])
    data3 = np.random.randn(100, 2) + np.array([5, -5])
    data = np.vstack((data1, data2, data3))
    return jsonify({"message": "Data generated successfully!"})

@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    global data
    if data is None:
        return jsonify({"error": "No data available. Please generate a dataset first."}), 400

    # 获取用户选择的参数
    n_clusters = int(request.form['n_clusters'])
    init_method = request.form['init_method']
    
    # 运行 KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, init_method=init_method, max_iter=300)
    labels, centroids = kmeans.fit(data)

    # 可视化结果
    img = plot_clusters(data, labels, centroids)
    
    # 返回图像到前端
    return jsonify({"image": img})

def plot_clusters(data, labels, centroids):
    # 绘制数据点，使用不同的颜色表示不同的簇
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5)
    
    # 绘制质心
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
    
    # 保存图像到内存
    buffer = BytesIO()
    plt.title('KMeans Clustering Result')
    plt.legend()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    
    # 编码为 base64
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

if __name__ == '__main__':
    app.run(port=3000, debug=True)
