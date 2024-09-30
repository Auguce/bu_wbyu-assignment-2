from flask import Flask, render_template, request, jsonify
import numpy as np
from kmeans import KMeans
import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

app = Flask(__name__)

data = None
current_kmeans = None  # 全局变量存储当前的KMeans实例

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_data', methods=['POST'])
def generate_data():
    global data, current_kmeans
    # 生成一个随机数据集，每次调用都会生成不同的数据
    data1 = np.random.randn(100, 2) + np.random.uniform(-10, 10, size=(1, 2))
    data2 = np.random.randn(100, 2) + np.random.uniform(-10, 10, size=(1, 2))
    data3 = np.random.randn(100, 2) + np.random.uniform(-10, 10, size=(1, 2))
    data = np.vstack((data1, data2, data3))
    
    # 重置当前的KMeans实例
    current_kmeans = None

    # 返回数据点坐标
    data_points = data.tolist()
    
    return jsonify({"message": "Data generated successfully!", "data_points": data_points})

@app.route('/run_kmeans', methods=['POST'])
def run_kmeans():
    global data, current_kmeans
    if data is None:
        return jsonify({"error": "No data available. Please generate a dataset first."}), 400

    n_clusters = int(request.form['n_clusters'])
    init_method = request.form['init_method']
    
    # 获取手动选择的质心
    manual_centroids = request.form.get('manual_centroids')
    if init_method == 'manual' and manual_centroids:
        manual_centroids = json.loads(manual_centroids)
        
        # 检查手动质心数量是否与n_clusters一致
        if len(manual_centroids) != n_clusters:
            return jsonify({"error": f"Number of manual centroids ({len(manual_centroids)}) does not match n_clusters ({n_clusters}), please choose more reasonable point"}), 400
        
        # 将点击的像素坐标转换为数据坐标
        canvas_width, canvas_height = 800, 600
        data_x_min, data_x_max = data[:, 0].min(), data[:, 0].max()
        data_y_min, data_y_max = data[:, 1].min(), data[:, 1].max()
        
        mapped_centroids = []
        for centroid in manual_centroids:
            x, y = float(centroid['x']), float(centroid['y'])
            mapped_x = data_x_min + (x / canvas_width) * (data_x_max - data_x_min)
            mapped_y = data_y_max - (y / canvas_height) * (data_y_max - data_y_min)  # 注意 y 轴方向
            mapped_centroids.append([mapped_x, mapped_y])
        
        manual_centroids = np.array(mapped_centroids)
    
    # 初始化并运行KMeans
    current_kmeans = KMeans(n_clusters=n_clusters, init_method=init_method, max_iter=300)
    
    # 如果使用手动初始化方法，提前设置 centroids
    if init_method == 'manual' and manual_centroids is not None:
        current_kmeans.centroids = manual_centroids
    
    try:
        labels, centroids, centroid_history, final_empty_clusters = current_kmeans.fit(data)
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    # 返回聚类结果
    cluster_labels = labels.tolist()
    centroids = centroids.tolist()
    centroid_history = [c.tolist() for c in centroid_history]
    
    response = {
        "cluster_labels": cluster_labels,
        "centroids": centroids,
        "centroid_history": centroid_history,
        "empty_clusters": final_empty_clusters  # 新增字段
    }
    
    return jsonify(response)

@app.route('/step_kmeans', methods=['POST'])
def step_kmeans():
    global data, current_kmeans
    if data is None:
        return jsonify({"error": "No data available. Please generate a dataset first."}), 400
    
    if current_kmeans is None:
        n_clusters = int(request.form.get('n_clusters', 3))
        init_method = request.form.get('init_method', 'random')
        
        manual_centroids = request.form.get('manual_centroids')
        if init_method == 'manual' and manual_centroids:
            manual_centroids = json.loads(manual_centroids)
            
            if len(manual_centroids) != n_clusters:
                return jsonify({"error": f"Number of manual centroids ({len(manual_centroids)}) does not match n_clusters ({n_clusters})."}), 400
            
            canvas_width, canvas_height = 800, 600
            data_x_min, data_x_max = data[:, 0].min(), data[:, 0].max()
            data_y_min, data_y_max = data[:, 1].min(), data[:, 1].max()
            
            mapped_centroids = []
            for centroid in manual_centroids:
                x, y = float(centroid['x']), float(centroid['y'])
                mapped_x = data_x_min + (x / canvas_width) * (data_x_max - data_x_min)
                mapped_y = data_y_max - (y / canvas_height) * (data_y_max - data_y_min)
                mapped_centroids.append([mapped_x, mapped_y])
            
            manual_centroids = np.array(mapped_centroids)
        else:
            manual_centroids = None
        
        current_kmeans = KMeans(n_clusters=n_clusters, init_method=init_method, max_iter=300)
        
        if init_method == 'manual' and manual_centroids is not None:
            current_kmeans.centroids = manual_centroids
        
        try:
            current_kmeans.initialize_centroids(data)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
    
    try:
        step_result = current_kmeans.step(data)
        if step_result is None:
            if current_kmeans.converged:
                return jsonify({"error": "KMeans has already converged."}), 400
            else:
                return jsonify({"error": "Maximum iterations reached."}), 400

        labels, centroids, centroid_history, empty_clusters = step_result
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    converged = current_kmeans.converged

    response = {
        "cluster_labels": labels.tolist(),
        "centroids": centroids.tolist(),
        "centroid_history": [c.tolist() for c in centroid_history],
        "empty_clusters": empty_clusters,
        "converged": converged
    }
    
    # 检查空簇并在返回的 JSON 中包含
    if empty_clusters:
        response["warning"] = f"{len(empty_clusters)} cluster(s) have no data points assigned. Please choose more appropriate centroids."

    return jsonify(response)

@app.route('/reset_kmeans', methods=['POST'])
def reset_kmeans():
    global current_kmeans
    current_kmeans = None
    return jsonify({"message": "KMeans has been reset."})

if __name__ == '__main__':
    app.run(port=3000, debug=True)
