import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# 数据准备
data = np.array([
    [537.08, 1.20],
    [783.01, 0.44],
    [629.16, -0.27],
    [1027.57, 0.17],
    [766.75, 0.33],
    [566.03, 0.95],
    [784.13, 0.41],
    [85.64, 0.11],
    [772.51, 0.40],
    [62.28, 0.12],
    [817.99, 0.05],
    [507.46, 0.05],
    [402.60, 0.43],
    [505.23, -0.48],
    [635.33, 4.13],
    [568.16, -0.32],
    [511.20, 1.21],
    [514.43, -0.05],
    [49.54, 0.13],
    [810.81, 18.93],
    [372.44, -3.06],
    [653.03, -0.29],
    [561.28, 2.15],
    [246.03, 0.83],
    [617.18, 0.31],
    [983.34, 3.34],
    [551.30, 0.45],
    [91.58, -0.31],
    [398.47, 0.08],
    [381.33, 0.15],
    [687.92, 2.48],
    [588.49, 0.74],
    [172.39, -0.49],
    [703.59, -1.08],
    [539.02, -0.94],
    [756.49, 0.65],
    [436.57, -0.89],
    [713.88, -6.08],
    [707.20, 0.95],
    [569.20, -0.24],
    [913.15, 2.81],
    [607.48, -0.13],
    [819.31, 1.11],
    [123.06, -0.29],
    [521.06, 0.31],
    [250.84, -0.22],
    [779.72, 0.41],
    [1121.94, -0.06],
    [304.75, 0.51],
    [245.12, 0.26]
])

# 拟合高斯混合模型，成分数为7
gmm = GaussianMixture(n_components=7, random_state=0)
gmm.fit(data)

# 获取结果
labels = gmm.predict(data)
means = gmm.means_

# 计算每个聚类的点数和范围
unique, counts = np.unique(labels, return_counts=True)
cluster_counts = dict(zip(unique, counts))

# 输出每个聚类的范围
cluster_ranges = {}
for i in range(7):
    points = data[labels == i]
    cluster_ranges[i] = {
        'min': points.min(axis=0),
        'max': points.max(axis=0)
    }

# 打印每个聚类的范围
for i in range(7):
    print(f'Cluster {i + 1}: Min = {cluster_ranges[i]["min"]}, Max = {cluster_ranges[i]["max"]}')

# 可视化
plt.figure(figsize=(12, 8))
colors = plt.cm.get_cmap('tab10', 7)

for i in range(7):
    plt.scatter(data[labels == i, 0], data[labels == i, 1], color=colors(i), label=f'Cluster {i + 1} (Count: {cluster_counts[i]})')
    plt.scatter(means[i, 0], means[i, 1], color='k', marker='X', s=200)  # 绘制均值

plt.title('Gaussian Mixture Model Clustering (7 Clusters)')
plt.xlabel('Distance')
plt.ylabel('Slope')
plt.legend()
plt.grid()
plt.show()
