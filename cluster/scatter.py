from sklearn import preprocessing, metrics
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering   # 聚类模型
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pickle as pkl
import random

# s = np.random.dirichlet(range(1,11), 30000)
# data = s
# with open("random_data_5000.pkl", "wb") as f:
#     pkl.dump(data, f)
with open("random_data_50000.pkl", "rb") as f:
    data = pkl.load(f)

# 1 用主成分分析法对传入的数据进行降维
pca = PCA(n_components=2)   # 输出降到几维
data = pca.fit_transform(data)  # 载入N维

# 2 聚类
# colors = ['#c04851','#fbb957','#8cc269','#66a9c9','#806d9e']
colors = ['#c04851','#3d6daa','#806d9e','#66a9c9','#fbb957']
pred_y_colors = []
model = KMeans(n_clusters=5)
model.fit(data)  # 完成聚类
pred_y = model.predict(data)  # 预测点在哪个聚类中
pred_y= list(pred_y)
# 用center控制聚类的类别标号固定,按照x值大小来固定类别
centers =  model.cluster_centers_
centers_x = [i[0] for i in centers]
centers_x = np.array(centers_x)
centers_sort = np.argsort(centers_x)
result = []

# 3 调整散点分布的扁度
def get_correlated_dataset(random_list, dependency, mu, scale):
    random_list = np.array(random_list)
    dependent = random_list.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]
mu = (0,0)
scale = (1,1)
dependency = [[0.85, -0.35],[0.15, -0.65]]

# # 1 均匀抽样
# random_list = random.sample(range(0, 50000), 10000)  #50000抽10000
# for i in range(5):
#     data_temp = []
#     for index in random_list:
#         if pred_y[index]==centers_sort[i]:
#             data_temp.append(data[index].tolist())
#     # x, y = get_correlated_dataset(data_temp, dependency, mu, scale)
#     # data_temp_x = np.array(x)
#     # data_temp_y = np.array(y)
#     data_temp = np.array(data_temp)
#     plt.scatter(data_temp[:,0], data_temp[:,1], c=colors[i], marker='o', s=1.2, linewidths=0,
#                 label='stratum' + str(i + 1))
#         # plt.legend(loc='upper left', markerscale=10)  # 每次都要执行
# 2 全集合
# plt.figure(figsize=(5.5,3.5))
# for i in range(5):
#     data_temp = []
#     for index,r in enumerate(pred_y):
#         if r==centers_sort[i]:
#             data_temp.append(data[index].tolist())
#     x, y = get_correlated_dataset(data_temp, dependency, mu, scale)
#     data_temp_x = np.array(x)
#     data_temp_y = np.array(y)
#     plt.scatter(data_temp_x, data_temp_y, c=colors[i], marker='o', s=0.8, linewidths=0,
#                 label='stratum' + str(i + 1), alpha=1)
#     plt.legend(loc='upper left', markerscale=7)  # 每次都要执行，用于绘制图注
# for i in range(5):
#     data_temp = []
#     for index,r in enumerate(pred_y):
#         if r==centers_sort[i]:
#             data_temp.append(data[index].tolist())
#     data_temp  = np.array(data_temp)
#     plt.scatter(data_temp[:,0], data_temp[:,1], c='gray', marker='o', s=0.8, linewidths=0,
#                 label='stratum' + str(i + 1), alpha=1)
#     # plt.scatter(data_temp[:,0], data_temp[:,1], c=colors[i], marker='o', s=1, linewidths=0,
#     #             label='stratum' + str(i + 1), alpha=1)
#     # plt.legend(loc='upper left', markerscale=7)  # 每次都要执行，用于绘制图注

# # 3 不均匀抽样(两组抽样结果)
# # 3.1 版本1
# for i in range(5):
#     data_temp = []
#     if i == 0:
#         num = 4000
#     elif i == 1:
#         num = 2000
#     elif i == 2:
#         num = 4000
#     elif i == 3:
#         num = 500
#     else:
#         num = 500
#     for index,r in enumerate(pred_y):
#         if pred_y[index]==centers_sort[i]:
#             data_temp.append(data[index].tolist())
#     data_index = random.sample(range(0, len(data_temp)), num)
#     data_temp = [data_temp[ind] for ind in data_index]
#     data_temp = np.array(data_temp)
#     plt.scatter(data_temp[:, 0], data_temp[:, 1], c=colors[i], marker='o', s=1.2, linewidths=0, label='stratum'+str(i+1))
#     # plt.legend(loc='upper left', markerscale=10)  # 每次都要执行
# 3.2 版本2
for i in range(5):
    data_temp = []
    if i == 0:
        num = 500
    elif i == 1:
        num = 500
    elif i == 2:
        num = 3000
    elif i == 3:
        num = 3500
    else:
        num = 3500
    for index,r in enumerate(pred_y):
        if pred_y[index]==centers_sort[i]:
            data_temp.append(data[index].tolist())
    random_index = random.sample(range(0, len(data_temp)), num)
    random_list = [data_temp[ind] for ind in random_index]
    # x, y = get_correlated_dataset(random_list, dependency, mu, scale)
    # data_temp_x = np.array(x)
    # data_temp_y = np.array(y)
    random_list = np.array(random_list)
    plt.scatter(random_list[:,0], random_list[:,1], c='gray', marker='o', s=1.2, linewidths=0, label='stratum'+str(i+1))
    # plt.scatter(random_list[:,0], random_list[:,1], c=colors[i], marker='o', s=1.2, linewidths=0, label='stratum'+str(i+1))
    # plt.legend(loc='upper left', markerscale=10)  # 每次都要执行

plt.axis('off')
plt.axis([-0.2,0.2,-0.2,0.2])
plt.savefig('cluster_10000_gray.pdf', format='pdf', dpi=200, bbox_inches="tight", transparent=True)
plt.show()
