#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
from sklearn import preprocessing, metrics
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering   # 聚类模型
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.sparse import coo_matrix, csr_matrix
import config

def cluster_training(dataset, cluster_number):
    # # 1.1 从pkl文件中读取数据
    # '''注意文件的读取路径！！！！！'''
    # # root_path = "../saved_exp_info/fedlab_save/num_cnt/cifar10_balanced_dirichlet_num_cnt.pkl"
    # root_path = f"{config.ROOT_PATH}saved_exp_info/data_partition_result/CIFAR10_bbal_0.01.pkl"
    print(f"clustering dataset:{dataset}")
    root_path = f"{config.ROOT_PATH}saved_exp_info/data_partition_result/"
    # if dataset[:5] == "CIFAR":
    #     datafile_name = "CIFAR10_bbal_10.pkl"
    # if dataset[:11] == "MNIST_shard":
    #     datafile_name = "MNIST_shard.pkl"
    # elif dataset[:10] == "MNIST_bbal":
    #     datafile_name = "MNIST_bbal_10.pkl"
    # elif dataset[:9] == "MNIST_iid":
    #     datafile_name = "MNIST_iid.pkl"
    
    datafile_name = dataset + ".pkl"

    root_path += datafile_name

    print("@@@ cluster开始读取data_partition_result文件：", root_path, " @@@")
    m_data = []
    data = []
    with open(root_path, 'rb') as f:
        while True:
            try:
                row_data = pickle.load(f)
                for m in row_data:
                    m_data.append(m)
            except EOFError:
                break
    print(len(m_data))

    # 1.2 对数据进行零均值标准化处理
    for d in m_data:
        da = []
        avg = np.mean(d)  # 求一组数据的均值
        std = np.std(d, ddof=1)  #  求一组数据的标准差
        for i in d:
            da.append((i-avg)/std)
        data.append(da)
    data = np.array(data)
    #
    # 1.3 用主成分分析法对传入的数据进行降维
    # np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28).astype(float)
    # data = np.array(data).reshape(-1, 1)
    pca = PCA(n_components=2)   # 输出降到几维
    data = pca.fit_transform(data)  # 载入N维
    #
    #
    # 2 三种聚类问题
    # 2.1 原型聚类：KMeans模型
    '''注意聚类的簇数！！！！！'''
    cluster_dict = {}
    if cluster_number:
        start_range = 2
        end_range = 11
    else:
        start_range = cluster_number
        end_range = cluster_number+1
    for i in range(start_range, end_range):  # 设置聚类簇数为10个
        model = KMeans(n_clusters=i)
        model.fit(data)  # 完成聚类
        pred_y = model.predict(data)  # 预测点在哪个聚类中
        print("当前聚类簇数为：", i)  # 输出每个样本的聚类标签
        print("聚类结果如下：")
        pred_y= list(pred_y)
        result = []
        # 输出聚类后的下标存放在result中
        for num in range(i):
            one_type = []
            for index, value in enumerate(pred_y):
                if value==num:
                    one_type.append(index)
            result.append(one_type)
        print(result)
        save_path = f'{config.ROOT_PATH}saved_exp_info/cluster_result/{datafile_name}.pkl'
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as output:
            pickle.dump(result, output)
        # 打印轮廓系数
        s_score = metrics.silhouette_score(data, pred_y, sample_size=len(data), metric='euclidean')
        print("当前聚类簇数为：", i, " 轮廓系数为：", s_score, "\n")
        cluster_dict[i] = s_score
        # # 打印CH分数
        # print("当前聚类簇数为：", i, " CH分数：", metrics.calinski_harabasz_score(data, pred_y))
        # # 打印戴维森系数
        # print("当前聚类簇数为：", i, " 戴维森系数：", metrics.davies_bouldin_score(data, pred_y))
        # for le in result:
        #     print(len(le))
    #     # # 绘制聚类效果图(3D版本)
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(111, projection='3d')
    #     # ax.scatter(data[:,0],data[:,1],data[:,2],c=pred_y,marker='.',s=20)
    #     # plt.title("k-Means")
    #     # plt.show()
    #     # 绘制聚类效果图(俯视图类型的)
    #     plt.scatter(data[:,0],data[:,1],c=pred_y,marker='o',s=0.1)
    #     plt.title("k-Means")
    #     plt.savefig('tupian.svg', format='svg', dpi=500)
    #     plt.show()
    #
    best_cluster_num = max(cluster_dict, key=cluster_dict.get)
    print(best_cluster_num, cluster_dict[best_cluster_num])

    return save_path
    #
    #
    # # # 2.2 密度聚类：DBSCAN模型
    # # model = DBSCAN(eps=0.5, min_samples=5)
    # # pred_y = model.fit_predict(data)  # 预测点在哪个聚类中
    # # pred_y= list(pred_y)
    # # print(pred_y)
    # # total = max(pred_y)+1
    # # result = []
    # # print("当前聚类簇数为：", total)  # 输出每个样本的聚类标签
    # # print("聚类结果如下：")
    # # # 输出聚类后的下标存放在result中
    # # for num in range(total):
    # #     one_type = []
    # #     for index, value in enumerate(pred_y):
    # #         if value==num:
    # #             one_type.append(index)
    # #     result.append(one_type)
    # # print(result)
    # # # 打印轮廓系数
    # # print("当前聚类簇数为：", total, " 轮廓系数为：", metrics.silhouette_score(data, pred_y, sample_size=len(data), metric='euclidean'))
    # # # 打印CH分数
    # # print("当前聚类簇数为：", total, " CH分数：", metrics.calinski_harabasz_score(data, pred_y))
    # # # 打印戴维森系数
    # # print("当前聚类簇数为：", total, " 戴维森系数：", metrics.davies_bouldin_score(data, pred_y))
    # # # 绘制聚类效果图(3D版本)
    # # fig = plt.figure()
    # # ax = fig.add_subplot(111, projection='3d')
    # # ax.scatter(data[:,0],data[:,1],data[:,2],c=pred_y)
    # # plt.title("DBSCAN")
    # # plt.show()
    #
    #
    # # # 2.3 层次聚类：凝聚模型
    # # for i in range(2, 6):
    # #     model = AgglomerativeClustering(n_clusters=i)
    # #     pred_y = model.fit_predict(data)  # 预测点在哪个聚类中
    # #     pred_y= list(pred_y)
    # #     result = []
    # #     print("当前聚类簇数为：", i)  # 输出每个样本的聚类标签
    # #     print("聚类结果如下：")
    # #     # 输出聚类后的下标存放在result中
    # #     for num in range(i):
    # #         one_type = []
    # #         for index, value in enumerate(pred_y):
    # #             if value==num:
    # #                 one_type.append(index)
    # #         result.append(one_type)
    # #     print(result)
    # #     # 打印轮廓系数
    # #     print("当前聚类簇数为：", i, " 轮廓系数为：", metrics.silhouette_score(data, pred_y, sample_size=len(data), metric='euclidean'))
    # #     # 打印CH分数
    # #     print("当前聚类簇数为：", i, " CH分数：", metrics.calinski_harabasz_score(data, pred_y))
    # #     # 打印戴维森系数
    # #     print("当前聚类簇数为：", i, " 戴维森系数：", metrics.davies_bouldin_score(data, pred_y))
    # #     # # 绘制聚类效果图(俯视图类型的)
    # #     # plt.scatter(data[:,0],data[:,1],c=pred_y,marker='.')
    # #     # plt.title("AgglomerativeClustering")
    # #     # plt.show()
    # #     # 绘制聚类效果图(3D版本)
    # #     fig = plt.figure()
    # #     ax = fig.add_subplot(111, projection='3d')
    # #     ax.scatter(data[:,0],data[:,1],data[:,2],c=pred_y,s=20)
    # #     plt.title("AgglomerativeClustering")
    # #     plt.show()
    #
    # print("@ Finish clustering @")