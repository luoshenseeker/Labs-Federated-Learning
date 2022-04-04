import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import config
from data_partition.CIFAR10_dirichlet import get_CIFAR10_dataloaders

from saved_exp_info.pkl_dictionary import pkl_dict


def read_pkl(filename):
    np.set_printoptions(threshold=np.inf)   # 解决显示不完全问题

    fr=open(filename,'rb')

    acc_hist = pickle.load(fr)

    server_acc = []
    # server_loss = np.dot(weights, loss_hist[i + 1])
    for i in range(len(acc_hist)):
        if filename[:5] == 'MNIST':
            n_samples = np.array([600 for _ in range(100)])
        elif filename[:7] == 'CIFAR10':
            if filename[8:12] == 'bbal':
                n_samples = np.array([600 for _ in range(100)])
            elif filename[8:12] == 'nbal':
                n_samples = [100] * 10 + [250] * 30 + [500] * 30 + [750] * 20 + [1000] * 10
        weights = n_samples / np.sum(n_samples)   # sample size为聚合权重
        if np.dot(weights, acc_hist[i]) != 0.0:
            server_acc.append(np.dot(weights, acc_hist[i]))
        # print(server_acc)
    return server_acc


def plot_acc(rows, cols, file_lists, num):
    fig, axis = plt.subplots(rows, cols, figsize=(12, rows / cols * 12))  # 平分画板，调节fig的大小
    # colors = ['#ff851b','#3d6daa','#c04851','#806d9e','#66a9c9']  # 黄 浅蓝 红 紫 深蓝 alg1
    colors = ['#ff851b','#806d9e','#c04851','#3d6daa','#66a9c9']  # 黄 紫 红 浅蓝 深蓝 md
    # 拼接多个图
    for i,key in enumerate(file_lists):
        if key[:5] == "MNIST" or key[:6] == "FMNIST":
            start = 0
            end = 200
            op = 5
            stp = 5
        elif key[:5] == "CIFAR":
            start = 0
            end = 400
            op = 10
            stp = 10

        ind_row = i // cols
        ind_col = i % cols
        plot_name = key
        y = {}
        meanop = {}
        meanop_5 = {}
        acc = pd.DataFrame()
        # 画一幅图里多条曲线
        n = len(file_lists[key])
        for k in range(n):
            y[k] = read_pkl(file_lists[key][k])
            acc[k] = y[k][start:end]
            meanop = acc[k].rolling(op).mean()  # 每10个算一个平均数，共200个平均数，前9个为0
            stdop = acc[k].rolling(op).std()  # 每10个算一个标准差，共200个方差，前9个为0
            meanop_5 = [meanop[i] for i in range(op-1,len(meanop),stp)]  # 每10个值标一个点，共20个点
            axis[i//cols,i%cols].plot(range(op, end + 1, stp), meanop_5, color=colors[k])
            axis[i//cols,i%cols].fill_between(range(op, end + 1),
                             meanop[op-1:] - 1.44 * stdop[op-1:],
                             meanop[op-1:] + 1.44 * stdop[op-1:],
                             color=colors[k],
                             alpha=0.35)
        axis[i//cols,i%cols].set_xlabel('Communication Rounds', {'size':15})
        axis[i//cols,i%cols].set_ylabel('Test Accuracy', {'size':15})
        axis[i//cols,i%cols].set_xlim([max(op,stp),end])  # 限制x轴的范围从有阴影和曲线开始
        axis[i//cols,i%cols].grid()
        axis[i//cols,i%cols].set_title(plot_name, {'size':18})
        axis[i//cols,i%cols].legend(loc='lower right', labels=["random_sampling", "cluster_sampling", "ours"])

    fig.show()
    # plt.savefig(f'../plot_result/pic{num}.pdf', format='pdf', dpi=200, transparent=True, bbox_inches="tight")


pkl_file = {'MNIST Non-iid p=1': ["MNIST_shard_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p1.0_m1_0.pkl",
                                  # "MNIST_shard_random_any_i200_N50_lr0.01_B50_d1.0_p1.0_m1_0.pkl",
                                  "MNIST_shard_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p1.0_m1_0.pkl",
                                  "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p1.0_m1_0.pkl"
                                  ],
            'MNIST Non-iid p=0.5': ["MNIST_shard_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.5_m1_0.pkl",
                                    "MNIST_shard_random_any_i200_N50_lr0.01_B50_d1.0_p0.5_m1_0.pkl",
                                    # "MNIST_shard_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.5_m1_0.pkl",
                                    # "MNIST_shard_clustered_2_cosine_i200_N50_lr0.01_B50_d1.0_p0.5_m1_0.pkl",
                                    "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.5_m1_0.pkl"
                                    ],
            'MNIST Non-iid p=0.3': ["MNIST_shard_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.3_m1_0.pkl",
                                    "MNIST_shard_random_any_i200_N50_lr0.01_B50_d1.0_p0.3_m1_0.pkl",
                                    # "MNIST_shard_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.3_m1_0.pkl",
                                    # "MNIST_shard_clustered_2_cosine_i200_N50_lr0.01_B50_d1.0_p0.3_m1_0.pkl",
                                    "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.3_m1_0.pkl"
                                    ],
            'MNIST Non-iid p=0.2': ["MNIST_shard_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.2_m1_0.pkl",
                                    "MNIST_shard_random_any_i200_N50_lr0.01_B50_d1.0_p0.2_m1_0.pkl",
                                    # "MNIST_shard_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.2_m1_0.pkl",
                                    # "MNIST_shard_clustered_2_cosine_i200_N50_lr0.01_B50_d1.0_p0.2_m1_0.pkl",
                                    "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.2_m1_0.pkl"
                                    ],
            'MNIST Non-iid p=0.1': ["MNIST_shard_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
                                    # "MNIST_shard_random_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
                                    "MNIST_shard_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
                                    # "MNIST_shard_clustered_2_cosine_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
                                    "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl"
                                    ],
            }

plot_acc(5, 2, pkl_file, 1)

