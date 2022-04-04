import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import config
from data_partition.CIFAR10_dirichlet import get_CIFAR10_dataloaders

from saved_exp_info.pkl_dictionary import pkl_dict


def read_pkl(filename, index):
    np.set_printoptions(threshold=np.inf)   # 解决显示不完全问题
    if index%2 == 0:
        fr = open("./acc/" + filename, 'rb')
    else:
        fr = open("./loss/" + filename, 'rb')

    hist = pickle.load(fr)

    server = []
    # server_loss = np.dot(weights, loss_hist[i + 1])
    for i in range(len(hist)):
        if filename[:5] == 'MNIST':
            n_samples = np.array([600 for _ in range(100)])
        elif filename[:7] == 'CIFAR10':
            n_samples = np.array([600 for _ in range(100)])
        weights = n_samples / np.sum(n_samples)   # sample size为聚合权重
        if np.dot(weights, hist[i]) != 0.0:
            server.append(np.dot(weights, hist[i]))
        # print(server_acc)
    return server

def plot_multi(rows, cols, file_lists, num):
    fig, axis = plt.subplots(rows, cols, figsize=(14, rows / cols * 12))  # 平分画板，调节fig的大小，黄金比例为4:3
    colors = ['#ff851b','#3d6daa','#c04851','#806d9e','#66a9c9']  # 黄 浅蓝 红 紫 深蓝 alg1
    # colors = ['#ff851b','#806d9e','#c04851','#3d6daa','#66a9c9']  # 黄 紫 红 浅蓝 深蓝 md
    # 拼接多个图
    for i,key in enumerate(file_lists):
        i = i * cols
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

        for j in range(i, i + cols):
            ind_row = j // cols
            ind_col = j % cols
            plot_name = key
            y = {}
            multi = pd.DataFrame()
            # 画一幅图里多条曲线
            n = len(file_lists[key])  # n表示一个图中包含的曲线数
            for k in range(n):
                y[k] = read_pkl(file_lists[key][k], j)
                multi[k] = y[k][start:end]
                meanop = multi[k].rolling(op).mean()  # 每10个算一个平均数，共200个平均数，前9个为0
                stdop = multi[k].rolling(op).std()  # 每10个算一个标准差，共200个方差，前9个为0
                meanop_5 = [meanop[i] for i in range(op-1,len(meanop),stp)]  # 每10个值标一个点，共20个点
                axis[ind_row,ind_col].plot(range(op, end + 1, stp), meanop_5, color=colors[k])
                axis[ind_row,ind_col].fill_between(range(op, end + 1),
                             meanop[op-1:] - 1.44 * stdop[op-1:],
                             meanop[op-1:] + 1.44 * stdop[op-1:],
                             color=colors[k],
                             alpha=0.35)
            axis[ind_row,ind_col].set_xlabel('Communication Rounds', {'size':15})
            if j%2 == 0:
                axis[ind_row,ind_col].set_ylabel('Test Accuracy', {'size':15})
            else:
                axis[ind_row, ind_col].set_ylabel('Train Loss', {'size': 15})
            axis[ind_row,ind_col].set_xlim([max(op,stp),end])  # 限制x轴的范围从有阴影和曲线开始
            axis[ind_row,ind_col].grid()
            axis[ind_row,ind_col].set_title(plot_name, {'size':18})
            if j == 0:
                # axis[ind_row,ind_col].legend(loc='lower right', labels=["random_sampling", "importance_sampling", "ours"])
                axis[ind_row,ind_col].legend(loc='lower right', labels=["random_sampling", "cluster_sampling", "ours"])

    plt.subplots_adjust(wspace=0.18, hspace=0.3)  # 调整子图之间的行列间距，w表示宽，h表示高
    # fig.tight_layout()  # 调整整体空白，自动调整子图位置
    plt.savefig(f'./plot_result/pic{num}.pdf', format='pdf', transparent=True, bbox_inches="tight")
    fig.show()

# pkl_file = {
#             'FedAvg with MNIST Non-iid': ["MNIST_shard_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
#                                     "MNIST_shard_random_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
#                                     # "MNIST_shard_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
#                                     # "MNIST_shard_clustered_2_cosine_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
#                                     "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl"
#                                     ],
#             'FedProx with MNIST Non-iid': ["MNIST_shard_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_mu0.01.pkl",
#                                     "MNIST_shard_random_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_mu0.01.pkl",
#                                     # "MNIST_shard_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_mu0.01.pkl",
#                                     "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_mu0.01.pkl"
#                                     ],
#             'FedNova with MNIST Non-iid': [
#                                     ],
#             'FedAvg with MNIST iid': ["MNIST_iid_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
#                                     "MNIST_iid_random_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
#                                     "MNIST_iid_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
#                                     "MNIST_iid_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl"
#                                     ]
#             }
pkl_file = {# FedAvg_CIFAR10_0.01
    r'CIFAR10 $\alpha$=0.01 on FedAvg': ["CIFAR10_bbal_0.01_FedAvg_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0.pkl0",
            # "CIFAR10_bbal_0.01_random_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0.pkl",
            "CIFAR10_bbal_0.01_clustered_1_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0.pkl",
            "CIFAR10_bbal_0.01_ours_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0.pkl",
            ],
            # FedAvg_CIFAR10_0.001
    r'CIFAR10 $\alpha$=0.001 on FedAvg': ["CIFAR10_bbal_0.001_FedAvg_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0.pkl",
            # "CIFAR10_bbal_0.001_random_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0.pkl",
            "CIFAR10_bbal_0.001_clustered_1_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0.pkl",
            "CIFAR10_bbal_0.001_ours_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0.pkl"
            ],
            # FedAvg_CIFAR10_shard
    'CIFAR10 Shard on FedAvg': ["CIFAR10_shard_FedAvg_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0.pkl",
            # "CIFAR10_shard_random_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0.pkl",
            "CIFAR10_shard_clustered_1_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0.pkl",
            "CIFAR10_shard_ours_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0.pkl"
            ],
            # FedProx_CIFAR10_0.01
    r'CIFAR10 $\alpha$=0.01 on FedPorx': [
            "CIFAR10_bbal_0.01_FedAvg_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0_mu0.05.pkl",
            # "CIFAR10_bbal_0.01_random_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0_mu0.05.pkl",
            "CIFAR10_bbal_0.01_clustered_1_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0_mu0.05.pkl",
            "CIFAR10_bbal_0.01_ours_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0_mu0.005.pkl",  # √
            ],
            # FedProx_CIFAR10_0.001
    r'CIFAR10 $\alpha$=0.001 on FedProx': ["CIFAR10_bbal_0.001_FedAvg_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0_mu0.05.pkl",
            # "CIFAR10_bbal_0.001_random_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0_mu0.05.pkl",
            "CIFAR10_bbal_0.001_clustered_1_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0_mu0.05.pkl",
            "CIFAR10_bbal_0.001_ours_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0_mu0.1.pkl"
            ],
            # FedProx_CIFAR10_shard
    'CIFAR10 Shard on FedProx': [
            "CIFAR10_shard_FedAvg_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0_mu0.015.pkl",  # √
            # "CIFAR10_shard_random_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0_mu0.05.pkl",
            "CIFAR10_shard_clustered_1_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0_mu0.05.pkl",
            "CIFAR10_shard_ours_any_i800_N80_lr0.05_B50_d1.0_p0.1_m1_0_mu0.015.pkl",  # √
            ],
            }

plot_multi(6, 2, pkl_file, 6)

