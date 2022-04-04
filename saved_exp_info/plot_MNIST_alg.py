import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import config
from data_partition.CIFAR10_dirichlet import get_CIFAR10_dataloaders

from saved_exp_info.pkl_dictionary import pkl_dict


def read_pkl(filename):
    np.set_printoptions(threshold=np.inf)   # 解决显示不完全问题
    fr = open("./acc/" + filename, 'rb')

    hist = pickle.load(fr)

    server = []
    # server_loss = np.dot(weights, loss_hist[i + 1])
    for i in range(len(hist)):
        if filename[:5] == 'MNIST' or filename[:6] == 'fmnist' or filename[:6] == 'FMNIST':
            n_samples = np.array([600 for _ in range(100)])
        elif filename[:7] == 'CIFAR10':
            if filename[8:12] == 'bbal':
                n_samples = np.array([600 for _ in range(100)])
            elif filename[8:12] == 'nbal':
                n_samples = [100] * 10 + [250] * 30 + [500] * 30 + [750] * 20 + [1000] * 10
        weights = n_samples / np.sum(n_samples)   # sample size为聚合权重
        if np.dot(weights, hist[i]) != 0.0:
            server.append(np.dot(weights, hist[i]))
        # print(server_acc)
    return server

def plot_multi(rows, cols, file_lists, num):
    fig, axis = plt.subplots(rows, cols, figsize=(14, rows / cols * 12))  # 平分画板，调节fig的大小，黄金比例为4:3
    color1 = ['#ff851b','#3d6daa','#c04851']  # alg1色卡
    color2 = ['#ff851b','#806d9e','#c04851']  # md色卡
    color3 = ['#ff851b','#c04851']  # FedNova专属色卡
    index1 = [0, 1, 3]  # md下标
    index2 = [0, 2, 3]  # alg1下标
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
        if i != 3 and i != 4:
            i = i * cols
            for j in range(i, i + cols):
                ind_row = j // cols  # 确定当前图片存放的行号
                ind_col = j % cols  # 确定当前图片存放的列号
                plot_name = key
                y = {}
                multi = pd.DataFrame()
                # 画一幅图里多条曲线
                if j % 2 == 0:
                    file_list = [file_lists[key][x] for x in index1]
                    n = len(file_list)  # n表示一个图中包含的曲线数
                    colors = color2
                else:
                    file_list = [file_lists[key][x] for x in index2]
                    n = len(file_list)  # n表示一个图中包含的曲线数
                    colors = color1
                for k in range(n):
                    if j == 6 or j == 7:
                        y[k] = [file_list[k][y]*100 for y in range(len(file_list[k]))]
                        end = 145
                    else:
                        y[k] = read_pkl(file_list[k])
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
                axis[ind_row,ind_col].set_ylabel('Test Accuracy', {'size':15})
                axis[ind_row,ind_col].set_xlim([max(op,stp),end])  # 限制x轴的范围从有阴影和曲线开始
                axis[ind_row,ind_col].grid()
                axis[ind_row,ind_col].set_title(plot_name, {'size':18})
                if j == 0:
                    axis[ind_row,ind_col].legend(loc='lower right', labels=["random_sampling", "importance_sampling", "ours"])
                elif j == 1:
                    axis[ind_row,ind_col].legend(loc='lower right', labels=["random_sampling", "cluster_sampling", "ours"])
        else:
            if i == 3:
                ind_row = i * cols // cols  # 确定当前图片存放的行号
                ind_col = i * cols % cols  # 确定当前图片存放的列号
            elif i == 4:
                ind_row = (i * cols-1) // cols  # 确定当前图片存放的行号
                ind_col = (i * cols-1) % cols  # 确定当前图片存放的列号
            plot_name = key
            y = {}
            multi = pd.DataFrame()
            n = len(file_lists[key])  # n表示一个图中包含的曲线数
            file_list = file_lists[key]
            colors = color3
            for k in range(n):
                # y[k] = [file_list[k][y] * 100 for y in range(len(file_list[k]))]  # MNIST FedNova版本
                y[k] = read_pkl(file_list[k])
                multi[k] = y[k][start:end]
                meanop = multi[k].rolling(op).mean()  # 每10个算一个平均数，共200个平均数，前9个为0
                stdop = multi[k].rolling(op).std()  # 每10个算一个标准差，共200个方差，前9个为0
                meanop_5 = [meanop[i] for i in range(op - 1, len(meanop), stp)]  # 每10个值标一个点，共20个点
                axis[ind_row, ind_col].plot(range(op, end + 1, stp), meanop_5, color=colors[k])
                axis[ind_row, ind_col].fill_between(range(op, end + 1),
                                                    meanop[op - 1:] - 1.44 * stdop[op - 1:],
                                                    meanop[op - 1:] + 1.44 * stdop[op - 1:],
                                                    color=colors[k],
                                                    alpha=0.35)

            axis[ind_row, ind_col].set_xlabel('Communication Rounds', {'size': 15})
            axis[ind_row, ind_col].set_ylabel('Test Accuracy', {'size': 15})
            axis[ind_row, ind_col].set_xlim([max(op, stp), end])  # 限制x轴的范围从有阴影和曲线开始
            axis[ind_row, ind_col].grid()
            axis[ind_row, ind_col].set_title(plot_name, {'size': 18})

    plt.subplots_adjust(wspace=0.18, hspace=0.3)  # 调整子图之间的行列间距，w表示宽，h表示高
    # fig.tight_layout()  # 调整整体空白，自动调整子图位置
    plt.savefig(f'./plot_result/pic{num}.pdf', format='pdf', transparent=True, bbox_inches="tight")
    fig.show()

# pkl_file = {
#             'FMNIST iid on FedAvg': ["MNIST_iid_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
#                                     "MNIST_iid_random_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
#                                     "MNIST_iid_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
#                                     "MNIST_iid_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl"
#                                     ],
#             'MNIST Non-iid on FedAvg': ["MNIST_shard_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
#                                     "MNIST_shard_random_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
#                                     "MNIST_shard_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
#                                     "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl"
#                                     ],
#             'MNIST Non-iid on FedProx': ["MNIST_shard_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_mu0.01.pkl",
#                                     "MNIST_shard_random_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_mu0.01.pkl",
#                                     "MNIST_shard_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_mu0.01.pkl",
#                                     "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_mu0.01.pkl"
#                                     ],
#             r'MNIST Non-iid on FedNova  $\rho$=0.88': [
#                 [0.0982, 0.101, 0.0963, 0.098, 0.0982, 0.1032, 0.0982, 0.0974, 0.1009, 0.101, 0.0982, 0.1135, 0.0892, 0.1009,
#                 0.1028, 0.0974, 0.15, 0.1135, 0.101, 0.1028, 0.1032, 0.0958, 0.101, 0.1009, 0.1135, 0.0982, 0.1135, 0.0892,
#                 0.0892, 0.101, 0.0974, 0.1043, 0.0958, 0.098, 0.101, 0.1032, 0.0982, 0.0974, 0.1009, 0.1135, 0.1032, 0.1028,
#                 0.098, 0.101, 0.1135, 0.0958, 0.0974, 0.0982, 0.1135, 0.0958, 0.098, 0.101, 0.0958, 0.0982, 0.1028, 0.0958,
#                 0.101, 0.0974, 0.1032, 0.0982, 0.1009, 0.098, 0.0982, 0.0982, 0.1135, 0.098, 0.0974, 0.1032, 0.1651, 0.0974,
#                 0.0982, 0.101, 0.1135, 0.0958, 0.1028, 0.101, 0.1135, 0.1032, 0.0958, 0.1032, 0.1326, 0.1646, 0.2443, 0.1077,
#                 0.1711, 0.2415, 0.3304, 0.2326, 0.1028, 0.2127, 0.3244, 0.2091, 0.0898, 0.1581, 0.1484, 0.2871, 0.3423, 0.2707,
#                 0.3319, 0.1833, 0.2779, 0.3, 0.3723, 0.4433, 0.3864, 0.361, 0.4638, 0.2586, 0.2326, 0.4539, 0.3643, 0.2974,
#                 0.2179, 0.1734, 0.3336, 0.262, 0.2221, 0.428, 0.4566, 0.4831, 0.4302, 0.5827, 0.5972, 0.5414, 0.4516, 0.3702,
#                 0.5125, 0.4447, 0.6321, 0.6426, 0.7183, 0.6046, 0.5151, 0.6506, 0.7393, 0.7138, 0.6585, 0.71, 0.4719, 0.6688,
#                 0.6294, 0.7216, 0.7676, 0.6435, 0.7314, 0.7361, 0.7003, 0.5061, 0.6888, 0.7126, 0.6662, 0.796, 0.8535, 0.7997,
#                 0.8284, 0.8955, 0.8817, 0.8633, 0.6531, 0.6844, 0.726, 0.8775, 0.8922, 0.8993, 0.8791, 0.8705, 0.8738, 0.844,
#                 0.8675, 0.9074, 0.9122, 0.8079, 0.8615, 0.8454, 0.7498, 0.7169, 0.8157, 0.8568, 0.883, 0.8565, 0.8941, 0.8788,
#                 0.8372, 0.9088, 0.8834, 0.936, 0.8709, 0.9102, 0.8757, 0.8864, 0.9052, 0.8499, 0.7654, 0.8713, 0.8929, 0.8629,
#                 0.9166, 0.869, 0.8434, 0.9194],
#                 [0.1135, 0.1104, 0.098, 0.1779, 0.0997, 0.1025, 0.1157, 0.101, 0.1009, 0.103, 0.0892, 0.101, 0.1032, 0.1653,
#                 0.2004, 0.1028, 0.1636, 0.0982, 0.1135, 0.1934, 0.1434, 0.1106, 0.1032, 0.1156, 0.1026, 0.1985, 0.1014, 0.189,
#                 0.365, 0.3416, 0.1975, 0.2742, 0.2408, 0.4206, 0.3695, 0.239, 0.4025, 0.5104, 0.4517, 0.2777, 0.2604, 0.4766,
#                 0.615, 0.4273, 0.3613, 0.4604, 0.6577, 0.7316, 0.6647, 0.473, 0.6885, 0.7929, 0.8129, 0.7482, 0.6863, 0.6855,
#                 0.7643, 0.8249, 0.8442, 0.7821, 0.7487, 0.7057, 0.6485, 0.6772, 0.7072, 0.6535, 0.7091, 0.7994, 0.8888, 0.8911,
#                 0.8879, 0.8702, 0.8229, 0.733, 0.7329, 0.7335, 0.8958, 0.9014, 0.9121, 0.899, 0.8894, 0.8325, 0.8547, 0.8339,
#                 0.8942, 0.8881, 0.8712, 0.8715, 0.8304, 0.8329, 0.8426, 0.8826, 0.8939, 0.9085, 0.896, 0.9029, 0.9265, 0.916,
#                 0.9174, 0.8903, 0.8836, 0.8378, 0.8632, 0.9066, 0.9313, 0.9286, 0.9316, 0.9296, 0.9177, 0.8831, 0.7974, 0.8054,
#                 0.8672, 0.9372, 0.9398, 0.9334, 0.9121, 0.8585, 0.8227, 0.7739, 0.9009, 0.9157, 0.9392, 0.9419, 0.9431, 0.9425,
#                 0.9463, 0.9338, 0.9244, 0.921, 0.9, 0.9346, 0.9415, 0.9428, 0.9472, 0.9411, 0.9465, 0.9356, 0.9424, 0.9269,
#                 0.9225, 0.9088, 0.936, 0.9186, 0.9406, 0.9328]
#                                     ],
#             r'MNIST Non-iid on FedNova  $\rho$=0.2': [
#                 [0.0982, 0.101, 0.0963, 0.098, 0.0982, 0.1032, 0.0982, 0.0974, 0.1009, 0.101, 0.0982, 0.1135, 0.0892, 0.1009,
#                 0.1028, 0.0974, 0.15, 0.1135, 0.101, 0.1028, 0.1032, 0.0958, 0.101, 0.1009, 0.1135, 0.0982, 0.1135, 0.0892,
#                 0.0892, 0.101, 0.0974, 0.1043, 0.0958, 0.098, 0.101, 0.1032, 0.0982, 0.0974, 0.1009, 0.1135, 0.1032, 0.1028,
#                 0.098, 0.101, 0.1135, 0.0958, 0.0974, 0.0982, 0.1135, 0.0958, 0.098, 0.101, 0.0958, 0.0982, 0.1028, 0.0958,
#                 0.101, 0.0974, 0.1032, 0.0982, 0.1009, 0.098, 0.0982, 0.0982, 0.1135, 0.098, 0.0974, 0.1032, 0.1651, 0.0974,
#                 0.0982, 0.101, 0.1135, 0.0958, 0.1028, 0.101, 0.1135, 0.1032, 0.0958, 0.1032, 0.1326, 0.1646, 0.2443, 0.1077,
#                 0.1711, 0.2415, 0.3304, 0.2326, 0.1028, 0.2127, 0.3244, 0.2091, 0.0898, 0.1581, 0.1484, 0.2871, 0.3423, 0.2707,
#                 0.3319, 0.1833, 0.2779, 0.3, 0.3723, 0.4433, 0.3864, 0.361, 0.4638, 0.2586, 0.2326, 0.4539, 0.3643, 0.2974,
#                 0.2179, 0.1734, 0.3336, 0.262, 0.2221, 0.428, 0.4566, 0.4831, 0.4302, 0.5827, 0.5972, 0.5414, 0.4516, 0.3702,
#                 0.5125, 0.4447, 0.6321, 0.6426, 0.7183, 0.6046, 0.5151, 0.6506, 0.7393, 0.7138, 0.6585, 0.71, 0.4719, 0.6688,
#                 0.6294, 0.7216, 0.7676, 0.6435, 0.7314, 0.7361, 0.7003, 0.5061, 0.6888, 0.7126, 0.6662, 0.796, 0.8535, 0.7997,
#                 0.8284, 0.8955, 0.8817, 0.8633, 0.6531, 0.6844, 0.726, 0.8775, 0.8922, 0.8993, 0.8791, 0.8705, 0.8738, 0.844,
#                 0.8675, 0.9074, 0.9122, 0.8079, 0.8615, 0.8454, 0.7498, 0.7169, 0.8157, 0.8568, 0.883, 0.8565, 0.8941, 0.8788,
#                 0.8372, 0.9088, 0.8834, 0.936, 0.8709, 0.9102, 0.8757, 0.8864, 0.9052, 0.8499, 0.7654, 0.8713, 0.8929, 0.8629,
#                 0.9166, 0.869, 0.8434, 0.9194],
#                 [0.1135, 0.1104, 0.098, 0.1779, 0.0997, 0.1025, 0.1157, 0.101, 0.1009, 0.103, 0.0892, 0.101, 0.1032, 0.1653,
#                 0.2004, 0.1028, 0.1636, 0.0982, 0.1135, 0.1934, 0.1434, 0.1106, 0.1032, 0.1156, 0.1026, 0.1985, 0.1014, 0.189,
#                 0.365, 0.3416, 0.1975, 0.2742, 0.2408, 0.4206, 0.3695, 0.239, 0.4025, 0.5104, 0.4517, 0.2777, 0.2604, 0.4766,
#                 0.615, 0.4273, 0.3613, 0.4604, 0.6577, 0.7316, 0.6647, 0.473, 0.6885, 0.7929, 0.8129, 0.7482, 0.6863, 0.6855,
#                 0.7643, 0.8249, 0.8442, 0.7821, 0.7487, 0.7057, 0.6485, 0.6772, 0.7072, 0.6535, 0.7091, 0.7994, 0.8888, 0.8911,
#                 0.8879, 0.8702, 0.8229, 0.733, 0.7329, 0.7335, 0.8958, 0.9014, 0.9121, 0.899, 0.8894, 0.8325, 0.8547, 0.8339,
#                 0.8942, 0.8881, 0.8712, 0.8715, 0.8304, 0.8329, 0.8426, 0.8826, 0.8939, 0.9085, 0.896, 0.9029, 0.9265, 0.916,
#                 0.9174, 0.8903, 0.8836, 0.8378, 0.8632, 0.9066, 0.9313, 0.9286, 0.9316, 0.9296, 0.9177, 0.8831, 0.7974, 0.8054,
#                 0.8672, 0.9372, 0.9398, 0.9334, 0.9121, 0.8585, 0.8227, 0.7739, 0.9009, 0.9157, 0.9392, 0.9419, 0.9431, 0.9425,
#                 0.9463, 0.9338, 0.9244, 0.921, 0.9, 0.9346, 0.9415, 0.9428, 0.9472, 0.9411, 0.9465, 0.9356, 0.9424, 0.9269,
#                 0.9225, 0.9088, 0.936, 0.9186, 0.9406, 0.9328]
#                                     ]
#             }
pkl_file = {
            'FMNIST iid on FedAvg': ["FMNIST_iid_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
                                    "FMNIST_iid_random_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
                                    "FMNIST_iid_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
                                    "FMNIST_iid_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl"
                                    ],
            'FMNIST Non-iid on FedAvg': ["FMNIST_shard_FedAvg_any_i300_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
                                    "FMNIST_shard_random_any_i300_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
                                    "FMNIST_shard_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
                                    "FMNIST_shard_ours_any_i300_N50_lr0.01_B50_d1.0_p0.1_m1_0.pkl",
                                    ],
            'FMNIST Non-iid on FedProx': ["FMNIST_shard_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_mu1.0.pkl",    # 72.308 ±0.894
                                    "FMNIST_shard_random_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_mu0.01.pkl",
                                    "FMNIST_shard_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_mu0.01.pkl",
                                    "FMNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_mu0.01.pkl",
                                    ],
            r'FMNIST Non-iid on FedNova  $\rho$=0.88': ["fmnist_noniid-labeldir_0.0_0.88.pkl",
                                                        "fmnist_ours_noniid-labeldir_0.0_0.88.pkl"
                                    ],
            r'FMNIST Non-iid on FedNova  $\rho$=0.70': ["fmnist_noniid-labeldir_0.0_0.7.pkl",
                                                        "fmnist_ours_noniid-labeldir_0.0_0.7.pkl"
                                    ]
            }
plot_multi(4, 2, pkl_file, 4)
