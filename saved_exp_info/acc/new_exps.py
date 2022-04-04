import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import config
from data_partition.CIFAR10_dirichlet import get_CIFAR10_dataloaders

from saved_exp_info.pkl_dictionary import pkl_dict, not_final_pkl_dict, acc_pkl_dict

pkl_file = ['MNIST_bbal_0.001_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.9_m1_0.pkl', 'MNIST_bbal_10_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.01_m1_0.pkl']


def read_pkl(filename):
    np.set_printoptions(threshold=np.inf)   # 解决显示不完全问题

    fr=open(filename,'rb')

    acc_hist = pickle.load(fr)

    server_acc = []
    # server_loss = np.dot(weights, loss_hist[i + 1])
    for i in range(len(acc_hist)):
        n_samples = np.array([800 for _ in range(100)])
        weights = n_samples / np.sum(n_samples)   # sample size为聚合权重
        if np.dot(weights, acc_hist[i]) != 0.0:
            server_acc.append(np.dot(weights, acc_hist[i]))
        # print(server_acc)

    return server_acc

n = len(pkl_file)

y = {}
acc = pd.DataFrame()

for k in range(n):
    # if pkl_file[k][:1] == 'm' or pkl_file[k][:1] == 'f' or pkl_file[k][:1] == 'c':
    print(pkl_file[k])
    y[k] = read_pkl(pkl_file[k])
    # else:
    #     y[k] = read_pkl(pkl_file[k])
    # y[k] = pkl_file[k]
    # print(y[k])
    acc[k] = y[k][0:200]
    # print(round(acc[k], 2).tolist())

    last_10 = np.array(round(acc[k][200 - 9:200 + 1], 3))
    avg = round(sum(last_10) / len(last_10), 2)
    fluctuation = round((max(last_10) - min(last_10)) / 2, 3)

    # print(f"μ{pkl_file[k][-9:-4]}: {avg} ±{fluctuation}")
    print(f"μ{pkl_file[k][-9:-4]}: {avg}")
