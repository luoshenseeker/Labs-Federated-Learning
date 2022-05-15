import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import config
from data_partition.CIFAR10_dirichlet import get_CIFAR10_dataloaders

from saved_exp_info.pkl_dictionary import pkl_dict, not_final_pkl_dict, acc_pkl_dict

print_threshold = 70

print_str_list = []

def read_pkl_origin(filename):
    np.set_printoptions(threshold=np.inf)   # 解决显示不完全问题

    # filename = "YOUR WORK DIR/data/NIID-Bench-origin/saved_exp_info/acc/" + filename
    # print('!origin!')
    filename = "YOUR WORK DIR/saved_exp_info/acc/" + filename

    fr=open(filename,'rb')

    acc_hist = pickle.load(fr)

    server_acc = acc_hist

    first_reach = False
    # 新版需用
    server_acc = []
    # server_loss = np.dot(weights, loss_hist[i + 1])
    for i in range(len(acc_hist)):
        n_samples = np.array([200 for _ in range(100)])
        weights = n_samples / np.sum(n_samples)   # sample size为聚合权重
        if np.dot(weights, acc_hist[i]) != 0.0:
            now = np.dot(weights, acc_hist[i])
            if not first_reach and now >= print_threshold:
                print_str_list.append(filename.split("/")[-1])
                print_str_list.append(f"Reached threashold {print_threshold} at {i+1}")
                first_reach = True
                break
                # print(filename)
                # print(f"Reached threashold {print_threshold} at {i+1}")
            server_acc.append(now)
    if not first_reach:
        print_str_list.append(filename.split("/")[-1])
        print_str_list.append(f"Not reached threashold {print_threshold} xxxxx")

    return server_acc

def plot_acc_with_order(pkl_file: list,):

    n = len(pkl_file)
    for k in range(n):

        # if pkl_file[k][:1] == 'm' or pkl_file[k][:1] == 'f' or pkl_file[k][:1] == 'c':
        read_pkl_origin(pkl_file[k])
    for s in print_str_list:
        print(s)

if __name__ == "__main__":
    # plot_acc_with_order("Mnist shard conv", [
    #     "MNIST_shard_FedAvg_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv.pkl",
    #     "MNIST_shard_random_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv.pkl", 
    #     "MNIST_shard_clustered_1_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv.pkl",
    #     "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv.pkl",
    #     ])
    plot_acc_with_order([
            "FMNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_ncon_cn10.pkl",
            "FMNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_ncon_cn9.pkl",
            "FMNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_ncon_cn8.pkl",
            "FMNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_ncon_cn7.pkl",
            "FMNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_ncon_cn6.pkl",
            "FMNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_ncon_cn5.pkl",
            "FMNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv_cn10.pkl",
            "FMNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv_cn9.pkl",
            "FMNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv_cn8.pkl",
            "FMNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv_cn7.pkl",
            "FMNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv_cn6.pkl",
            "FMNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv_cn5.pkl",
            "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_ncon_cn10.pkl",
            "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_ncon_cn9.pkl",
            "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_ncon_cn8.pkl",
            "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_ncon_cn7.pkl",
            "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_ncon_cn6.pkl",
            "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_ncon_cn5.pkl",
            "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv_cn10.pkl",
            "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv_cn9.pkl",
            "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv_cn8.pkl",
            "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv_cn7.pkl",
            "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv_cn6.pkl",
            "MNIST_shard_ours_any_i200_N50_lr0.01_B50_d1.0_p0.1_m1_0_AVG_conv_cn5.pkl",
        ]
    )

    # 0.5 touch 60: 
