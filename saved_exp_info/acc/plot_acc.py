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


exp_name = 'MNIST_shard'
# list_len = pickle.load(open(f"{config.ROOT_PATH}saved_exp_info/len_dbs/{exp_name}.pkl",'rb'))
# n_samples = np.array([len for len in list_len])  # 需改进为不依赖dataloaders
pkl_file = pkl_dict[exp_name]

if pkl_file[0][:5] == "MNIST" or pkl_file[0][:6] == "FMNIST":
    start = 0
    end = 200
    op = 5
    stp = 5
elif pkl_file[0][:5] == "CIFAR":
    start = 0
    end = 400
    op = 10
    stp = 10

y = {}
meanop = {}
meanop_5 = {}
# colors = ['#ff851b','#3d6daa','#c04851','#806d9e','#66a9c9']  # 黄 浅蓝 红 紫 深蓝 alg1
colors = ['#ff851b','#806d9e','#c04851','#3d6daa','#66a9c9']  # 黄 紫 红 浅蓝 深蓝 md

acc = pd.DataFrame()

n = len(pkl_file)
for k in range(n):

    y[k] = read_pkl(pkl_file[k])
    acc[k] = y[k][start:end]

    meanop[k]=acc[k].rolling(op).mean()  # 每10个算一个平均数，共200个平均数，前9个为0
    stdop1=acc[k].rolling(op).std()  # 每10个算一个标准差，共200个方差，前9个为0
    meanop_5[k] = [meanop[k][i] for i in range(stp-1,len(meanop[k]),stp)]  # 每10个值标一个点，共20个点
    plt.plot(range(stp, end + 1, stp), meanop_5[k], color=colors[k])
    plt.fill_between(range(start + 1, end + 1),
                     meanop[k] - 1.44 * stdop1,
                     meanop[k] + 1.44 * stdop1,
                     color=colors[k],
                     alpha=0.35)

if n == 5:
    plt.legend(labels=["FedAvg", "MD", "alg1", "alg2", "ours"])
elif n == 4:
    plt.legend(labels=["FedAvg", "alg1", "alg2", "ours"])
elif n == 3:
    # plt.legend(loc='lower right', labels=["random_sampling", "cluster_sampling", "ours"])
    plt.legend(loc='lower right',labels=["random_sampling", "importance_sampling", "ours"])

filename = exp_name+'_acc'
plt.xlim([max(op,stp),end])  #设置x轴显示的范围
plt.grid()
plt.xlabel('Communication Rounds', {'size':15})
plt.ylabel('Test Accuracy', {'size':15})
# plt.title('CIFAR10 '+r'$\alpha$'+'=0.01', {'size':18})
plt.title('MNIST Non-iid p=0.1', {'size':18})  # title的大小设置为18
# plt.savefig(f'../plot_result/{filename}_md.pdf', format='pdf', dpi=500, bbox_inches="tight")
plt.show()
