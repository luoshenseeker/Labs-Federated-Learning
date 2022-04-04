import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import config
from data_partition.CIFAR10_dirichlet import get_CIFAR10_dataloaders

from saved_exp_info.pkl_dictionary import pkl_dict

# plot_class三位编码：第一位表示算法:md/agl1，第二位表示画图类别:acc/loss
def plot_one(ax, key, file_lists, plot_class):
    color1 = ['#ff851b', '#3d6daa', '#c04851']  # alg1色卡
    color2 = ['#ff851b', '#806d9e', '#c04851']  # md色卡
    color3 = ['#ff851b', '#c04851']  # FedNova专属色卡
    index1 = [0, 1, 3]  # md下标
    index2 = [0, 2, 3]  # alg1下标
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

    plot_name = key
    y = {}
    multi = pd.DataFrame()
    # 画一幅图里多条曲线
    if plot_class[0] == '0':
        file_lists = [file_lists[i] for i in index1]
        colors = color1
    elif plot_class[0] == '1':
        file_lists = [file_lists[i] for i in index2]
        colors = color2
    else:
        colors = color3
    n = len(file_lists)  # n表示一个图中包含的曲线数
    for k in range(n):
        if plot_class[1] == '0':
            fr = open("./acc/" + file_lists[k], 'rb')
        else:
            fr = open("./loss/" + file_lists[k], 'rb')
        line = pickle.load(fr)  # 从文件中读取一条线的数据
        y[k] = line[start:end]
        multi[k] = y[k][start:end]
        meanop = multi[k].rolling(op).mean()  # 每10个算一个平均数，共200个平均数，前9个为0
        stdop = multi[k].rolling(op).std()  # 每10个算一个标准差，共200个方差，前9个为0
        meanop_5 = [meanop[i] for i in range(op - 1, len(meanop), stp)]  # 每10个值标一个点，共20个点
        ax.plot(range(op, end + 1, stp), meanop_5, color=colors[k])
        ax.fill_between(range(op, end + 1),
                                            meanop[op - 1:] - 1.44 * stdop[op - 1:],
                                            meanop[op - 1:] + 1.44 * stdop[op - 1:],
                                            color=colors[k],
                                            alpha=0.35)
    ax.set_xlabel('Communication Rounds', {'size': 15})
    if plot_class[1] == '0':
        ax.set_ylabel('Test Accuracy', {'size': 15})
    else:
        ax.set_ylabel('Train Loss', {'size': 15})
    ax.set_xlim([max(op, stp), end])  # 限制x轴的范围从有阴影和曲线开始
    ax.grid()
    ax.set_title(plot_name, {'size': 18})

# pkl_dic要专门整一版以键作为文件名的
def plot_multi(rows, cols, pkl_dic, pkl_lists, mark_lists, num):
    fig, axis = plt.subplots(rows, cols, figsize=(14, rows / cols * 12))  # 平分画板，调节fig的大小，黄金比例为4:3
    for i in range(rows*cols):
        ind_row = i // cols
        ind_col = i % cols
        key = pkl_lists[i]
        one_plot = pkl_dic[key]
        plot_one(axis[ind_row, ind_col], key, one_plot, mark_lists[i])
        if i == 0:
            axis[ind_row, ind_col].legend(loc='lower right', labels=["random_sampling", "importance_sampling", "ours"])

    plt.subplots_adjust(wspace=0.18, hspace=0.3)  # 调整子图之间的行列间距，w表示宽，h表示高
    # fig.tight_layout()  # 调整整体空白，自动调整子图位置
    plt.savefig(f'./plot_result/pic{num}.pdf', format='pdf', transparent=True, bbox_inches="tight")
    fig.show()

