import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import random

data_name = 'CIFAR10_bbal_0.1.pkl'
raw_file = pkl.load(open(data_name, 'rb'))
index = random.sample(range(0, len(raw_file)), 10)  # 随机选10个client
file = [raw_file[ind] for ind in index]
x = np.array([cl for cl in range(1,11)])
lx=['1','2','3','4','5','6','7','8','9','10']
# 画dir分布图
with open('partition_plot.txt', 'w') as f:
    for i in range(len(file)):
        plt.figure(figsize=(4, 1)) # 调整输出图片的尺寸
        y = np.array(file[i])
        plt.plot(x, y, 'r-', linewidth=1)
        plt.bar(x, y, alpha=0.3, color='b', width=0.5)
        # plt.xlabel('Class')
        # plt.ylabel('Number')
        # plt.title('Distribution of client'+str(i+1))
        f.writelines([f"k={i} ", f" & {file[i][0]}", f" & {file[i][1]}", f" & {file[i][2]}", f" & {file[i][3]}",
                      f" & {file[i][4]}", f" & {file[i][5]}", f" & {file[i][6]}", f" & {file[i][7]}",
                      f" & {file[i][8]}", f" & {file[i][9]}\n"])
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f"client{i}_dir_partition.pdf", format='pdf', dpi=500, bbox_inches="tight", transparent=True)
        plt.show()

# # 画分布图10 clients
# def clients_10(file_name=0):
#     category_names = ['0', '1', '2', '3', '4', '5','6', '7','8', '9']
#     results = {
#         'client 1': [10, 15, 17, 32, 26, 100],
#         'client 2': [26, 22, 29, 10, 13, 100],
#         'client 3': [35, 37, 7, 2, 19, 100],
#         'client 4': [32, 11, 9, 15, 33, 100],
#         'client 5': [21, 29, 5, 5, 40, 100],
#         'client 6': [8, 19, 5, 30, 38, 100]
#     }
#
#
#     def survey(results, category_names):
#         """
#         Parameters
#         ----------
#         results : dict
#             A mapping from question labels to a list of answers per category.
#             It is assumed all lists contain the same number of entries and that
#             it matches the length of *category_names*.
#         category_names : list of str
#             The category labels.
#         """
#         labels = list(results.keys())
#         data = np.array(list(results.values()))
#         data_cum = data.cumsum(axis=1)
#         category_colors = plt.get_cmap('rainbow')(
#         np.linspace(0.15, 0.85, data.shape[1]))
#
#         fig, ax = plt.subplots(figsize=(9.2, 5))
#         ax.invert_yaxis()
#         ax.xaxis.set_visible(False)
#         ax.set_xlim(0, np.sum(data, axis=1).max())
#
#         for i, (colname, color) in enumerate(zip(category_names, category_colors)):
#             widths = data[:, i]
#             starts = data_cum[:, i] - widths
#             rects = ax.barh(labels, widths, left=starts, height=0.5,
#                             label=colname, color=color)
#
#             r, g, b, _ = color
#             text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
#             ax.bar_label(rects, label_type='center', color=text_color)
#         ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
#                 loc='lower left', fontsize='small')
#
#         return fig, ax
#
#
#     survey(results, category_names)
#     plt.show()
#
# # 画直方图和核密度图
# def all_samples(file_name):
#     data = pd.DataFrame()
#     f = open(file_name, 'rb')
#     samples = pkl.load(f)
#     for s in samples:
#         for i in s:
#             data['sum_samples'].append(i)
#     print()
#     data.sum_samples.plot(kind='hist', bins=20, color='steelblue', edgecolor='black',
#                           density=True, stacked=True, label='hist')
#     data.sum_samples.plot(kind='kde', color='red', label='density')
#     plt.xlabel('num of samples')
#     plt.ylabel('density')
#     plt.legend()
#     plt.show()
#
# clients_10()
# # file = "CIFAR10_bbal_0.01.pkl"
# # all_samples(file)

