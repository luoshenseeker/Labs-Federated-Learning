#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import config

# from data_partition.fedlab_func.utils.dataset import functional as F

"""
-------
CIFAR 10
Dirichilet distribution

Partition the CIFAR-10 dataset using a Dirichlet distribution with alpha = {0.001, 0.01, 0.1, 10}
通过get_num_cnt()方法可以得到数字0-9在100个clients上的分布，形式为list(100,10)
----
"""

# dataset下载目录
DATASET_FOLDER = config.ROOT_PATH + "data/CIFAR10/"


class CIFARDataset(Dataset):
    """Convert the CIFAR CIFAR10 file into a Pytorch Dataset"""

    def __init__(self, file_path: str, k: int):
        dataset = pickle.load(open(file_path, "rb"))

        self.X = dataset[0][k]
        self.y = np.array(dataset[1][k])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        # 3D input 32x32x3
        x = torch.Tensor(self.X[idx]).permute(2, 0, 1) / 255
        x = (x - 0.5) / 0.5
        y = self.y[idx]

        return x, y


class CIFARShardDataset(Dataset):
    """Convert the MNIST pkl file into a Pytorch Dataset"""

    def __init__(self, file_path, k):

        with open(file_path, "rb") as pickle_file:
            dataset = pickle.load(pickle_file)
            self.features = np.vstack(dataset[0][k])

            vector_labels = list()
            for idx, digit in enumerate(dataset[1][k]):
                vector_labels += [digit] * len(dataset[0][k][idx])

            self.labels = np.array(vector_labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        # 3D input 32x32x3
        x = torch.Tensor(self.features[idx]).permute(2, 0, 1) / 255
        x = (x - 0.5) / 0.5
        # y = self.labels[idx]
        y = torch.LongTensor([self.labels[idx]])[0]

        return x, y

def get_1shard(ds, row_0: int, digit: int, samples: int):
    """return an array from `ds` of `digit` starting of
    `row_0` in the indices of `ds`"""

    row = row_0

    shard = list()

    while len(shard) < samples:
        if ds.targets[row] == digit:
            shard.append(ds.data[row])
        row += 1

    return row, shard

def clients_set_CIFAR10_shard(file_name, n_clients, batch_size=100, shuffle=True):
    """Download for all the clients their respective dataset"""
    print(file_name)

    list_dl = list()
    for k in range(n_clients):
        dataset_object = CIFARShardDataset(file_name, k)
        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )
        list_dl.append(dataset_dl)

    return list_dl

def create_CIFAR10_ds_1shard_per_client(n_clients, samples_train, samples_test):
    try:
        CIFAR10_train = datasets.CIFAR10(
            root=DATASET_FOLDER,
            train=True,
            download=False,
            # transform=transforms.ToTensor(),
        )
    except:
            CIFAR10_train = datasets.CIFAR10(
            root=DATASET_FOLDER,
            train=True,
            download=True,
            # transform=transforms.ToTensor(),
        )

    try:
        CIFAR10_test = datasets.CIFAR10(
            root=DATASET_FOLDER,
            train=False,
            download=False,
            # transform=transforms.ToTensor(),
        )
    except:
            CIFAR10_test = datasets.CIFAR10(
            root=DATASET_FOLDER,
            train=False,
            download=True,
            # transform=transforms.ToTensor(),
        )
    shards_train, shards_test = [], []
    labels = []

    for i in range(10):
        row_train, row_test = 0, 0
        for j in range(10):
            row_train, shard_train = get_1shard(
                CIFAR10_train, row_train, i, samples_train
            )
            row_test, shard_test = get_1shard(
                CIFAR10_test, row_test, i, samples_test
            )

            shards_train.append([shard_train])
            shards_test.append([shard_test])

            labels += [[i]]

    X_train = np.array(shards_train)
    X_test = np.array(shards_test)

    y_train = labels
    y_test = y_train

    folder = DATASET_FOLDER
    train_path = f"CIFAR10_shard_train_{n_clients}_{samples_train}.pkl"
    with open(folder + train_path, "wb") as output:
        pickle.dump((X_train, y_train), output)

    test_path = f"CIFAR10_shard_test_{n_clients}_{samples_test}.pkl"
    with open(folder + test_path, "wb") as output:
        pickle.dump((X_test, y_test), output)


# 2
def partition_CIFAR10_dataset(
        dataset,
        file_name: str,
        balanced: bool,
        matrix,
        n_clients: int,
        n_classes: int,
        train: bool,
):
    """Partition dataset into `n_clients`.
    Each client i has matrix[k, i] of data of class k"""

    list_clients_X = [[] for i in range(n_clients)]
    list_clients_y = [[] for i in range(n_clients)]

    # if balanced:
    #     client_sample_nums = F.balance_split(n_clients, num_samples=len(dataset.targets))
    # elif not balanced and train:
    #     client_sample_nums = F.lognormal_unbalance_split(n_clients, num_samples=len(dataset.targets), unbalance_sgm=0.3)
    # elif not balanced and not train:
    #     client_sample_nums = (F.lognormal_unbalance_split(n_clients, num_samples=len(dataset.targets), unbalance_sgm=0.3)/5).astype(np.int)

    if balanced:
        client_sample_nums = [500] * n_clients
    elif not balanced and train:
        client_sample_nums = (
                [100] * 10 + [250] * 30 + [500] * 30 + [750] * 20 + [1000] * 10
        )
    elif not balanced and not train:
        client_sample_nums = [20] * 10 + [50] * 30 + [100] * 30 + [150] * 20 + [200] * 10
    # print("client_sample_nums:", client_sample_nums)

    list_idx = []
    # 按label分类为10*5000，存入list_idx
    for k in range(n_classes):
        idx_k = np.where(np.array(dataset.targets) == k)[0]
        list_idx += [idx_k]

    for idx_client, n_sample in enumerate(client_sample_nums):

        clients_idx_i = []
        client_samples = 0

        for k in range(n_classes):

            if k < 9:
                samples_digit = int(matrix[idx_client, k] * n_sample)  # index？
                # print(f"matrix[{idx_client}, {k}]:", matrix[idx_client, k])
                # print("n_sample:", n_sample)
                # print("samples_digit:", samples_digit)
            if k == 9:
                samples_digit = n_sample - client_samples
            client_samples += samples_digit

            clients_idx_i = np.concatenate(
                (clients_idx_i, np.random.choice(list_idx[k], samples_digit))
            )

        clients_idx_i = clients_idx_i.astype(int)

        for idx_sample in clients_idx_i:
            list_clients_X[idx_client] += [dataset.data[idx_sample]]
            list_clients_y[idx_client] += [dataset.targets[idx_sample]]

        list_clients_X[idx_client] = np.array(list_clients_X[idx_client])

    folder = DATASET_FOLDER
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y), output)


# 1
def create_CIFAR10_dirichlet(
        dataset_name: str,
        balanced: bool,
        alpha: float,
        n_clients: int,
        n_classes: int,
):
    """Create a CIFAR dataset partitioned according to a
    dirichilet distribution Dir(alpha)"""

    from numpy.random import dirichlet

    # shape ``(size, k)``
    matrix = dirichlet([alpha] * n_classes, size=n_clients)
    # if matrix.isnull().any():
    #     matrix.replace(np.nan, 0)  # 删除无穷大的值
    # print("alpha:", alpha)
    # print("matrix:", matrix)

    try:
        CIFAR10_train = datasets.CIFAR10(
            root=DATASET_FOLDER,
            train=True,
            download=False,
            transform=transforms.ToTensor(),
        )
    except:
            CIFAR10_train = datasets.CIFAR10(
            root=DATASET_FOLDER,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

    try:
        CIFAR10_test = datasets.CIFAR10(
            root=DATASET_FOLDER,
            train=False,
            download=False,
            transform=transforms.ToTensor(),
        )
    except:
            CIFAR10_test = datasets.CIFAR10(
            root=DATASET_FOLDER,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

    file_name_train = f"{dataset_name}_train_{n_clients}.pkl"
    partition_CIFAR10_dataset(
        CIFAR10_train,
        file_name_train,
        balanced,
        matrix,
        n_clients,
        n_classes,
        True,
    )

    file_name_test = f"{dataset_name}_test_{n_clients}.pkl"
    partition_CIFAR10_dataset(
        CIFAR10_test,
        file_name_test,
        balanced,
        matrix,
        n_clients,
        n_classes,
        False,
    )


# 3
def clients_set_CIFAR(
        file_name: str, n_clients: int, batch_size: int, shuffle=True
):
    """Download for all the clients their respective dataset

    Args:
        file_name ():
        n_clients ():
        batch_size ():
        shuffle ():

    Returns:

    """
    print(file_name)

    list_dl = list()

    for k in range(n_clients):
        dataset_object = CIFARDataset(file_name, k)

        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )

        list_dl.append(dataset_dl)

    return list_dl


# 0
def get_CIFAR10_dataloaders(dataset_name, batch_size: int, shuffle=True, reset_data=False):
    """得到训练集DataLoader对象列表

    Args:
        dataset_name (str): {dataset}_{balanced}_{alpha}, such as "CIFAR10_nbal_10.0"
        batch_size (int): default=50

    Returns:
        list_dls_train : list of DataLoader objects

    """
    folder = DATASET_FOLDER

    try:
        CIFAR10_train = datasets.CIFAR10(
            root=DATASET_FOLDER,
            train=True,
            download=False,
            transform=transforms.ToTensor(),
        )
    except:
            CIFAR10_train = datasets.CIFAR10(
            root=DATASET_FOLDER,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
    list_dls_train_full = torch.utils.data.DataLoader(CIFAR10_train)

    try:
        CIFAR10_test = datasets.CIFAR10(
            root=DATASET_FOLDER,
            train=False,
            download=False,
            transform=transforms.ToTensor(),
        )
    except:
            CIFAR10_test = datasets.CIFAR10(
            root=DATASET_FOLDER,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
    list_dls_test_full = torch.utils.data.DataLoader(CIFAR10_test)

    if dataset_name == "CIFAR10_iid":
        n_clients = 100
        samples_train, samples_test = 600, 100

        CIFAR10_train_split = torch.utils.data.random_split(
            CIFAR10_train, [samples_train] * n_clients
        )
        list_dls_train = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in CIFAR10_train_split
        ]

        CIFAR10_test_split = torch.utils.data.random_split(
            CIFAR10_test, [samples_test] * n_clients
        )
        list_dls_test = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in CIFAR10_test_split
        ]

    elif dataset_name == "CIFAR10_shard":
        n_clients = 100
        samples_train, samples_test = 500, 80

        file_name_train = f"CIFAR10_shard_train_{n_clients}_{samples_train}.pkl"
        path_train = DATASET_FOLDER + file_name_train

        file_name_test = f"CIFAR10_shard_test_{n_clients}_{samples_test}.pkl"
        path_test = DATASET_FOLDER + file_name_test

        if not os.path.isfile(path_train):
            create_CIFAR10_ds_1shard_per_client(
                n_clients, samples_train, samples_test
            )

        list_dls_train = clients_set_CIFAR10_shard(
            path_train, n_clients, batch_size=batch_size, shuffle=shuffle
        )

        list_dls_test = clients_set_CIFAR10_shard(
            path_test, n_clients, batch_size=batch_size, shuffle=shuffle
        )

    # dirichlet
    else:
        n_classes = 10
        n_clients = 100
        balanced = dataset_name[8:12] == "bbal"
        alpha = float(dataset_name[13:])
        # alpha = Decimal(dataset_name[13:]).quantize(Decimal('0.00000'))

        file_name_train = f"{dataset_name}_train_{n_clients}.pkl"
        path_train = folder + file_name_train

        file_name_test = f"{dataset_name}_test_{n_clients}.pkl"
        path_test = folder + file_name_test

        # 固定数据集
        if not os.path.isfile(path_train) or reset_data:
            print("⚠⚠⚠ creating new dataset alpha:", alpha, " ⚠⚠⚠")
            create_CIFAR10_dirichlet(
                dataset_name, balanced, alpha, n_clients, n_classes
            )

        list_dls_train = clients_set_CIFAR(
            path_train, n_clients, batch_size, True
        )

        list_dls_test = clients_set_CIFAR(
            path_test, n_clients, batch_size, True
        )

    # Save in a file the number of samples owned per client
    list_len = list()
    for dl in list_dls_train:
        list_len.append(len(dl.dataset))
    with open(f"{config.ROOT_PATH}saved_exp_info/len_dbs/{dataset_name}.pkl", "wb") as output:
        pickle.dump(list_len, output)

    return list_dls_train, list_dls_test, list_dls_train_full, list_dls_test_full


def get_num_cnt(dataset_name, list_dls_train):
    # labels: 100*[6, 8, 3, 6, ...]
    labels = []
    for dl in list_dls_train:
        labels_temp = []
        for data in dl:
            labels_temp += data[1].tolist()  # 合并在各batch中的标签
        labels.append(labels_temp)
    # print("labels:", labels)

    # dl_len = []
    # for dl in list_dls_train:
    #     dl_len.append(len(dl.dataset))
    # print("dl_len:", dl_len)

    # num_cnt: 100*[cnt0, cnt1, ..., cnt9]
    num_cnt = []
    for label_ in labels:  # label_: [6, 8, 3, 6, ...]
        cnt = []
        total = len(label_)
        for num in range(10):
            cnt.append(label_.count(num))
        num_cnt.append(cnt)

    with open(f"{config.ROOT_PATH}saved_exp_info/data_partition_result/{dataset_name}.pkl", "wb") as output:
        pickle.dump(num_cnt, output)
    print("Data partition result successfully saved!")

    # print num_cnt
    print("num_cnt table: ")
    num_cnt_table = pd.DataFrame(num_cnt, columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    # 完整print 100行
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(num_cnt_table)

#
# batch_size = 50
# list_dls_train, list_dls_test = get_CIFAR10_dataloaders("CIFAR10_bbal_0.01", 50, reset_data=True)
# print("len_dls", len(list_dls_train))
