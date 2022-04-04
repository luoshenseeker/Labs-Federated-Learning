#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gzip
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import config
from .fedlab.utils.dataset import functional as F

"""
-------
MNIST

通过get_num_cnt()方法可以得到数字0-9在100个clients上的分布，形式为list(100,10)
----
"""

# DATASET_NAME = 'MNIST_iid'
# assert DATASET_NAME in ['MNIST_iid', 'MNIST_shard', 'MNIST_bbal_0.01']
# dataset下载目录
DATASET_FOLDER = config.ROOT_PATH + "data/MNIST/"

batch_size = 50
shuffle = True


class MnistDataset(Dataset):
    """Convert the MNIST pkl file into a Pytorch Dataset"""

    def __init__(self, file_path, k):
        dataset = pickle.load(open(file_path, "rb"))

        self.X = dataset[0][k]
        self.Y = np.array(dataset[1][k])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 3D input 1x28x28
        x = torch.Tensor([self.X[idx]]) / 255
        y = torch.LongTensor([self.Y[idx]])[0]

        return x, y

class RawMnistDataset(Dataset):
    def __init__(self, folder, data_name, label_name,transform=None):
        (train_data, train_labels) = load_data(folder, data_name, label_name) # 其实也可以直接使用torch.load(),读取之后的结果为torch.Tensor形式
        # (train_data, train_labels) = torch.load(folder, data_name, label_name)
        self.train_data = train_data
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):

        img, target = self.train_data[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_data)


def load_data(data_folder, data_name, label_name):
    """RawMnistDataset function
    data_folder: 文件目录
    data_name： 数据文件名
    label_name：标签数据文件名
    """
    with gzip.open(os.path.join(data_folder,label_name), 'rb') as lbpath: # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder,data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return (x_train, y_train)


class MnistShardDataset(Dataset):
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

        # 3D input 1x28x28
        x = torch.Tensor([self.features[idx]]) / 255
        y = torch.LongTensor([self.labels[idx]])[0]

        return x, y

def get_1shard(ds, row_0: int, digit: int, samples: int):
    """return an array from `ds` of `digit` starting of
    `row_0` in the indices of `ds`"""

    row = row_0

    shard = list()

    while len(shard) < samples:
        if ds.train_labels[row] == digit:
            shard.append(ds.train_data[row].numpy())
        row += 1

    return row, shard

def clients_set_MNIST_shard(file_name, n_clients, batch_size=100, shuffle=True):
    """Download for all the clients their respective dataset"""
    print(file_name)

    list_dl = list()
    for k in range(n_clients):
        dataset_object = MnistShardDataset(file_name, k)
        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )
        list_dl.append(dataset_dl)

    return list_dl

def create_MNIST_ds_1shard_per_client(n_clients, samples_train, samples_test):

    MNIST_train = datasets.MNIST(root=DATASET_FOLDER, train=True, download=True)
    MNIST_test = datasets.MNIST(root=DATASET_FOLDER, train=False, download=True)

    shards_train, shards_test = [], []
    labels = []

    for i in range(10):
        row_train, row_test = 0, 0
        for j in range(10):
            row_train, shard_train = get_1shard(
                MNIST_train, row_train, i, samples_train
            )
            row_test, shard_test = get_1shard(
                MNIST_test, row_test, i, samples_test
            )

            shards_train.append([shard_train])
            shards_test.append([shard_test])

            labels += [[i]]

    X_train = np.array(shards_train)
    X_test = np.array(shards_test)

    y_train = labels
    y_test = y_train

    folder = DATASET_FOLDER
    train_path = f"MNIST_shard_train_{n_clients}_{samples_train}.pkl"
    with open(folder + train_path, "wb") as output:
        pickle.dump((X_train, y_train), output)

    test_path = f"MNIST_shard_test_{n_clients}_{samples_test}.pkl"
    with open(folder + test_path, "wb") as output:
        pickle.dump((X_test, y_test), output)

# 2
def partition_MNIST_dataset(
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

    if balanced:
        client_sample_nums = F.balance_split(n_clients, num_samples=len(dataset.train_labels))
    elif not balanced and train:
        client_sample_nums = F.lognormal_unbalance_split(n_clients, num_samples=len(dataset.train_labels), unbalance_sgm=0.4)
    elif not balanced and not train:
        client_sample_nums = (F.lognormal_unbalance_split(n_clients, num_samples=len(dataset.train_labels), unbalance_sgm=0.4)/6).astype(np.int)

    print("client_sample_nums:", client_sample_nums)
    # if balanced:
    #     client_sample_nums = [600] * n_clients
    # elif not balanced and train:
    #     client_sample_nums = (
    #         [60] * 15 + [300] * 20 + [600] * 30 + [900] * 25 + [1200] * 10
    #     )
    # elif not balanced and not train:
    #     client_sample_nums = [10] * 15 + [50] * 20 + [100] * 30 + [150] * 25 + [200] * 10
    # print("client_sample_nums:", client_sample_nums)

    list_idx = []
    # 按label分类为10*5000，存入list_idx
    for k in range(n_classes):
        idx_k = np.where(np.array(dataset.train_labels) == k)[0]
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
            list_clients_X[idx_client] += [dataset.train_data[idx_sample]]
            list_clients_y[idx_client] += [dataset.train_labels[idx_sample]]

        list_clients_X[idx_client] = np.array(list_clients_X[idx_client])   # 存完了一个client的600张图和label

    folder = DATASET_FOLDER
    print("len:", np.shape(list_clients_X), np.shape(list_clients_y))   # len: (100, 600) (100, 600)
    with open(folder + file_name, "wb") as output:
        pickle.dump((list_clients_X, list_clients_y), output)


# 1
def create_MNIST_dirichlet(
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
    print("matrix:", np.shape(matrix))

    # MNIST_train = datasets.MNIST(
    #     root=DATASET_FOLDER,
    #     train=True,
    #     download=True,
    #     transform=transforms.ToTensor(),
    # )
    MNIST_train = RawMnistDataset(
        DATASET_FOLDER+'MNIST/raw', "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", transform=transforms.ToTensor())

    # MNIST_test = datasets.MNIST(
    #     root=DATASET_FOLDER,
    #     train=False,
    #     download=True,
    #     transform=transforms.ToTensor(),
    # )
    MNIST_test = RawMnistDataset(
        DATASET_FOLDER+'MNIST/raw', "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", transform=transforms.ToTensor())


    file_name_train = f"{dataset_name}_train_{n_clients}.pkl"
    partition_MNIST_dataset(
        MNIST_train,
        file_name_train,
        balanced,
        matrix,
        n_clients,
        n_classes,
        True,
    )

    file_name_test = f"{dataset_name}_test_{n_clients}.pkl"
    partition_MNIST_dataset(
        MNIST_test,
        file_name_test,
        balanced,
        matrix,
        n_clients,
        n_classes,
        False,
    )


# 3
def clients_set_MNIST(
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
        dataset_object = MnistDataset(file_name, k)

        dataset_dl = DataLoader(
            dataset_object, batch_size=batch_size, shuffle=shuffle
        )

        list_dl.append(dataset_dl)

    return list_dl


# 0
def get_MNIST_dataloaders(dataset_name, batch_size: int, shuffle=True, reset_data=False):
    """得到训练集DataLoader对象列表

    Args:
        dataset_name (str): {dataset}_{balanced}_{alpha}, such as "CIFAR10_nbal_10.0"
        batch_size (int): default=50

    Returns:
        list_dls_train : list of DataLoader objects

    """

    print("dataset_name:", dataset_name)

    if dataset_name == "MNIST_iid":

        n_clients = 100
        samples_train, samples_test = 600, 100

        mnist_trainset = datasets.MNIST(
            root=DATASET_FOLDER,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        mnist_train_split = torch.utils.data.random_split(
            mnist_trainset, [samples_train] * n_clients
        )
        list_dls_train = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in mnist_train_split
        ]

        mnist_testset = datasets.MNIST(
            root=DATASET_FOLDER,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        mnist_test_split = torch.utils.data.random_split(
            mnist_testset, [samples_test] * n_clients
        )
        list_dls_test = [
            torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in mnist_test_split
        ]

    elif dataset_name == "MNIST_shard":
        n_clients = 100
        samples_train, samples_test = 500, 80

        file_name_train = f"MNIST_shard_train_{n_clients}_{samples_train}.pkl"
        path_train = DATASET_FOLDER + file_name_train

        file_name_test = f"MNIST_shard_test_{n_clients}_{samples_test}.pkl"
        path_test = DATASET_FOLDER + file_name_test

        if not os.path.isfile(path_train):
            create_MNIST_ds_1shard_per_client(
                n_clients, samples_train, samples_test
            )

        list_dls_train = clients_set_MNIST_shard(
            path_train, n_clients, batch_size=batch_size, shuffle=shuffle
        )

        list_dls_test = clients_set_MNIST_shard(
            path_test, n_clients, batch_size=batch_size, shuffle=shuffle
        )

    # dirichlet
    elif dataset_name[:5] == "MNIST":
        n_classes = 10
        n_clients = 100
        balanced = dataset_name[6:10] == "bbal"
        alpha = float(dataset_name[11:])

        file_name_train = f"{dataset_name}_train_{n_clients}.pkl"
        path_train = DATASET_FOLDER + file_name_train

        file_name_test = f"{dataset_name}_test_{n_clients}.pkl"
        path_test = DATASET_FOLDER + file_name_test

        # 固定数据集
        if not os.path.isfile(path_train):
            print("creating dataset alpha:", alpha)
            create_MNIST_dirichlet(
                dataset_name, balanced, alpha, n_clients, n_classes
            )

        list_dls_train = clients_set_MNIST(
            path_train, n_clients, batch_size, True
        )

        list_dls_test = clients_set_MNIST(
            path_test, n_clients, batch_size, True
        )

    list_len = list()
    for dl in list_dls_train:
        list_len.append(len(dl.dataset))
    # print("list_len: ", list_len)
    with open(f"{config.ROOT_PATH}saved_exp_info/len_dbs/{dataset_name}.pkl", "wb") as output:
        pickle.dump(list_len, output)

    return list_dls_train, list_dls_test

#
# list_dls_train, list_dls_test = get_MNIST_dataloaders("MNIST_bbal_0.1", 50)