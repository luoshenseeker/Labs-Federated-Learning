#!/usr/bin/env python
# coding: utf-8
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer

import numpy as np
from copy import deepcopy

import random

from torch.autograd import Variable

import config

if config.USE_GPU:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 将数据处理成Variable, 如果有GPU, 可以转成cuda形式
def get_variable(x):
    x = Variable(x)
    return x.cuda() if config.USE_GPU else x


def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)


def FedAvg_agregation_process(model, clients_models_hist: list, weights: list):
    """Creates the new model of a given iteration with the models of the other
    clients"""

    new_model = deepcopy(model)
    set_to_zero_model_weights(new_model)

    for k, client_hist in enumerate(clients_models_hist):

        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution = client_hist[idx].data * weights[k]
            layer_weights.data.add_(contribution)

    return new_model


def FedAvg_agregation_process_for_FA_sampling(
    model, clients_models_hist: list, weights: list
):
    """Creates the new model of a given iteration with the models of the other
    clients"""

    new_model = deepcopy(model)

    for layer_weigths in new_model.parameters():
        layer_weigths.data.sub_(sum(weights) * layer_weigths.data)

    for k, client_hist in enumerate(clients_models_hist):

        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution = client_hist[idx].data * weights[k]
            layer_weights.data.add_(contribution)

    return new_model

def Fed_SCAFFOLD_aggregation_process(
    model, server_controls, delta_model_group, delta_controls_group, clients_models_hist: list, weights: list, total_samples, tot_users_num: int
):
    new_model = deepcopy(model)
    # set_to_zero_model_weights(new_model)
    assert(len(clients_models_hist) > 0)
    num_of_selected_users = len(clients_models_hist)
    num_of_users = tot_users_num
    for delta_controls, delta_model in zip(delta_controls_group, delta_model_group):
        for param, control, del_control, del_model in zip(new_model.parameters(), server_controls,
                                                            delta_controls, delta_model):
            # param.data = param.data + del_model.data * num_of_samples / total_samples / num_of_selected_users
            param.data = param.data + del_model.data / num_of_selected_users
            control.data = control.data + del_control.data / num_of_users
    return new_model


def accuracy_dataset(model, dataset):
    """Compute the accuracy of `model` on `test_data`"""

    correct = 0

    for features, labels in dataset:

        features = get_variable(features)
        labels = get_variable(labels)

        predictions = model(features)
        _, predicted = predictions.max(1, keepdim=True)

        correct += torch.sum(predicted.view(-1, 1) == labels.view(-1, 1)).item()

    accuracy = 100 * correct / len(dataset.dataset)

    return accuracy


def loss_dataset(model, train_data, loss_f):
    """Compute the loss of `model` on `test_data`"""
    loss = 0
    for idx, (features, labels) in enumerate(train_data):

        features = get_variable(features)
        labels = get_variable(labels)

        predictions = model(features)
        loss += loss_f(predictions, labels)

    loss /= idx + 1
    return loss


def loss_classifier(predictions, labels):

    criterion = nn.CrossEntropyLoss()
    return criterion(predictions, labels)


def n_params(model):
    """return the number of parameters in the model"""

    n_params = sum(
        [
            np.prod([tensor.size()[k] for k in range(len(tensor.size()))])
            for tensor in list(model.parameters())
        ]
    )

    return n_params


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters"""

    tensor_1 = list(model_1.parameters())
    tensor_2 = list(model_2.parameters())

    norm = sum(
        [
            torch.sum((tensor_1[i] - tensor_2[i]) ** 2)
            for i in range(len(tensor_1))
        ]
    )

    return norm


def local_learning(model, mu: float, optimizer, train_data, n_SGD: int, loss_f):

    model_0 = deepcopy(model)

    for _ in range(n_SGD):

        features, labels = next(iter(train_data))

        features = get_variable(features)
        labels = get_variable(labels)

        optimizer.zero_grad()

        predictions = model(features)

        batch_loss = loss_f(predictions, labels)
        batch_loss += mu / 2 * difference_models_norm_2(model, model_0)

        batch_loss.backward()
        optimizer.step()

def local_learning_scaffold(model, mu: float, optimizer, train_data, local_epoch: int, loss_f, delta_model, trainloaderfull, server_controls, client_controls, delta_client_controls, lr, batch_size):
    server_model = deepcopy(model)
    opt = 2
    if opt == 1:
        grads = [torch.zeros_like(p.data) for p in model.parameters() if p.requires_grad]
        # get grads
        optimizer.zero_grad()
        for x, y in trainloaderfull:
            x = get_variable(x)
            y = get_variable(y)
            output = model(x)
            loss = loss_f(output, y)
            loss.backward()
        for param, clone_param in zip(model.parameters(), grads):
            clone_param.data = param.data.clone()
            if(param.grad != None):
                if(clone_param.grad == None):
                    clone_param.grad = torch.zeros_like(param.grad)
                clone_param.grad.data = param.grad.data.clone()

    for _ in range(local_epoch):

        features, labels = next(iter(train_data))   # 单次训练只训练一个batch

        features = get_variable(features)
        labels = get_variable(labels)

        optimizer.zero_grad()

        predictions = model(features)

        batch_loss = loss_f(predictions, labels)
        batch_loss += mu / 2 * difference_models_norm_2(model, server_model)

        batch_loss.backward()
        optimizer.step(server_controls, client_controls)

        # get mode difference
        model_bak = deepcopy(model)
        server_model_bak = deepcopy(server_model)
        for local, server, delta in zip(model_bak.parameters(), server_model_bak.parameters(), delta_model):
            delta.data = local.data.detach() - server.data.detach()

        # get client new controls
        new_controls = [torch.zeros_like(p.data) for p in model.parameters() if p.requires_grad]
        if opt == 1:
            for new_control, grad in zip(new_controls, grads):
                new_control.data = grad.data
        if opt == 2:
            train_samples = len(train_data)
            for server_control, control, new_control, delta in zip(server_controls, client_controls, new_controls, delta_model):
                a = 1 / (math.ceil(train_samples / batch_size) * lr)
                new_control.data = control.data - server_control.data - delta.data * a

        # get controls difference
        for control, new_control, delta in zip(client_controls, new_controls, delta_client_controls):
            delta.data = new_control.data - control.data
            control.data = new_control.data
                 


import pickle


def save_pkl(dictionnary, directory, file_name):
    """Save the dictionnary in the directory under the file_name with pickle"""
    with open(f"saved_exp_info/{directory}/{file_name}.pkl", "wb") as output:
        pickle.dump(dictionnary, output)


def save_log(log, directory, file_name):
    inf = str(log)
    with open(f"saved_exp_info/{directory}/{file_name}.txt", "w") as output:
        output.write(inf)


def FedProx_sampling_random(
    model,
    sampling,
    n_sampled,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    decay=1,
    metric_period=1,
    mu=0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    loss_f = loss_classifier

    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):

        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i {sampling}: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):

        clients_params = []

        np.random.seed(i)
        sampled_clients = np.random.choice(
            K, size=n_sampled, replace=True, p=weights
        )

        for k in sampled_clients:

            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL
        model = FedAvg_agregation_process(
            deepcopy(model), clients_params, weights=[1 / n_sampled] * n_sampled
        )

        if i % metric_period == 0:
            # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            server_loss = np.dot(weights, loss_hist[i + 1])
            server_acc = np.dot(weights, acc_hist[i + 1])

            print(
                f"====> i {sampling}: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist


def FedProx_stratified_sampling(
    dataset: str,
    sampling: str,
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    sim_type: str,
    iter_FP=0,
    decay=1.0,
    metric_period=1,
    mu=0.0,
):
    """all the clients are considered in this implementation of FedProx
        Parameters:
            - `model`: common structure used by the clients and the server
            - `training_sets`: list of the training sets. At each index is the
                trainign set of client "index"
            - `n_iter`: number of iterations the server will run
            - `testing_set`: list of the testing sets. If [], then the testing
                accuracy is not computed
            - `mu`: regularixation term for FedProx. mu=0 for FedAvg
            - `epochs`: number of epochs each client is running
            - `lr`: learning rate of the optimizer
            - `decay`: to change the learning rate at each iteration

        returns :
            - `model`: the final global model
        """

    from py_func.clustering import get_matrix_similarity_from_grads

    loss_f = loss_classifier

    # Variables initialization
    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    # if dataset[:5] == "MNIST":
    #     list_dls_train, list_dls_test, list_dls_train_full, list_dls_test_full = get_MNIST_dataloaders(dataset, batch_size)
    # elif dataset[:5] == "CIFAR":
    #     fr = open('saved_exp_info/cluster_result/cifar10_balance_dir.pkl', 'rb')
    #     cluster_result = pickle.load(fr)
    # elif dataset[:6] == "FMNIST":
    #     list_dls_train, list_dls_test, list_dls_train_full, list_dls_test_full = get_FMNIST_dataloaders(dataset, batch_size)

    from cluster.cluster_train import cluster_training

    result_path = cluster_training(dataset)
    fr = open(result_path, 'rb')
    cluster_result = pickle.load(fr)

    N_CLASSES = len(cluster_result)
    SIZE_CLASSES = [len(cls) for cls in cluster_result]
    N_CLIENTS = sum(len(c) for c in cluster_result)  # number of clients

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i {sampling}: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    # INITILIZATION OF THE GRADIENT HISTORY AS A LIST OF 0

    for i in range(n_iter):

        clients_params = []
        clients_models = []
        sampled_clients_for_grad = []

        # 鲁棒性问题！抽样率p的改变
        # GET THE CLIENTS' CHOSEN PROBABILITY
        chosen_p = np.zeros((N_CLASSES, N_CLIENTS)).astype(float)
        for j, cls in enumerate(cluster_result):
            for k in range(N_CLIENTS):
                if k in cls:
                    chosen_p[j][k] = round(1/SIZE_CLASSES[j], 12)

        # chosen_p /= float(chosen_p.sum(), 12)  # normalize

        # np.set_printoptions(threshold=np.inf)  # 解决显示不完全问题
        # print(chosen_p)
        # print(np.shape(chosen_p))

        # distri_clusters = np.zeros((n_sampled, n_clients))
        # for l in range(n_sampled):
        #     distri_clusters[l] /= np.sum(distri_clusters[l])

        from py_func.clustering import sample_clients

        # allocation需更改每一类采的个数相同问题
        # shard分片分10类，从每类中取一个
        # print("手动实现随机抽样！！！")
        # from numpy.random import choice
        # selects = choice(100, 10, replace=False, p=[0.01 for _ in range(100)])
        # for k in selects:

        print("### chosen client index: ###")
        for _ in range(int(n_sampled / N_CLASSES)):  # 每个class抽两个client训练，每个epoch中12345,12345
            for k in sample_clients(chosen_p):
                print(f"{k},", end="")

                local_model = deepcopy(model)
                local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

                local_learning(
                    local_model,
                    mu,
                    local_optimizer,
                    training_sets[k],
                    n_SGD,
                    loss_f,
                )

                # SAVE THE LOCAL MODEL TRAINED
                list_params = list(local_model.parameters())
                list_params = [
                    tens_param.detach() for tens_param in list_params
                ]
                clients_params.append(list_params)
                clients_models.append(deepcopy(local_model))

                sampled_clients_for_grad.append(k)
                sampled_clients_hist[i, k] = 1
        print("\b ")

        # CREATE THE NEW GLOBAL MODEL AND SAVE IT
        model = FedAvg_agregation_process(
            deepcopy(model), clients_params, weights=[1 / n_sampled] * n_sampled
        )

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL

        if i % metric_period == 0:

            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            server_loss = np.dot(weights, loss_hist[i + 1])
            server_acc = np.dot(weights, acc_hist[i + 1])

            print(
                f"====> i {sampling}: {i + 1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    # save_pkl(models_hist, "local_model_history", file_name)
    # save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist


def FedProx_clustered_sampling(
    sampling: str,
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    sim_type: str,
    iter_FP=0,
    decay=1.0,
    metric_period=1,
    mu=0.0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    from scipy.cluster.hierarchy import linkage
    from py_func.clustering import get_matrix_similarity_from_grads

    if sampling == "clustered_2":
        from py_func.clustering import get_clusters_with_alg2
    from py_func.clustering import sample_clients

    loss_f = loss_classifier

    # Variables initialization
    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i {sampling}: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    # INITILIZATION OF THE GRADIENT HISTORY AS A LIST OF 0

    if sampling == "clustered_1":
        from py_func.clustering import get_clusters_with_alg1

        distri_clusters = get_clusters_with_alg1(n_sampled, weights)

    elif sampling == "clustered_2":
        from py_func.clustering import get_gradients

        gradients = get_gradients(sampling, model, [model] * K)

    for i in range(n_iter):

        previous_global_model = deepcopy(model)

        clients_params = []
        clients_models = []
        sampled_clients_for_grad = []

        if i < iter_FP:
            print("MD sampling, now quit!")
        #
        #     np.random.seed(i)
        #     sampled_clients = np.random.choice(
        #         K, size=n_sampled, replace=True, p=weights
        #     )
        #
        #     for k in sampled_clients:
        #
        #         local_model = deepcopy(model)
        #         local_optimizer = optim.SGD(local_model.parameters(), lr=lr)
        #
        #         local_learning(
        #             local_model,
        #             mu,
        #             local_optimizer,
        #             training_sets[k],
        #             n_SGD,
        #             loss_f,
        #         )
        #
        #         # SAVE THE LOCAL MODEL TRAINED
        #         list_params = list(local_model.parameters())
        #         list_params = [
        #             tens_param.detach() for tens_param in list_params
        #         ]
        #         clients_params.append(list_params)
        #         clients_models.append(deepcopy(local_model))
        #
        #         sampled_clients_for_grad.append(k)
        #         sampled_clients_hist[i, k] = 1

        else:
            if sampling == "clustered_2":

                # GET THE CLIENTS' SIMILARITY MATRIX
                sim_matrix = get_matrix_similarity_from_grads(
                    gradients, distance_type=sim_type
                )

                # GET THE DENDROGRAM TREE ASSOCIATED
                linkage_matrix = linkage(sim_matrix, "ward")

                distri_clusters = get_clusters_with_alg2(
                    linkage_matrix, n_sampled, weights
                )

            for k in sample_clients(distri_clusters):

                local_model = deepcopy(model)
                local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

                local_learning(
                    local_model,
                    mu,
                    local_optimizer,
                    training_sets[k],
                    n_SGD,
                    loss_f,
                )

                # SAVE THE LOCAL MODEL TRAINED
                list_params = list(local_model.parameters())
                list_params = [
                    tens_param.detach() for tens_param in list_params
                ]
                clients_params.append(list_params)
                clients_models.append(deepcopy(local_model))

                sampled_clients_for_grad.append(k)
                sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL AND SAVE IT
        model = FedAvg_agregation_process(
            deepcopy(model), clients_params, weights=[1 / n_sampled] * n_sampled
        )

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        if i % metric_period == 0:

            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            server_loss = np.dot(weights, loss_hist[i + 1])
            server_acc = np.dot(weights, acc_hist[i + 1])

            print(
                f"====> i {sampling}: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        # UPDATE THE HISTORY OF LATEST GRADIENT
        if sampling == "clustered_2":
            gradients_i = get_gradients(
                sampling, previous_global_model, clients_models
            )
            for idx, gradient in zip(sampled_clients_for_grad, gradients_i):
                gradients[idx] = gradient

        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist


def FedProx_sampling_target(
    model,
    sampling,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    decay=1,
    mu=0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    loss_f = loss_classifier

    # Variables initialization
    n_samples = sum([len(db.dataset) for db in training_sets])
    weights = [len(db.dataset) / n_samples for db in training_sets]
    print("Clients' weights:", weights)

    loss_hist = [
        [
            float(loss_dataset(model, dl, loss_f).detach())
            for dl in training_sets
        ]
    ]
    acc_hist = [[accuracy_dataset(model, dl) for dl in testing_sets]]
    server_hist = [
        [tens_param.detach().numpy() for tens_param in list(model.parameters())]
    ]
    models_hist = []
    sampled_clients_hist = []

    server_loss = sum(
        [weights[i] * loss_hist[-1][i] for i in range(len(weights))]
    )
    server_acc = sum(
        [weights[i] * acc_hist[-1][i] for i in range(len(weights))]
    )
    print(f"====> i {sampling}: 0 Loss: {server_loss} Server Test Accuracy: {server_acc}")

    for i in range(n_iter):

        clients_params = []
        clients_models = []
        sampled_clients_i = []

        for j in range(n_sampled):

            k = j * 10 + np.random.randint(10)

            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            clients_models.append(deepcopy(local_model))

            sampled_clients_i.append(k)

        # CREATE THE NEW GLOBAL MODEL
        model = FedAvg_agregation_process(
            deepcopy(model), clients_params, weights=[1 / n_sampled] * n_sampled
        )
        models_hist.append(clients_models)

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        loss_hist += [
            [
                float(loss_dataset(model, dl, loss_f).detach())
                for dl in training_sets
            ]
        ]
        acc_hist += [[accuracy_dataset(model, dl) for dl in testing_sets]]

        server_loss = sum(
            [weights[i] * loss_hist[-1][i] for i in range(len(weights))]
        )
        server_acc = sum(
            [weights[i] * acc_hist[-1][i] for i in range(len(weights))]
        )

        print(
            f"====> i {sampling}: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
        )

        server_hist.append(deepcopy(model))

        sampled_clients_hist.append(sampled_clients_i)

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist


def FedProx_FedAvg_sampling(
    model,
    sampling,
    n_sampled,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    decay=1,
    metric_period=1,
    mu=0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    loss_f = loss_classifier

    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):

        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i {sampling}: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):

        clients_params = []

        np.random.seed(i)
        sampled_clients = random.sample([x for x in range(K)], n_sampled)
        # print("sampled clients", sampled_clients)

        for k in sampled_clients:

            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL
        model = FedAvg_agregation_process_for_FA_sampling(
            deepcopy(model),
            clients_params,
            weights=[weights[client] for client in sampled_clients],
        )

        if i % metric_period == 0:
            # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            server_loss = np.dot(weights, loss_hist[i + 1])
            server_acc = np.dot(weights, acc_hist[i + 1])

            print(
                f"====> i {sampling}: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist

scaffold_weight_decay = 0
class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)
        pass

    def step(self, server_controls, client_controls, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group, c, ci in zip(self.param_groups, server_controls, client_controls):
            p = group['params'][0]  #  原实现中，对于cifar数据进行了字典包装
            # p = group[0]  # TODO：查看是否需要取列表元素 取第一个是因为params在一个列表里，列表长为1
            if p.grad is None:
                continue
            d_p = p.grad.data + c.data - ci.data
            p.data = p.data - d_p.data * group['lr']
        # for group in self.param_groups:
        #     for p, c, ci in zip(group['params'], server_controls, client_controls):
        #         if p.grad is None:
        #             continue
        #         d_p = p.grad.data + c.data - ci.data
        #         p.data = p.data - d_p.data * group['lr']
        return loss


def SCAFFOLD_sampling_random(
    model,
    sampling,
    n_sampled,
    training_sets: list,
    testing_sets: list,
    training_sets_full,
    testing_sets_full,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    batch_size: int,
    decay=1,
    metric_period=1,
    mu=0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    loss_f = loss_classifier

    server_controls = [torch.zeros_like(p.data) for p in model.parameters() if p.requires_grad]  # init server control

    K = len(training_sets)  # number of clients
    controls_group = [deepcopy(server_controls) for i in range(K)]             # init clients control and related vars needed later
    # server_controls = [deepcopy(server_controls) for i in range(K)]
    delta_controls_group = [deepcopy(server_controls) for i in range(K)]
    delta_model_group = [deepcopy(server_controls) for i in range(K)]

    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):

        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i {sampling}: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):

        clients_params = []
        clients_params_delta_control = []
        clients_params_delta_model = []
        total_samples = 0

        np.random.seed(i)
        sampled_clients = np.random.choice(
            K, size=n_sampled, replace=True, p=weights
        )

        for k in sampled_clients:

            local_model = deepcopy(model)
            local_optimizer = SCAFFOLDOptimizer(local_model.parameters(), lr=lr, weight_decay=scaffold_weight_decay)
            total_samples += len(training_sets[k].dataset)

            local_learning_scaffold(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f,
                delta_model=delta_model_group[k],
                trainloaderfull=training_sets_full,
                server_controls=server_controls,
                client_controls=controls_group[k],
                delta_client_controls=delta_controls_group[k],
                lr=lr,
                batch_size=batch_size
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            clients_params_delta_control.append(delta_controls_group[k])
            clients_params_delta_model.append(delta_model_group[k])

            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL
        model = Fed_SCAFFOLD_aggregation_process(
            model=model,
            server_controls=server_controls,
            delta_model_group=clients_params_delta_model,
            delta_controls_group=clients_params_delta_control,
            clients_models_hist=clients_params,
            tot_users_num=K,
            total_samples=total_samples,
            weights=[1 / n_sampled] * n_sampled
        )

        if i % metric_period == 0:
            # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            server_loss = np.dot(weights, loss_hist[i + 1])
            server_acc = np.dot(weights, acc_hist[i + 1])

            print(
                f"====> i {sampling}: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist

def SCAFFOLD_FedAvg_sampling(
    model,
    sampling,
    n_sampled,
    training_sets: list,
    testing_sets: list,
    training_sets_full,
    testing_sets_full,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    batch_size: int,
    decay=1,
    metric_period=1,
    mu=0,
):
    """all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            trainign set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularixation term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    loss_f = loss_classifier

    server_controls = [torch.zeros_like(p.data) for p in model.parameters() if p.requires_grad]  # init server control

    K = len(training_sets)  # number of clients
    controls_group = [deepcopy(server_controls) for i in range(K)]             # init clients control and related vars needed later
    # server_controls = [deepcopy(server_controls) for i in range(K)]
    delta_controls_group = [deepcopy(server_controls) for i in range(K)]
    delta_model_group = [deepcopy(server_controls) for i in range(K)]

    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):

        loss_hist[0, k] = float(loss_dataset(model, dl, loss_f).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i {sampling}: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):

        clients_params = []
        clients_params_delta_control = []
        clients_params_delta_model = []
        total_samples = 0

        np.random.seed(i)
        sampled_clients = random.sample([x for x in range(K)], n_sampled)
        # print("sampled clients", sampled_clients)

        for k in sampled_clients:

            local_model = deepcopy(model)
            local_optimizer = SCAFFOLDOptimizer(local_model.parameters(), lr=lr, weight_decay=scaffold_weight_decay)
            total_samples += len(training_sets[k].dataset)

            local_learning_scaffold(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_f,
                delta_model=delta_model_group[k],
                trainloaderfull=training_sets_full,
                server_controls=server_controls,
                client_controls=controls_group[k],
                delta_client_controls=delta_controls_group[k],
                lr=lr,
                batch_size=batch_size
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            clients_params_delta_control.append(delta_controls_group[k])
            clients_params_delta_model.append(delta_model_group[k])

            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL
        model = Fed_SCAFFOLD_aggregation_process(
            model=model,
            server_controls=server_controls,
            delta_model_group=clients_params_delta_model,
            delta_controls_group=clients_params_delta_control,
            clients_models_hist=clients_params,
            tot_users_num=K,
            total_samples=total_samples,
            weights=[1 / n_sampled] * n_sampled
        )

        if i % metric_period == 0:
            # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
            for k, dl in enumerate(training_sets):
                loss_hist[i + 1, k] = float(
                    loss_dataset(model, dl, loss_f).detach()
                )

            for k, dl in enumerate(testing_sets):
                acc_hist[i + 1, k] = accuracy_dataset(model, dl)

            server_loss = np.dot(weights, loss_hist[i + 1])
            server_acc = np.dot(weights, acc_hist[i + 1])

            print(
                f"====> i {sampling}: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
            )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)
    save_pkl(sampled_clients_hist, "sampled_clients", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist