#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import time
import argparse
import config
import numpy as np
from copy import deepcopy

class DatasetObject:
    def __init__(self, dataset, n_client, name, train_set, test_set):
        self.dataset  = dataset
        self.n_client = n_client
        self.name = name
        clnt_x = []
        clnt_y = []
        for clnt in train_set:
            for X, y in clnt:
                clnt_x.append(X.numpy())
                clnt_y.append(y.numpy().reshape(-1,1))
        self.clnt_x = np.asarray(clnt_x)
        self.clnt_y = np.asarray(clnt_y)
        tst_x = []
        tst_y = []
        for clnt in test_set:
            for X, y in clnt:
                tst_x.append(X.numpy())
                tst_y.append(y.numpy().reshape(-1,1))
        self.tst_x = np.asarray(tst_x)
        self.tst_y = np.asarray(tst_y)
        self.train_set = deepcopy(train_set)
        self.test_set = deepcopy(test_set)


def main(args):
    start_time = time.time()

    """UPLOADING THE DATASETS"""
    import sys

    # print(
    #     "dataset - sampling - sim_type - seed - n_SGD - lr - decay - p - force - mu"
    # )
    # print(sys.argv[1:])
    print(args)

    dataset = args.dataset + args.datasetarg
    model_name = args.dataset
    sampling = args.sampling
    sim_type = args.sim_type
    seed = args.seed
    n_SGD = args.n_SGD
    lr = args.learning_rate
    decay = args.decay
    p = args.p
    force = args.force
    mu = args.mu
    update_method = args.update_method
    convex = args.convex
    cluster_number = args.cluster_number

    NEWLINE = '\n'

    """GET THE HYPERPARAMETERS"""
    from py_func.hyperparams import get_hyperparams

    n_iter, batch_size, meas_perf_period = get_hyperparams(dataset, n_SGD)
    print("number of iterations", n_iter)
    print("batch size", batch_size)
    print("percentage of sampled clients", p)
    print("metric_period", meas_perf_period)
    print("regularization term", mu)
    print("GPU:", config.USE_GPU)
    print("FL_update method:", update_method)

    """NAME UNDER WHICH THE EXPERIMENT'S VARIABLES WILL BE SAVED"""
    from py_func.hyperparams import get_file_name

    file_name = get_file_name(
        dataset, sampling, sim_type, seed, n_SGD, lr, decay, p, mu, update_method, convex, cluster_number
    )
    print(file_name)

    if os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") and not force:
        print("-"*100)
        print("实验存在且不要求覆盖实验，进行下一个")
        print("-"*100)
        return

    """GET THE DATASETS USED FOR THE FL TRAINING"""
    from data_partition.CIFAR10_dirichlet import get_CIFAR10_dataloaders, get_num_cnt
    from data_partition.MNIST import get_MNIST_dataloaders
    from data_partition.FMNIST import get_FMNIST_dataloaders

    if dataset[:5] == "MNIST":
        list_dls_train, list_dls_test, list_dls_train_full, list_dls_test_full = get_MNIST_dataloaders(dataset, batch_size)
    elif dataset[:5] == "CIFAR":
        list_dls_train, list_dls_test, list_dls_train_full, list_dls_test_full = get_CIFAR10_dataloaders(dataset, batch_size)
    elif dataset[:6] == "FMNIST":
        list_dls_train, list_dls_test, list_dls_train_full, list_dls_test_full = get_FMNIST_dataloaders(dataset, batch_size)

    get_num_cnt(dataset, list_dls_train)

    # """CLUSTER THE CLIENTS"""
    # from cluster import cluster_train
    #
    # fr = open('saved_exp_info/cluster_result/cifar10_balance_dir.pkl', 'rb')
    # cluster_result = pickle.load(fr)

    """NUMBER OF SAMPLED CLIENTS"""
    n_sampled = int(p * len(list_dls_train))
    print("number fo sampled clients", n_sampled)

    """LOAD THE INTIAL GLOBAL MODEL"""
    from py_func.create_model import load_model

    model_0 = load_model(dataset, seed, convex)
    print(model_0)

    if update_method == "AVG":
        """FEDAVG with random sampling"""

        if sampling == "random" and (
                not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force
        ):
            from py_func.FedProx import FedProx_sampling_random

            FedProx_sampling_random(
                model_0,
                sampling,
                n_sampled,
                list_dls_train,
                list_dls_test,
                n_iter,
                n_SGD,
                lr,
                file_name,
                decay,
                meas_perf_period,
                mu,
            )

            # from py_func.FedProx import FedProx_stratified_sampling
            #
            # FedProx_stratified_sampling(
            #     sampling,
            #     model_0,
            #     n_sampled,
            #     list_dls_train,
            #     list_dls_test,
            #     n_iter,
            #     n_SGD,
            #     lr,
            #     file_name,
            #     sim_type,
            #     0,
            #     decay,
            #     meas_perf_period,
            #     mu,
            # )

        """Run FEDAVG with clustered sampling"""
        if (sampling == "ours") and (
                not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force
        ):
            from py_func.FedProx import FedProx_stratified_sampling

            FedProx_stratified_sampling(
                dataset,
                sampling,
                model_0,
                n_sampled,
                list_dls_train,
                list_dls_test,
                n_iter,
                n_SGD,
                lr,
                file_name,
                sim_type,
                0,
                decay,
                meas_perf_period,
                mu,
                cluster_number=cluster_number
            )

        if (sampling == "clustered_1" or sampling == "clustered_2") and (
                not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force
        ):
            from py_func.FedProx import FedProx_clustered_sampling

            FedProx_clustered_sampling(
                sampling,
                model_0,
                n_sampled,
                list_dls_train,
                list_dls_test,
                n_iter,
                n_SGD,
                lr,
                file_name,
                sim_type,
                0,
                decay,
                meas_perf_period,
                mu,
            )

        """RUN FEDAVG with perfect sampling for MNIST-shard"""
        if (
                sampling == "perfect"
                and dataset == "MNIST_shard"
                and (not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force)
        ):
            from py_func.FedProx import FedProx_sampling_target

            FedProx_sampling_target(
                model_0,
                sampling,
                n_sampled,
                list_dls_train,
                list_dls_test,
                n_iter,
                n_SGD,
                lr,
                file_name,
                decay,
                mu,
            )

        """RUN FEDAVG with its original sampling scheme sampling clients uniformly"""
        if sampling == "FedAvg" and (
                not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force
        ):
            from py_func.FedProx import FedProx_FedAvg_sampling

            FedProx_FedAvg_sampling(
                model_0,
                sampling,
                n_sampled,
                list_dls_train,
                list_dls_test,
                n_iter,
                n_SGD,
                lr,
                file_name,
                decay,
                meas_perf_period,
                mu,
            )

    if update_method == "SCAFFOLD":
        """FEDSCAFFOLD with random sampling"""
        from utils_methods import train_SCAFFOLD
        data_obj = DatasetObject(dataset=model_name, 
                                n_client=len(list_dls_train), 
                                name=file_name, 
                                train_set=list_dls_train,
                                test_set=list_dls_test)
        model_func = lambda : load_model(dataset, seed, convex)
        data_path = ''
        import torch
        if not os.path.exists('%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name)):
            if not os.path.exists('%sModel/%s/' %(data_path, data_obj.name)):
                print("Create a new directory")
                os.mkdir('%sModel/%s/' %(data_path, data_obj.name))
            torch.save(model_0.state_dict(), '%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name))
        else:
            # Load model
            model_0.load_state_dict(torch.load('%sModel/%s/%s_init_mdl.pt' %(data_path, data_obj.name, model_name)))    

        save_period = 200
        act_prob = 0.15
        [fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all] = train_SCAFFOLD(data_obj=data_obj, act_prob=act_prob ,
                                            learning_rate=lr, batch_size=batch_size, n_minibatch=n_SGD, file_name=file_name,
                                            com_amount=n_iter, print_per=n_SGD//2, weight_decay=0, n_sampled=n_sampled,
                                            model_func=model_func, init_model=model_0,
                                            sch_step=1, sch_gamma=1, save_period=save_period, suffix=model_name, 
                                            trial=False, data_path=data_path, lr_decay_per_round=decay)

        sampling = ""

        if sampling == "random" and (
                not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force
        ):
            from py_func.FedProx import SCAFFOLD_sampling_random

            SCAFFOLD_sampling_random(
                model=model_0,
                sampling=sampling,
                n_sampled=n_sampled,
                training_sets=list_dls_train,
                testing_sets=list_dls_test,
                n_iter=n_iter,
                n_SGD=n_SGD,
                lr=lr,
                file_name=file_name,
                batch_size=batch_size,
                decay=decay,
                metric_period=meas_perf_period,
                mu=mu,
                training_sets_full=list_dls_train_full,
                testing_sets_full=list_dls_test_full
            )

        """Run FEDSCAFFOLD with clustered sampling"""
        if (sampling == "ours") and (
                not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force
        ):
            from py_func.FedProx import SCAFFOLD_stratified_sampling

            SCAFFOLD_stratified_sampling(
                dataset=dataset,
                sampling=sampling,
                model=model_0,
                n_sampled=n_sampled,
                training_sets=list_dls_train,
                testing_sets=list_dls_test,
                n_iter=n_iter,
                n_SGD=n_SGD,
                lr=lr,
                file_name=file_name,
                batch_size=batch_size,
                sim_type=sim_type,
                iter_FP=0,
                decay=decay,
                metric_period=meas_perf_period,
                mu=mu,
                training_sets_full=list_dls_train_full,
                testing_sets_full=list_dls_test_full,
                cluster_number=cluster_number
            )

        if (sampling == "clustered_1" or sampling == "clustered_2") and (
                not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force
        ):
            from py_func.FedProx import SCAFFOLD_clustered_sampling

            SCAFFOLD_clustered_sampling(
                sampling=sampling,
                model=model_0,
                n_sampled=n_sampled,
                training_sets=list_dls_train,
                testing_sets=list_dls_test,
                n_iter=n_iter,
                n_SGD=n_SGD,
                lr=lr,
                file_name=file_name,
                batch_size=batch_size,
                sim_type=sim_type,
                iter_FP=0,
                decay=decay,
                metric_period=meas_perf_period,
                mu=mu,
                training_sets_full=list_dls_train_full,
                testing_sets_full=list_dls_test_full
            )

        """RUN FEDSCAFFOLD with perfect sampling for MNIST-shard"""
        if (
                sampling == "perfect"
                and dataset == "MNIST_shard"
                and (not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force)
        ):
            from py_func.FedProx import FedProx_sampling_target

            FedProx_sampling_target(
                model_0,
                sampling,
                n_sampled,
                list_dls_train,
                list_dls_test,
                n_iter,
                n_SGD,
                lr,
                file_name,
                decay,
                mu,
            )

        """RUN FEDSCAFFOLD with its original sampling scheme sampling clients uniformly"""
        if sampling == "FedAvg" and (
                not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force
        ):
            from py_func.FedProx import SCAFFOLD_FedAvg_sampling

            SCAFFOLD_FedAvg_sampling(
                model=model_0,
                sampling=sampling,
                n_sampled=n_sampled,
                training_sets=list_dls_train,
                testing_sets=list_dls_test,
                n_iter=n_iter,
                n_SGD=n_SGD,
                lr=lr,
                file_name=file_name,
                batch_size=batch_size,
                decay=decay,
                metric_period=meas_perf_period,
                mu=mu,
                training_sets_full=list_dls_train_full,
                testing_sets_full=list_dls_test_full
            )

    print("EXPERIMENT IS FINISHED")
    print("Paras: ", args)
    print("start at: " + time.asctime(time.localtime(start_time)))
    print("Finish at: " + time.asctime(time.localtime(time.time())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST",
                        choices=["CIFAR10", "MNIST", "FMNIST"])
    parser.add_argument("--datasetarg", type=str, default="_shard")
    parser.add_argument("--sampling", type=str, default="ours",
                        choices=["random", "ours", "clustered_1", "clustered_2", "FedAvg"])
    parser.add_argument("--sim_type", type=str, default="any",
                        choices=["cosine", "L2", "L1", "any"])
    parser.add_argument("--update_method", type=str, default="AVG",
                        choices=["SCAFFOLD","AVG"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_SGD", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=0.1)
    parser.add_argument("--mu", type=float, default=0)
    parser.add_argument("--cluster_number", type=int, default=5)
    parser.add_argument('-f', '--force', dest="force", action='store_true', help="Force to simulate.")
    parser.add_argument('-v', "--convex", dest='convex', action='store_true', help="Convexity of MNIST")
    # parser.add_argument("--model", type=str, default="cnn", choices=["linear", "mclr", "dnn", "cnn"])
    # parser.add_argument("--batch_size", type=int, default=60)
    # parser.add_argument("--num_glob_iters", type=int, default=5)
    # parser.add_argument("--local_epochs", type=int, default=1)
    # parser.add_argument("--hyper_learning_rate", type=float, default=0.02, help=" Learning rate of FEDL")
    # parser.add_argument("--algorithm", type=str, default="FedAvg", choices=["FEDL", "FedAvg", "SCAFFOLD"])
    # parser.add_argument("--clients_per_round", type=int, default=0, help="Number of Users per round")
    # parser.add_argument("--L", type=int, default=0.004, help="Regularization term")
    # parser.add_argument("--rho", type=float, default=0, help="Condition Number")
    # parser.add_argument("--noise", type=float, default=False, help="Applies noisy channel effect")
    # parser.add_argument("--pre-coding", type=float, default=False, help="Applies pre-coding")
    # parser.add_argument("--times", type=int, default=1, help="Running time")
    args = parser.parse_args()
    # args.dataset += args.datasetarg
    main(args)
