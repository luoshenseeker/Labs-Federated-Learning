#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import time
import argparse


def main(args):
    start_time = time.time()

    """UPLOADING THE DATASETS"""
    import sys

    # print(
    #     "dataset - sampling - sim_type - seed - n_SGD - lr - decay - p - force - mu"
    # )
    # print(sys.argv[1:])
    print(args)

    dataset = args.dataset
    sampling = args.sampling
    sim_type = args.sim_type
    seed = args.seed
    n_SGD = args.n_SGD
    lr = args.learning_rate
    decay = args.decay
    p = args.p
    force = args.force
    mu = args.mu

    NEWLINE = '\n'

    """GET THE HYPERPARAMETERS"""
    from py_func.hyperparams import get_hyperparams

    n_iter, batch_size, meas_perf_period = get_hyperparams(dataset, n_SGD)
    print("number of iterations", n_iter)
    print("batch size", batch_size)
    print("percentage of sampled clients", p)
    print("metric_period", meas_perf_period)
    print("regularization term", mu)

    """NAME UNDER WHICH THE EXPERIMENT'S VARIABLES WILL BE SAVED"""
    from py_func.hyperparams import get_file_name

    file_name = get_file_name(
        dataset, sampling, sim_type, seed, n_SGD, lr, decay, p, mu
    )
    print(file_name)

    """GET THE DATASETS USED FOR THE FL TRAINING"""
    from data_partition.CIFAR10_dirichlet import get_CIFAR10_dataloaders, get_num_cnt
    from data_partition.MNIST import get_MNIST_dataloaders
    from data_partition.FMNIST import get_FMNIST_dataloaders

    if dataset[:5] == "MNIST":
        list_dls_train, list_dls_test = get_MNIST_dataloaders(dataset, batch_size)
    elif dataset[:5] == "CIFAR":
        list_dls_train, list_dls_test = get_CIFAR10_dataloaders(dataset, batch_size)
    elif dataset[:6] == "FMNIST":
        list_dls_train, list_dls_test = get_FMNIST_dataloaders(dataset, batch_size)

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

    model_0 = load_model(dataset, seed)
    print(model_0)

    """FEDAVG with random sampling"""
    if sampling == "random" and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or force
    ):
        from py_func.FedProx import FedProx_sampling_random

        FedProx_sampling_random(
            model_0,
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

    print("EXPERIMENT IS FINISHED")
    print("Paras: " + sys.argv[1:])
    print("start at: " + time.asctime(time.localtime(start_time)))
    print("Finish at " + time.asctime(time.localtime(time.time())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="CIFAR",
                        choices=["CIFAR", "MNIST", "FMNIST"])
    parser.add_argument("--sampling", type=str, default="random",
                        choices=["random", "ours", "important", "cluster"])
    parser.add_argument("--sim_type", type=str, default="any",
                        choices=["cosine", "L2", "L1", "any"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_SGD", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--decay", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=0.01)
    parser.add_argument("--mu", type=float, default=0)
    parser.add_argument('-f', '--force', action='force', help="Force to simulate.")
    parser.add_argument('-v', "--convex", action='convex', help="Convexity of MNIST")
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
    main(args)
