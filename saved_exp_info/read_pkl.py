import pickle
import numpy as np


def read_pkl(dataloaders):
    np.set_printoptions(threshold=np.inf) #解决显示不完全问题

    fr=open('acc/CIFAR10_bbal_0.01_ours_any_i800_N80_lr0.05_B50_d1.0_p0.08_m1_0.pkl', 'rb')

    acc_hist = pickle.load(fr)

    # inf = str(acc_hist)
    # ft = open("./acc_hist.txt",'w')
    # ft.write(inf)

    server_acc = []
    # server_loss = np.dot(weights, loss_hist[i + 1])
    for i in range(len(acc_hist)):
        n_samples = np.array([len(db.dataset) for db in dataloaders])
        weights = n_samples / np.sum(n_samples)
        if np.dot(weights, acc_hist[i]) != 0.0:
            server_acc.append(np.dot(weights, acc_hist[i]))
        # print(server_acc)

    # inf = str(server_acc)
    # ft = open("./server_acc.txt", 'w')
    # ft.write(inf)

    from matplotlib import pyplot as plt

    # 共600轮，做了300次evaluate
    x = np.arange(0, 800)
    y = server_acc
    plt.plot(x, y)
    plt.show()
