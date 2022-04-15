import platform

# dataset name
DATASET_NAME = 'MNIST_iid'  # Unbalanced, alpha=10.0, Balanced→bbal
# assert DATASET_NAME in ['ml-1m', 'pinterest-20']

sysstr = platform.system()
if sysstr == "Linux":
    USE_GPU = True
    # paths
    ROOT_PATH = '/home/shengy/luoshenseeker/Labs-Federated-Learning/'
elif sysstr == "Windows":
    USE_GPU = False
    # paths
    ROOT_PATH = "C:\\Users\\luoshenseeker\\home\\work\\科研\\Selection\\Labs-Federated-Learning\\"
else:
    USE_GPU = True
    # paths
    ROOT_PATH = '/home/shengy/luoshenseeker/Labs-Federated-Learning/'

# train_rating = main_path + '{}.train.rating'.format(dataset)
# test_rating = main_path + '{}.test.rating'.format(dataset)
# test_negative = main_path + '{}.test.negative'.format(dataset)
#
# model_path = './models/'
# BPR_model_path = model_path + 'NeuMF.pth'
