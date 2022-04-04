import pickle

fr=open('cifar10_balance_dir.pkl', 'rb')
pkl_contents = pickle.load(fr)
print(pkl_contents)
