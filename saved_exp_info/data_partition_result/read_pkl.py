import pickle

fr=open('MNIST_iid.pkl','rb')
pkl_contents = pickle.load(fr)
print(pkl_contents)
print(len(pkl_contents))

