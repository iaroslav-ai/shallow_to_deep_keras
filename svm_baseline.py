import numpy as np

from datasets import abalone_dataset, chess_dataset

#Tr, Val, Ts = abalone_dataset([0.5,0.25,0.25])
Tr, Val, Ts = chess_dataset([0.5,0.25,0.25])

from sklearn.svm import SVR


def eval(clf, dat):
    Yp = clf.predict(dat["X"])
    err = np.mean( np.abs( Yp - dat["Y"] ) )
    return err 

bval = 10000;
res = []

for c in [0.01, 0.1, 1,10,100]:
    for eps in [0.5, 0.1, 0.05, 0.01]:
        clf = SVR(C=c, epsilon=eps)
        clf.fit(Tr["X"], Tr["Y"])
        val = eval(clf, Val)
        if val < bval:
            res = [c, eps];
            print val, res;
            bval = val;

clf = SVR(C=c, epsilon=eps)
clf.fit(Tr["X"], Tr["Y"])
print eval(clf, Ts)