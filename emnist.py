#emnist reader, dear god these people need to work on data orginization

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import string

lets = string.ascii_lowercase

mat = scipy.io.loadmat("emnist-letters.mat")
dat = mat['dataset']

labs = dat[0][0][1][0][0][1].flatten()

dic = {}
for l in string.ascii_lowercase:
    dic[l] = []
    

for i in range(len(dat[0][0][1][0][0][0])):
    im = dat[0][0][1][0][0][0][i].reshape((28,28))
    lab = labs[i]
    dic[lets[(lab-1)]].append(im)


print("ok")
    

