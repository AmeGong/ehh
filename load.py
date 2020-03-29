import scipy.io as scio
import numpy as np
import math

data = scio.loadmat(r"D:\VSCode_python\EHH\bouc-wen.mat")

uval_multisine = data['uval_multisine']
yval_multisine = data['yval_multisine']

uval_sinesweep = data['uval_sinesweep']
yval_sinesweep = data['yval_sinesweep']

u = data['u'].T
y = data['y']

u_interval = (np.min(u,axis=1),np.max(u,axis=1))
y_interval = (np.min(y,axis=1),np.max(y,axis=1))

u = np.reshape(u,(5,8192))
y = np.reshape(y,(5,8192))

u_use = u[3,:].reshape((1,-1))
y_use = y[3,:].reshape((1,-1))

n_delayed = 15

def reformat(x,y,n_delayed):
    length = x.shape[1]
    y_train = np.zeros(length-n_delayed)
    x_train = np.zeros((length-n_delayed,2*n_delayed))
    for i in range(n_delayed,length):
        y_delayed = (y[0,range(i-1,i-1-n_delayed,-1)]-y_interval[0])/(y_interval[1]-y_interval[0])
        u_delayed = (u[0,i:i-n_delayed:-1]-u_interval[0])/(u_interval[1]-u_interval[0])
        x_train[i-n_delayed,:] = np.concatenate((y_delayed,u_delayed))
        y_train[i-n_delayed] = y[0,i]
    return x_train, y_train

x_train,y_train = reformat(u_use, y_use, n_delayed)

