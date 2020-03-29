import scipy.io as scio
import numpy as np
import math
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt


def squashing(x):
    x_norm2 = np.linalg.norm(x)
    if x_norm2 == 0:
        return np.zeros(x.shape)
    return ((x_norm2**2)/x_norm2)*(x/(1+x_norm2**2))


class input_layer():
    def __init__(self,n_inputs, quantiles, squash_flag = False, minus_quantiles = False):
        self.n_inputs = n_inputs
        self.quantiles = quantiles
        self.n_quantiles = quantiles.shape[0]
        self.shape = (self.n_quantiles, self.n_inputs)
        self.squash_flag = squash_flag
        self.minus_quantiles = minus_quantiles

    def forward(self, inputs):
        self.outputs = inputs - self.quantiles
        self.outputs[np.where(self.outputs<0)] = 0
        
        if(self.minus_quantiles):
            for i in range(self.outputs.shape[0]):
                for j in range(self.outputs.shape[1]):
                    if(self.outputs[i][j] == 0):
                        self.outputs[i][j] = self.quantiles[i][j]

        if(self.squash_flag):
            for i in range(inputs.shape[0]):
                self.outputs[:,i] = squashing(self.outputs[:,i])
        
        return self.outputs
    

class hidden_layer():
    def __init__(self, shape, n_size = 3, n_stride = 1, func = np.min, learn_flag = False, squash_flag = False, minus_quantiles = False):
        self.n_size = n_size
        self.n_stride = n_stride
        self.squash_flag = squash_flag
        self.minus_quantiles = minus_quantiles
        self.learn_flag = learn_flag
        n_input = shape[1]
        self.shape = (shape[0],(n_input - self.n_size)//self.n_stride + 1)
        self.n_quantiles = self.shape[0]
        self.n_outputs = self.shape[1]
        self.func = func

        # initialize adjacency
        # self.adjacency = np.repeat(1/self.n_size,self.n_size*self.n_outputs).reshape(self.n_size,self.n_outputs) 
        self.adjacency = np.random.standard_normal((self.n_size,self.n_outputs))


    def forward(self,inputs):
        for i in range(self.n_quantiles):
            self.outputs[:,i] = self.func((self.adjacency[:,i].reshape((1,-1)))*inputs[:,i*self.n_stride:i*self.n_stride + self.n_size],axis = 1)
            # self.outputs[:,i] = np.dot(inputs[:,i*self.n_stride:i*self.n_stride+self.n_size], (self.adjacency[:,i].reshape((-1,1)))).reshape(-1)
        return self.outputs
    
    def dynamic_routing(self, inputs, iterate_times = 3, l_rate = 0.00001,decay_weights = 0.0004):
        # n_input = inputs.shape[1]

        self.outputs = np.zeros((self.n_quantiles, self.n_outputs))
        for r in range(iterate_times):
            for i in range(self.n_outputs):
                # if(i*self.n_stride + self.n_size > n_input-1):
                #     self.outputs[:,i] = np.min((self.adjacency[:,i].reshape((1,-1)))*inputs[:,n_input-self.n_size:],axis = 1) 
                #     # self.outputs[:,i] = np.dot(inputs[:,n_input-self.n_size:], (self.adjacency[:,i].reshape((-1,1)))).reshape(-1)
                # else:
                self.outputs[:,i] = np.min((self.adjacency[:,i].reshape((1,-1)))*inputs[:,i*self.n_stride:i*self.n_stride + self.n_size],axis = 1)
                    # self.outputs[:,i] = np.dot(inputs[:,i*self.n_stride:i*self.n_stride+self.n_size], (self.adjacency[:,i].reshape((-1,1)))).reshape(-1)
                    
                if(self.squash_flag):
                    self.outputs[:,i] = squashing(self.outputs[:,i])
                
                if(self.learn_flag):
                    self.adjacency[:,i] = (1-decay_weights)*self.adjacency[:,i] + l_rate * np.dot(self.outputs[:,i].T,inputs[:,i*self.n_stride:i*self.n_stride+self.n_size]).reshape(-1)
                else:
                    self.adjacency[:,i] += np.dot(self.outputs[:,i].T,inputs[:,i*self.n_stride:i*self.n_stride+self.n_size]).reshape(-1)
                    self.adjacency[:,i] = np.exp(self.adjacency[:,i])/sum(np.exp(self.adjacency[:,i]))

                # self.outputs[:,i] = np.dot(inputs, (self.adjacency[:,i].reshape((-1,1)))).reshape(-1)
                # self.adjacency[:,i] = (1-decay_weights)*self.adjacency[:,i] + l_rate*np.dot(self.outputs[:,i].T,inputs).reshape(-1)
        return self.outputs


def train_dynamic_routing(input_layer,hidden,lassoreg,x_train,y_train, batchsize = 2048, learning_rate = 0.0001, decay_weights = 0.0004):
    error = 0
    n_inputs = batchsize

    n_neurons =  input_layer.n_inputs
    for h in hidden:
        n_neurons += h.n_outputs

    inputs = np.zeros((n_inputs,n_neurons*input_layer.n_quantiles))

    for j in range(x_train.shape[0] // batchsize):
        x_batch = x_train[j*batchsize:(j+1)*batchsize]
        y_batch = y_train[j*batchsize:(j+1)*batchsize]

        for i in range(batchsize):
            new_inputs = input_layer.forward(x_batch[i])

            for h in hidden:
                new_inputs = h.dynamic_routing(new_inputs, iterate_times = 5, l_rate=learning_rate, decay_weights=decay_weights)
            
            lasso_inputs = input_layer.outputs
            for h in hidden:
                lasso_inputs = np.concatenate((lasso_inputs,h.outputs),axis = 1)

            inputs[i,:] = lasso_inputs.T.reshape(-1)
        lassoreg.fit(inputs,y_batch)
        y_predict = lassoreg.predict(inputs).reshape(y_batch.shape)
        error += np.sum((y_batch-y_predict)**2)

    rmse = math.sqrt(error/x_train.shape[0])
    return rmse


def ehh_simulation(input_layer, hidden, lassoreg, u, y, n_delay, u_interval, y_interval):
    length = y.shape[1]
    y_predict = np.zeros((length))
    y_predict[0:n_delay] = y[0,0:n_delay].reshape((n_delay))
    for i in range(n_delay,length):
        y_memory = (y_predict[range(i-1,i-1-n_delay,-1)] - y_interval[0])/(y_interval[1]-y_interval[0])
        u_memory = (u[0,i:i-n_delay:-1] - u_interval[0])/(u_interval[1]-u_interval[0])
        x = np.concatenate((y_memory,u_memory))
        new_inputs = input_layer.forward(x)
        for h in hidden:
            new_inputs = h.forward(new_inputs)

        x = input_layer.outputs
        for h in hidden:
            x = np.concatenate((x,h.outputs),axis = 1)
        
        x = x.T.reshape(1,-1)
        y_predict[i] = lassoreg.predict(x)
    y_predict = np.reshape(y_predict,y.shape)
    return y_predict

def ehh_predict(input_layer, hidden, lassoreg, u, y):
    y_predict = np.zeros(y.shape)
    for i in range(y_predict.shape[0]):
        new_inputs = input_layer.forward(u[i])
        for h in hidden:
            new_inputs = h.forward(new_inputs)
        
        x = input_layer.outputs
        for h in hidden:
            x = np.concatenate((x,h.outputs),axis = 1)
        
        x = x.T.reshape(1,-1)
        y_predict[i] = lassoreg.predict(x)
    y_predict = np.reshape(y_predict,y.shape)
    return y_predict

    
if __name__ == '__main__':
    import load 
    x_train = load.x_train
    y_train = load.y_train
    uval_multisine = load.uval_multisine
    yval_multisine = load.yval_multisine
    uval_sinesweep = load.uval_sinesweep
    yval_sinesweep = load.yval_sinesweep
    u_interval = load.u_interval
    y_interval = load.y_interval

    n_delay = 15
    # quantiles = [0,.45,.6,.8]
    quantiles = [0,.15,.5,.75]
    # 0.15 0.5 .75
    x_quantile = np.quantile(x_train,quantiles,axis = 0)


    in_layer = input_layer(n_delay*2, x_quantile, squash_flag = False, minus_quantiles = False)
    h1 = hidden_layer(in_layer.shape, n_size=5, n_stride=1, func = np.min, learn_flag = True, squash_flag=False)
    h2 = hidden_layer(h1.shape, n_size=5, n_stride=1, func = np.max, learn_flag = True, squash_flag=False)
    h3 = hidden_layer(h2.shape, n_size=3, n_stride=1, func = np.min, learn_flag = True, squash_flag=False)
    h4 = hidden_layer(h3.shape, n_size=3, n_stride=1, func = np.max, learn_flag = True, squash_flag=False)
    

    lassoreg = Lasso(alpha=1e-8, normalize=True, max_iter=1e5)
    # hidden = [h1]
    hidden = [h1,h2,h3,h4]

    l_rate = 1e-4
    decay_weight = 8e-3

    rmse = train_dynamic_routing(in_layer, hidden, lassoreg, x_train, y_train, learning_rate=l_rate, decay_weights=decay_weight)
    print(">train RMSE error=%.5f dB"%(20*math.log10(rmse)))

    ## simulate
    y_simulate = ehh_simulation(in_layer, hidden,lassoreg, uval_multisine, yval_multisine, n_delay, u_interval, y_interval)
    error = np.sum((y_simulate-yval_multisine)**2)
    rmse = math.sqrt(error/(y_simulate.shape[1]-n_delay))

    print(">simulate multisine RMSE error=%.5f dB"%(20*math.log10(rmse)))

    y_simulate = ehh_simulation(in_layer, hidden,lassoreg, uval_sinesweep, yval_sinesweep, n_delay, u_interval, y_interval)
    error = np.sum((y_simulate-yval_sinesweep)**2)
    rmse = math.sqrt(error/(y_simulate.shape[1]-n_delay))

    print(">simulate sinesweep RMSE error=%.5f dB"%(20*math.log10(rmse)))





# if __name__ == '__main__':
#     dir = r'.\EHH\data\sumdata_norm.csv'
#     x_train, y_train, x_test, y_test = readin(dir,type = 'csv')
#     y_max = np.max(y_train)
#     y_min = np.min(y_train)
#     y_train = (y_train-y_min)/(y_max - y_min)
#     dimension = x_train.shape[1]
#     quantiles = [.15,.5,.75]
#     # 0.15 0.5 .75
#     n_hidden = 30

#     x_quantile = np.quantile(x_train,quantiles,axis = 0)
#     x_quantile = np.concatenate((np.zeros((1,dimension)), x_quantile), axis=0)

#     in_layer = input_layer(dimension, x_quantile, squash_flag = False, minus_quantiles = False)
#     h1 = hidden_layer(in_layer.shape, n_size=5, n_stride=1, learn_flag = True, squash_flag=False)
#     h2 = hidden_layer(h1.shape, n_size=3, n_stride=1, learn_flag = True, squash_flag=False)
#     # h3 = hidden_layer(h2.shape, n_size=3, n_stride=1, learn_flag = True, squash_flag=False)
    

#     lassoreg = Lasso(alpha=1e-5, normalize=True, max_iter=1e5)
#     # hidden = [h1]
#     hidden = [h1,h2]

#     l_rate = 0.00000001
#     decay_weights = 0.0008

#     train_rmse = train_dynamic_routing(in_layer,hidden,lassoreg,x_train,y_train, learning_rate=l_rate, decay_weights=decay_weights)


#     y_predict = ehh_predict(in_layer,hidden,lassoreg,x_test,y_test)
#     y_predict = y_predict*(y_max - y_min) + y_min
#     error = np.sum((y_predict-y_test)**2)
#     rmse = math.sqrt(error/len(y_test))

#     # x_axis = np.arange(0,len(y_test))
#     print(">predict RMSE error=%.20f "%(rmse))
    
#     # plt.plot(x_axis, y_test, 'r', label = u'y_test')
#     # plt.plot(x_axis, y_predict, 'b', label = u'y_predict')
#     # plt.legend()
#     # plt.show()

# def readin(dir = r"D:\VSCode_python\EHH\bouc-wen.mat",type = 'mat'):
#     if type == 'mat':
#         data = scio.loadmat(dir)
#         u = data['u']
#         y = data['y']
        

#         uval_multisine = data['uval_multisine']
#         yval_multisine = data['yval_multisine']

#         uval_sinesweep = data['uval_sinesweep']
#         yval_sinesweep = data['yval_sinesweep']

#         return u, y, uval_multisine, yval_multisine, uval_sinesweep, yval_sinesweep
#     elif type == 'csv':
#         data = np.genfromtxt(dir,delimiter=',')
#         data_train = data[0:18736]
#         data_test = data[18736:]

#         # p = np.load("./EHH/permutation.npy")

#         # # p = np.random.permutation(len(data_train))
#         # data_train = data_train[p]
#         # # np.save("./EHH/permutation.npy",p)

#         # np.random.shuffle(data_train)

#         x_train = data_train[:,1:]
#         y_train = data_train[:,0].reshape((-1,1))

#         x_test = data_train[:,1:]
#         y_test = data_test[:,0].reshape((-1,1))

#         return x_train, y_train, x_test, y_test