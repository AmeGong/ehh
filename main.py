from ehh_torch import ehh_torch
import load
import numpy as np
import torch as t
from torch.autograd import Variable
import torch.utils.data as Data
import math


x_train = load.x_train
y_train = load.y_train
quantiles = np.quantile(x_train,[0,.15,.5,.75], axis = 0)
x_train = t.from_numpy(x_train)
y_train = t.from_numpy(y_train).float()
y_train = t.unsqueeze(y_train,dim=1)
quantiles = t.from_numpy(quantiles)

torch_dataset = Data.TensorDataset(x_train,y_train)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=64,
    shuffle = True)


net = ehh_torch(quantiles)
optimizer = t.optim.Adam(net.parameters(),lr=0.015) 
loss = t.nn.MSELoss()

for epoch in range(10):
    for step,(batch_x, batch_y) in enumerate(loader):
        y_predict = net.forward(batch_x)
        output = t.sqrt(loss(y_predict, batch_y))
        all_linear1_params = t.cat([x.view(-1) for x in net.linear_model1.parameters()])
        l1_regularization = 0.5 * t.norm(all_linear1_params, 1)
        output = output + l1_regularization
        optimizer.zero_grad() 
        output.backward()
        optimizer.step()
        if step%20 == 0:
            print(">%d epoch, %d batch error is %.10fdB" %(epoch,step,20*math.log10(t.sqrt(loss(y_predict, batch_y)).data.numpy())))
