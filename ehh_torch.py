import torch as t
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import torch.nn.functional as F

class conv_hh(t.nn.Module):
    def __init__(self, n_inputs, n_quantiles, patch_size, stride, in_depth, out_depth, func='min', bias=True):
        super(conv_hh, self).__init__()
        self.n_quantiles = n_quantiles
        self.out_depth = out_depth
        self.stride = stride
        self.patch_size = patch_size
        self.n_outputs = (n_inputs - patch_size)//stride + 1
        self.func = func
        self.weights = t.nn.Parameter(t.randn(self.n_outputs, n_quantiles, patch_size, in_depth, out_depth))
        if bias:
            self.bias = t.nn.Parameter(t.randn(self.n_outputs, in_depth, out_depth))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        batch_size, _, n_inputs, in_depth = x.size()
        ret = t.Tensor(batch_size, self.n_quantiles, self.n_outputs, self.out_depth)
        for b in range(batch_size):
            for n in range(self.n_outputs):
                for out_d in range(self.out_depth):
                    tmp = t.sum(x[b, :, n*self.stride:n*self.stride+self.patch_size, :]*self.weights[n,:,:,:,out_d]+self.bias[n,:,out_d],dim=2)
                    if self.func == 'min':
                        ret[b, :, n, out_d] = t.min(tmp, dim=1)[0]
                    elif self.func == 'max':
                        ret[b, :, n, out_d] = t.max(tmp, dim=1)[0]
        return ret

class ehh_torch(t.nn.Module):
    def __init__(self,quantiles):
        super(ehh_torch, self).__init__()
        self.quantiles = quantiles
        self.n_quantiles = quantiles.size()[0]
        self.con_hh1 = conv_hh(
            n_inputs=30, n_quantiles=self.n_quantiles, patch_size=3, stride=1, 
            in_depth=1, out_depth=2, func='min')

        self.con_hh2 = conv_hh(
            n_inputs=self.con_hh1.n_outputs, n_quantiles=self.n_quantiles, patch_size=3, stride=1, 
            in_depth=self.con_hh1.out_depth, out_depth=2, func='max')

        self.linear_model1 = t.nn.Linear(138,138)
        self.linear_model2 = t.nn.Linear(138,1)
    
    def forward(self, x):
        def source_output(x):
            batch_size, n_inputs = x.size()
            ret = t.Tensor(batch_size, self.n_quantiles, n_inputs, 1)
            for b in range(batch_size):
                ret[b,:,:,0] = t.where((x[b]-self.quantiles)>0,(x[b]-self.quantiles), t.zeros_like(self.quantiles))
            return ret
        
        batch_size, _ = x.size()
        src_out = source_output(x)
        linear_input = t.mean(src_out, 1).view(batch_size, -1)
        # linear_input = src_out.view(batch_size,-1)
        con_hh_ret = self.con_hh1.forward(src_out)
        linear_input = t.cat((linear_input, t.mean(con_hh_ret, 1).view(batch_size, -1)), 1)

        con_hh_ret = self.con_hh2.forward(con_hh_ret)
        linear_input = t.cat((linear_input, t.mean(con_hh_ret, 1).view(batch_size, -1)), 1)
        
        l1= F.relu(self.linear_model1(linear_input))
        y_predict = self.linear_model2(l1)
        return y_predict




# class ehh_torch(t.nn.Module):
#     def __init__(self, n_inputs, quantiles):
#         super(ehh_torch, self).__init__()
#         self.n_inputs = n_inputs
#         self.quantiles = quantiles
#         self.n_quantiles = quantiles.size()[0]
#         self.conv_weights = []
#         self.conv_biases = []
#         self.conv_config = []
        

#         self.n_hidden = None
    
#     def add_conv(self,n_inputs, patch_size, stride, in_depth, out_depth):
#         n_outputs = (n_inputs - patch_size)//stride + 1
#         # weights = Variable(t.randn(n_outputs, self.n_quantiles, patch_size, in_depth, out_depth), requires_grad = True)
#         # bias = Variable(t.randn(n_outputs, in_depth, out_depth), requires_grad = True)
#         weights = t.nn.Parameter(t.randn(n_outputs, self.n_quantiles, patch_size, in_depth, out_depth), requires_grad=True)
#         bias = t.nn.Parameter(t.randn(n_outputs, in_depth, out_depth), requires_grad=True)
#         self.conv_weights.append(weights)
#         self.conv_biases.append(bias)
#         self.conv_config.append({
#             'n_outputs':n_outputs,
#             'patch_size':patch_size,
#             'in_depth':in_depth,
#             'out_depth':out_depth,
#             'stride':stride
#         }
#         )
    

#     def forward(self, x):
#         def forward_conv(x, weight, bias, config):
#             batch_size, _, n_inputs, in_depth = x.size()
#             ret = t.Tensor(batch_size, self.n_quantiles, config['n_outputs'], config['out_depth'])
#             for b in range(batch_size):
#                 for n in range(config['n_outputs']):
#                     for out_d in range(config['out_depth']):
#                         tmp = t.sum(x[b, :, n*config['stride']:n*config['stride']+config['patch_size'],:]*weight[n,:,:,:,out_d]+bias[n,:,out_d],dim=2)
#                         ret[b, :, n, out_d] = t.min(tmp, dim=1)[0]
#             return ret

#         def source_output(x, quantiles):
#             batch_size, n_inputs = x.size()
#             n_quantiles = quantiles.size()[0]
#             ret = t.Tensor(batch_size, n_quantiles, n_inputs,1)
#             for b in range(batch_size):
#                 ret[b,:,:,0] = t.where((x[b]-quantiles)>0,(x[b]-quantiles), t.zeros_like(quantiles))
#             return ret

#         batch_size = x.size()[0]
#         x = source_output(x, self.quantiles)
#         accumulate_ret = x.view(batch_size,-1)
#         for weight, bias, config in zip(self.conv_weights, self.conv_biases, self.conv_config):
#             x = forward_conv(x, weight, bias, config)
#             accumulate_ret = t.cat((accumulate_ret,x.view(batch_size,-1)),1)
        
#         linear_model = t.nn.Linear(accumulate_ret.size()[1],1)
#         y_predict = linear_model(accumulate_ret)

#         return y_predict
    
        

        

        
