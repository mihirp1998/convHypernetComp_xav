import torch.nn.functional as F

def batchConv2d(layer,kernel,batchsize,stride,padding,bias,dilation=1):
    out_dim = kernel.shape[1]
    kernel = kernel.contiguous().view([-1]+list(kernel.shape[2:]))
     # mixing batch size and output dim
    layer =layer.view([1,-1] + list(layer.shape[2:]))
     # mixing batch size and input dim
    #print(layer.shape,kernel.shape,"check") 
    layer = F.conv2d(layer, kernel,groups=batchsize,stride=stride,padding=padding,dilation=dilation)
    layer= layer.view(batchsize,out_dim,layer.shape[2],layer.shape[3])
    #unsqueezing the layer
    return layer