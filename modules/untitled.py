def batchConv2d(layer,kernel):
    out_dim = kernel.shape[1]
    kernel.contiguous().view([[-1]+list(kernel.shape[2:])])
     # mixing batch size and output dim
    layer =layer.view([1,-1] + list(layer.shape[2:]))
     # mixing batch size and input dim
    layer = F.conv2d(layer, kernel,groups=self.batchsize)
    layer= layer.view(self.batchsize,out_dim,layer.shape[2],x.shape[3])
    #unsqueezing the layer
    return layer