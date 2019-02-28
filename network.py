import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from modules import ConvLSTMCell,ConvLSTMCellTemp, Sign
import numpy as np
from  utils import batchConv2d
from torch.nn.utils import weight_norm


class EncoderCell(nn.Module):
    def __init__(self):
        super(EncoderCell, self).__init__()

        self.rnn1 = ConvLSTMCellTemp(
            64,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCellTemp(
            256,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCellTemp(
            512,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

    def forward(self, input,conv_w, hidden1, hidden2, hidden3,batchsize):
        init_conv,rnn1_i,rnn1_h,rnn2_i,rnn2_h,rnn3_i,rnn3_h = conv_w
        self.batchsize=batchsize
#        x = batchConv2d(input,init_conv,self.batchsize,stride=2, padding=1, bias=False)
        x= F.conv2d(input,init_conv,stride=2,padding=1)

        hidden1 = self.rnn1(x,rnn1_i,rnn1_h,hidden1,self.batchsize)
        x = hidden1[0]

        hidden2 = self.rnn2(x,rnn2_i,rnn2_h,hidden2,self.batchsize)
        x = hidden2[0]

        hidden3 = self.rnn3(x,rnn3_i,rnn3_h,hidden3,self.batchsize)
        x = hidden3[0]

        return x, hidden1, hidden2, hidden3


class Binarizer(nn.Module):
    def __init__(self):
        super(Binarizer, self).__init__()
        #self.conv = nn.Conv2d(512, 32, kernel_size=1, bias=False)
        self.sign = Sign()

    def forward(self, input,init_conv,batchsize):
        #feat = self.conv(input)
        feat= F.conv2d(input,init_conv,stride=1,padding=0)
        #feat = batchConv2d(input,init_conv,batchsize,stride=1, padding=0, bias=False)
        x = F.tanh(feat)
        return self.sign(x)


class HyperNetwork(nn.Module):

    def __init__(self,num_vids):
        super(HyperNetwork, self).__init__()
        emb_size = num_vids
        emb_dimension= 16

        self.context_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=False)
        initrange = 0.5 / emb_dimension
        self.context_embeddings.weight.data.uniform_(-initrange, initrange)
        # self.z_dim = z_dim
        self.enclayer =   [[64,3,3,3]]+[[1024,64,3,3]]+[[1024,256,1,1]]+[[2048,256,3,3]]+[[2048,512,1,1]]+[[2048,512,3,3]]+[[2048,512,1,1]]
        self.declayer = [[512,32,1,1]]+[[2048,512,3,3]] + [[2048,512,1,1]] +[[2048,128,3,3]] + [[2048,512,1,1]] + [[1024,128,3,3]] + [[1024,256,3,3]] + [[512,64,3,3]] + [[512,128,3,3]] + [[32,3,1,1]]
        self.binlayer=[[32,512,1,1]]
        # layer = np.array(declayer+enclayer+binlayer)
        #total is 18333792
        # self.layer_cum= np.cumsum(layer) 
        #self.layer_cum= np.cumsum(declayer) 


        # f = self.layer_cum[-1]
        #f=2359296
        # out = self.out_size,self.f_size,self.f_size,self.in_size
        # self.linear = weight_norm(nn.Linear(emb_dimension, f, bias=True))
        # self.linear.bias = Parameter(torch.fmod(torch.randn(self.linear.bias.shape),0.1))
        #self.linear.weight_g = Parameter(torch.fmod(torch.randn(self.linear.weight_g.shape),20))
        #self.linear.weight_v = Parameter(torch.fmod(torch.randn(self.linear.weight_v.shape),2))
        #self.linear1 = nn.Linear(32, f, bias=True)
        #self.linear1 = nn.DataParallel(self.linear1)
        #uncomment
        # self.encoderWeights = [Parameter(torch.fmod(torch.randn((emb_dimension, self.total(i) )),2)) for i in self.enclayer]
        # self.decoderWeights = [Parameter(torch.fmod(torch.randn((emb_dimension, self.total(i) )),2)) for i in self.declayer]
        # self.binWeights = [Parameter(torch.fmod(torch.randn((emb_dimension, self.total(i) )),2)) for i in self.binlayer]
        self.encoderWeights = nn.ParameterList([Parameter(torch.nn.init.xavier_normal_(torch.randn((emb_dimension, self.total(i) )),2)) for i in self.enclayer])
        self.decoderWeights = nn.ParameterList([Parameter(torch.nn.init.xavier_normal_(torch.randn((emb_dimension, self.total(i) )),2)) for i in self.declayer])
        self.binWeights = nn.ParameterList([Parameter(torch.nn.init.xavier_normal_(torch.randn((emb_dimension, self.total(i) )),2)) for i in self.binlayer])

        #self.w1 =torch.nn.init.xavier_normal(self.w1)
        self.encoderBias = nn.ParameterList([Parameter(torch.fmod(torch.zeros((self.total(i))),2)) for i in self.enclayer])
        self.decoderBias = nn.ParameterList([Parameter(torch.fmod(torch.zeros((self.total(i))),2)) for i in self.declayer])
        self.binBias = nn.ParameterList([Parameter(torch.fmod(torch.zeros((self.total(i))),2)) for i in self.binlayer])
        #self.b1 =torch.nn.init.xavier_normal(self.b1)

        #self.w2 = Parameter(torch.fmod(torch.randn((h,f)),2))
        #self.b2 = Parameter(torch.fmod(torch.randn((f)),2))
    def total(self,tensor_shape):
        return tensor_shape[0]*tensor_shape[1]*tensor_shape[2]*tensor_shape[3]

    def forward(self,id_num,batchsize):
        self.batchsize= batchsize
        contextEmbed = self.context_embeddings(id_num)
        #h_final= self.linear(contextEmbed)
        #h_final = self.linear1(h_final)
        enc_kernels = [(torch.matmul(contextEmbed,self.encoderWeights[i])  + self.encoderBias[i]).view(self.enclayer[i]) for i in range(len(self.encoderWeights))]
        dec_kernels = [(torch.matmul(contextEmbed,self.decoderWeights[i])  + self.decoderBias[i]).view(self.declayer[i]) for i in range(len(self.decoderWeights))]
        bin_kernels = [(torch.matmul(contextEmbed,self.binWeights[i])  + self.binBias[i]).view(self.binlayer[i]) for i in range(len(self.binWeights))]

        # h_final = self.linear(contextEmbed
        # print(enc_kernels[0].shape)

        return enc_kernels,dec_kernels,bin_kernels[0]



class DecoderCell(nn.Module):
    def __init__(self):
        super(DecoderCell, self).__init__()
        
        ''' self.conv1 = nn.Conv2d(
            32, 512, kernel_size=1, stride=1, padding=0, bias=False)'''
        self.rnn1 = ConvLSTMCellTemp(
            512,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCellTemp(
            128,
            512,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCellTemp(
            128,
            256,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        self.rnn4 = ConvLSTMCellTemp(
            64,
            128,
            kernel_size=3,
            stride=1,
            padding=1,
            hidden_kernel_size=3,
            bias=False)
        '''
        self.conv2 = nn.Conv2d(
            32, 3, kernel_size=1, stride=1, padding=0, bias=False)
        '''


    def forward(self, input,conv_w, hidden1, hidden2, hidden3, hidden4,batchsize):
        self.batchsize=batchsize
        init_conv,rnn1_i,rnn1_h,rnn2_i,rnn2_h,rnn3_i,rnn3_h,rnn4_i,rnn4_h,final_conv = conv_w

        #x = self.conv1(input)
        #x = batchConv2d(input,init_conv,self.batchsize,stride=1, padding=0, bias=False)
        x= F.conv2d(input,init_conv,stride=1,padding=0)
        # conv_w_i,conv_w_h,kernel = conv_w
        # conv_w_i = conv_w_i.contiguous().view([-1,64,conv_w_i.shape[3],conv_w_i.shape[4]])
        # conv_w_h = conv_w_h.contiguous().view([-1,128,conv_w_h.shape[3],conv_w_h.shape[4]])
        # kernel = kernel.contiguous().view([-1,32,kernel.shape[3],kernel.shape[4]])
        hidden1 = self.rnn1(x,rnn1_i,rnn1_h,hidden1,self.batchsize)
        x = hidden1[0]
        x = F.pixel_shuffle(x, 2)

        hidden2 = self.rnn2(x,rnn2_i,rnn2_h, hidden2,self.batchsize)
        x = hidden2[0]
        x = F.pixel_shuffle(x, 2)

        hidden3 = self.rnn3(x,rnn3_i,rnn3_h,hidden3,self.batchsize)
        x = hidden3[0]
        x = F.pixel_shuffle(x, 2)
        # x =x.view(1,-1,x.shape[2],x.shape[3])
        hidden4 = self.rnn4(x,rnn4_i,rnn4_h,hidden4,self.batchsize)
        x = hidden4[0]
        x = F.pixel_shuffle(x, 2)
        # x =x.view(1,-1,x.shape[2],x.shape[3])
        #x = batchConv2d(x,final_conv,self.batchsize,stride=1, padding=0, bias=False)
        x= F.conv2d(x,final_conv,stride=1,padding=0)        
        # x= F.conv2d(x, kernel,groups=self.batchsize)
        # x= x.view(self.batchsize,3,x.shape[2],x.shape[3])
        x = F.tanh(x) / 2

        return x, hidden1, hidden2, hidden3, hidden4

if __name__ == "__main__":
    hp  = HyperNetwork(2)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())

    print("count params ",count_parameters(hp))
    a,b,c = hp(torch.tensor(1),4)
    print([i.shape for i in a],[i.shape for i in b],[i.shape for i in c])

