import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from modules import ConvLSTMCell,ConvLSTMCellTemp, Sign
import numpy as np
from  utils import batchConv2d

class EncoderCell(nn.Module):
    def __init__(self):
        super(EncoderCell, self).__init__()

        self.conv = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1, bias=False)

        #self.hyper1 = HyperConvLSTMCell(64,256,256,128,stride=2)
        self.rnn1 = ConvLSTMCell(
            64,
            256,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn2 = ConvLSTMCell(
            256,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        self.rnn3 = ConvLSTMCell(
            512,
            512,
            kernel_size=3,
            stride=2,
            padding=1,
            hidden_kernel_size=1,
            bias=False)

    def forward(self, input, hidden1, hidden2, hidden3):
        x = self.conv(input)

        hidden1 = self.rnn1(x, hidden1)
        x = hidden1[0]

        hidden2 = self.rnn2(x, hidden2)
        x = hidden2[0]

        hidden3 = self.rnn3(x, hidden3)
        x = hidden3[0]

        return x, hidden1, hidden2, hidden3


class Binarizer(nn.Module):
    def __init__(self):
        super(Binarizer, self).__init__()
        self.conv = nn.Conv2d(512, 32, kernel_size=1, bias=False)
        self.sign = Sign()

    def forward(self, input):
        feat = self.conv(input)
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
        enclayer =   np.array([64*3*3*3]+[1024*64*3*3]+[1024*256*1*1]+[2048*256*3*3]+[2048*512*1*1]+[2048*512*3*3]+[2048*512*1*1])
        declayer = np.array([512*32*1*1]+[2048*512*3*3] + [2048*512*1*1] +[2048*128*3*3] + [2048*512*1*1] + [1024*128*3*3] + [1024*256*3*3] + [512*64*3*3] + [512*128*3*3] + [32*3*1*1])
        #total is 18333792
        self.enclayer_cum= np.cumsum(enclayer) 
        self.declayer_cum= np.cumsum(declayer) 

        f = self.enclayer_cum[-1] + self.declayer_cum[-1]
        #f=2359296
        # out = self.out_size*self.f_size*self.f_size*self.in_size
        self.w1 = Parameter(torch.fmod(torch.randn((emb_dimension, f)),2))
        self.b1 = Parameter(torch.fmod(torch.zeros((f)),2))

        # self.w2 = Parameter(torch.fmod(torch.randn((h,f)),2))
        # self.b2 = Parameter(torch.fmod(torch.randn((f)),2))

    def forward(self,id_num,batchsize):
        self.batchsize= batchsize
        contextEmbed = self.context_embeddings(id_num)

        h_final = torch.matmul(contextEmbed, self.w1) 

        init_conv = h_final[:,:self.declayer_cum[0]]
        init_conv = init_conv.view(self.batchsize,512,32,1,1)
        #print("datatype",init_conv.dtype)
        rnn1_i = h_final[:,self.declayer_cum[0]:self.declayer_cum[1]]
        #print(rnn1_i.shape)
        rnn1_i = rnn1_i.view(self.batchsize,2048,512,3,3)
        
        rnn1_h = h_final[:,self.declayer_cum[1]:self.declayer_cum[2]]
        rnn1_h = rnn1_h.view(self.batchsize,2048,512,1,1)

        rnn2_i = h_final[:,self.declayer_cum[2]:self.declayer_cum[3]]
        rnn2_i = rnn2_i.view(self.batchsize,2048,128,3,3)

        rnn2_h = h_final[:,self.declayer_cum[3]:self.declayer_cum[4]]
        rnn2_h = rnn2_h.view(self.batchsize,2048,512,1,1)

        rnn3_i = h_final[:,self.declayer_cum[4]:self.declayer_cum[5]]
        rnn3_i = rnn3_i.view(self.batchsize,1024,128,3,3)

        rnn3_h = h_final[:,self.declayer_cum[5]:self.declayer_cum[6]]
        rnn3_h = rnn3_h.view(self.batchsize,1024,256,3,3)

        rnn4_i = h_final[:,self.declayer_cum[6]:self.declayer_cum[7]]
        rnn4_i = rnn4_i.view(self.batchsize,512,64,3,3)

        rnn4_h = h_final[:,self.declayer_cum[7]:self.declayer_cum[8]]
        rnn4_h = rnn4_h.view(self.batchsize,512,128,3,3)

        final_conv = h_final[:,self.declayer_cum[8]:self.declayer_cum[9]]
        final_conv = final_conv.view(self.batchsize,3,32,1,1)


        

        return [init_conv,rnn1_i,rnn1_h,rnn2_i,rnn2_h,rnn3_i,rnn3_h,rnn4_i,rnn4_h,final_conv]



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
        x = batchConv2d(input,init_conv,self.batchsize,stride=1, padding=0, bias=False)
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
        x = batchConv2d(x,final_conv,self.batchsize,stride=1, padding=0, bias=False)
        # x= F.conv2d(x, kernel,groups=self.batchsize)
        # x= x.view(self.batchsize,3,x.shape[2],x.shape[3])
        x = F.tanh(x) / 2

        return x, hidden1, hidden2, hidden3, hidden4