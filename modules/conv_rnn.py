import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from  utils import batchConv2d
import torch.nn.functional as F

class ConvRNNCellBase(nn.Module):
    def __repr__(self):
        s = (
            '{name}({input_channels}, {hidden_channels}, kernel_size={kernel_size}'
            ', stride={stride}')
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        s += ', hidden_kernel_size={hidden_kernel_size}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class ConvLSTMCell(ConvRNNCellBase):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 hidden_kernel_size=1,
                 bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.hidden_kernel_size = _pair(hidden_kernel_size)

        hidden_padding = _pair(hidden_kernel_size // 2)

        gate_channels = 4 * self.hidden_channels
        self.conv_ih = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=gate_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias)

        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=gate_channels,
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            dilation=1,
            bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_ih.reset_parameters()
        self.conv_hh.reset_parameters()

    def forward(self, input, hidden):
        hx, cx = hidden
        gates = self.conv_ih(input) + self.conv_hh(hx)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy


class ConvLSTMCellTemp(ConvRNNCellBase):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 hidden_kernel_size=1,
                 bias=True):
        super(ConvLSTMCellTemp, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.bias=bias
        self.hidden_kernel_size = _pair(hidden_kernel_size)

        self.hidden_padding = _pair(hidden_kernel_size // 2)

        gate_channels = 4 * self.hidden_channels
        '''
        self.conv_ih = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=gate_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias)

        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=gate_channels,
            kernel_size=hidden_kernel_size,
            stride=1,
            padding=hidden_padding,
            dilation=1,
            bias=bias)
        '''    

    def forward(self, input ,conv_w_i, conv_w_h, hidden,batchsize):
        hx, cx = hidden
        #hx =hx.view(1,-1,hx.shape[2],hx.shape[3])
        #print(conv_w_i.shape,"weird")
        # gates = self.conv_ih(input) + self.conv_hh(hx)
       # print("hx",input.shape,hx.shape)
        #gate_input =  F.conv2d(input, conv_w_i, groups=self.batchsize,stride=self.stride,padding=self.padding)
        #gate_hidden =  F.conv2d(hx, conv_w_h, groups=self.batchsize,stride=1,padding=self.hidden_padding)
        #print("gate",gate_input.shape,gate_hidden.shape)
        #gate_input = gate_input.view(self.batchsize,512,gate_input.shape[2],gate_input.shape[3])
        #gate_hidden = gate_hidden.view(self.batchsize,512,gate_hidden.shape[2],gate_hidden.shape[3])
        #print(gate_input.shape,gate_hidden.shape)
        
        gate_input =  F.conv2d(input,conv_w_i,stride=self.stride,padding=self.padding)
        gate_hidden = F.conv2d(hx,conv_w_h,stride=1,padding=self.hidden_padding)

        #gate_input= batchConv2d(input,conv_w_i,batchsize,stride=self.stride,padding=self.padding,dilation=self.dilation,bias=self.bias)
        #gate_hidden= batchConv2d(hx,conv_w_h,batchsize,stride=1,padding=self.hidden_padding,dilation=1,bias=self.bias)

        gates = gate_input + gate_hidden

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)

        return hy, cy

import pickle
import random
'''
class HyperConvLSTMCell(ConvRNNCellBase):
    def __init__(self,input_channels,main_num_units,hyper_unit,context_input_channels,stride=1,hyper_embedding = 512):
        super(HyperConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.num_units = main_num_units
        self.hyper_num_unit = hyper_unit
        self.hyper_embedding =hyper_embedding
        self.gate_params  = self.num_units * 4
        self.context_input_channels = context_input_channels
        # self.hyper_input_units = context_input_channels+ main_num_units  
        self.hyper_input_units = context_input_channels 
        self.stride = _pair(stride)

 
        # print(self.hyper_input_units,self.hyper_num_unit)

        self.hyper_cell = ConvLSTMCell(
            self.hyper_input_units,
            self.hyper_num_unit,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            hidden_kernel_size=1,
            bias=False)
        
        self.hyper_cell = self.hyper_cell

#         self.temp_conv = nn.Conv2d(
#                     input_channels , 
#                     input_channels, 
#                     kernel_size=3, stride=2, padding=1, bias=False)
#         self.temp_conv = self.temp_conv

        self.conv_z_input  = nn.Conv2d(self.hyper_num_unit, self.hyper_embedding, 
                        kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_z_state  = nn.Conv2d(self.hyper_num_unit, self.hyper_embedding, 
                        kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_alpha_input  = nn.Conv2d(self.hyper_embedding , self.gate_params, 
                    kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_alpha_state  = nn.Conv2d(self.hyper_embedding , self.gate_params, 
                    kernel_size=1, stride=1, padding=0, bias=False)

        
        
        self.conv_transform_gates_input  = nn.Conv2d(self.input_channels , self.gate_params, 
                    kernel_size=3, stride=self.stride, padding=1, bias=False)

        self.conv_transform_gates_states  = nn.Conv2d(self.num_units , self.gate_params, 
                    kernel_size=1, stride=1, padding=0, bias=False)

        #self.alphas_s = []
        #self.alphas_s = pickle.load(open("alphas_s.p","rb"))
        #self.alphas_i = []
        #self.alphas_i =  pickle.load(open("alphas_i.p","rb"))   



        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        self.hyper_cell.reset_parameters()
        self.conv_z_input.reset_parameters()
        self.conv_z_state.reset_parameters()
        self.conv_alpha_input.reset_parameters()
        self.conv_alpha_state.reset_parameters()
        self.conv_transform_gates_input.reset_parameters()
        self.conv_transform_gates_states.reset_parameters()

    def hyper_norm_input(self,input_layer,hyper_h,i):
        zw = self.conv_z_input(hyper_h)
        alpha = self.conv_alpha_input(zw)
        # pickle.dump(alpha,open("pit3_i.p","wb"))
        # self.alphas_i.append(alpha)
        # alpha = self.alphas_i[i]
        # if i == 15:
        #     pickle.dump(self.alphas_i,open("alphas_i.p","wb"))

        result = input_layer * alpha
        
        return result

    
    def hyper_norm_state(self,input_layer,hyper_h,i):
        zw = self.conv_z_state(hyper_h)
        
        alpha = self.conv_alpha_state(zw)
        # alpha = self.alphas_s[i]
        # pickle.dump(alpha,open("pit3_s.p","wb"))
        # self.alphas_s.append(alpha)
        # if i == 15:
        #     pickle.dump(self.alphas_s,open("alphas_s.p","wb"))

        result = input_layer * alpha
        
        return result    
    # def reset_parameters(self):
    #     self.conv_ih.reset_parameters()
    #     self.conv_hh.reset_parameters()

    def forward(self, input,context, hidden,i):
        h,c = hidden
        main_h = h[:,:self.num_units]
        main_c = c[:,:self.num_units]
        hyper_h = h[:,self.num_units:]
        hyper_c = h[:,self.num_units:]
        # print("input shape ",input.shape)
        hyper_states = (hyper_h,hyper_c)
        # if self.encoder:
        # print(context.shape,"context")
        #     input = self.temp_conv(input)
        # hyper_input = torch.cat([context,main_h],dim=1)
        hyper_input = context
        # print("hyper shape ",hyper_input.shape,hyper_states[0].shape)

        #print(hyper_input.shape,hyper_states[0].shape)
        hyper_h,hyper_c = self.hyper_cell(hyper_input,hyper_states)

        input_below_ = self.conv_transform_gates_input(input)
        input_below_ = self.hyper_norm_input(input_below_,hyper_h,i)

        state_below_ = self.conv_transform_gates_states(main_h)
        state_below_ = self.hyper_norm_state(state_below_,hyper_h,i)

        lstm_matrix = input_below_ + state_below_

        i,j,f,o = lstm_matrix.chunk(4,1)


        new_main_c = (self.sigmoid(f)*main_c) + (self.sigmoid(i)*self.tanh(j))
        new_main_h = self.tanh(new_main_c)* self.sigmoid(o)

        new_total_h =torch.cat([new_main_h,hyper_h],dim=1)

        new_total_c = torch.cat([new_main_c,hyper_c],dim=1)


        return (new_total_h,new_total_c),new_main_h'''