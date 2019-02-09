import time
import os
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms
from torch.nn.parameter import Parameter
from unet import UNet,Feedforward
parser = argparse.ArgumentParser()
parser.add_argument(
    '--batch-size', '-N', type=int, default=128, help='batch size')
parser.add_argument(
    '--train', '-f', required=True, type=str, help='folder of training images')
parser.add_argument(
    '--max-epochs', '-e', type=int, default=200, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
# parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument(
    '--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
args = parser.parse_args()

## load 32x32 patches from images
import dataset

train_transform = transforms.Compose([
    transforms.RandomCrop((32, 32)),
    transforms.ToTensor(),
])

train_set = dataset.ImageFolder(root=args.train,train=True,file_name ="outValid15_100Vids.p")

train_loader = data.DataLoader(
    dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

print('total images: {}; total batches: {}'.format(
    len(train_set), len(train_loader)))

# >>> 512*18*22
# 202752
## load networks on GPU
import network
hypernet = network.HyperNetwork(train_set.vid_count).cuda()
encoder = network.EncoderCell().cuda()
binarizer = network.Binarizer().cuda()
decoder = network.DecoderCell().cuda()
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("hypernet ",count_parameters(hypernet))    
print("encoder ",count_parameters(encoder))    
print("decoder ",count_parameters(decoder))    
print("binarizer ",count_parameters(binarizer))    

#ff = Feedforward(202752).cuda()
#unet = UNet(9,1).cuda()

solver = optim.Adam(
    [
        {
            'params': encoder.parameters()
        },
        {
            'params': binarizer.parameters()
        },
        {
            'params': decoder.parameters()
        },
        {
            'params': hypernet.parameters()
        }
    ],
    lr=args.lr)



def resume(epoch=None):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'
    print("Loaded")
    '''
    encoder.load_state_dict(
        torch.load('checkpoint100_new/encoder_temp.pth'.format(s, epoch)))
    binarizer.load_state_dict(
        torch.load('checkpoint100_new/binarizer_temp.pth'.format(s, epoch)))
    decoder.load_state_dict(
        torch.load('checkpoint100_new/decoder_temp.pth'.format(s, epoch)))
    unet.load_state_dict(
        torch.load('checkpoint100_new/unet_temp.pth'.format(s, epoch)))
        '''
    encoder.load_state_dict(
        torch.load('checkpoint100_small/encoder_{}_{:08d}.pth'.format(s, epoch)))
    binarizer.load_state_dict(
        torch.load('checkpoint100_small/binarizer_{}_{:08d}.pth'.format(s, epoch)))
    decoder.load_state_dict(
        torch.load('checkpoint100_small/decoder_{}_{:08d}.pth'.format(s, epoch)))
    #unet.load_state_dict(
     #   torch.load('checkpoint100_small/unet_{}_{:08d}.pth'.format(s, epoch)))
    #ff.load_state_dict(
     #   torch.load('checkpoint100_small/ff_{}_{:08d}.pth'.format(s, epoch)))
    '''
    hypernet_dict = hypernet.state_dict()
    pretrain_hypernet = torch.load('checkpoint100_small/hypernet_{}_{:08d}.pth'.format(s, epoch))
    pretrain_hypernet= {k: v for k, v in pretrain_hypernet.items() if k in hypernet_dict}
    hypernet_dict.update(pretrain_hypernet)
    print("non common keys ",[i for i in hypernet_dict.keys() if i not in pretrain_hypernet.keys()])
    hypernet.load_state_dict(hypernet_dict)
    '''
    hypernet.load_state_dict(
        torch.load('checkpoint100_small/hypernet_{}_{:08d}.pth'.format(s, epoch)))

def save(index, epoch=True):
    if not os.path.exists('checkpoint100_small'):
        os.mkdir('checkpoint100_small')

    if epoch:
        s = 'epoch'
    else:
        s = 'iter'

    torch.save(encoder.state_dict(), 'checkpoint100_small/encoder_{}_{:08d}.pth'.format(s, index))

    torch.save(binarizer.state_dict(),'checkpoint100_small/binarizer_{}_{:08d}.pth'.format(s, index))

    torch.save(decoder.state_dict(), 'checkpoint100_small/decoder_{}_{:08d}.pth'.format(s, index))

    #torch.save(unet.state_dict(), 'checkpoint100_small/unet_{}_{:08d}.pth'.format(s, index))    
    #torch.save(ff.state_dict(), 'checkpoint100_small/ff_{}_{:08d}.pth'.format(s, index))    
    torch.save(hypernet.state_dict(), 'checkpoint100_small/hypernet_{}_{:08d}.pth'.format(s, index))   

#
#resume()

scheduler = LS.MultiStepLR(solver, milestones=[2, 3, 20, 50, 100], gamma=0.5)

last_epoch = 0
if args.checkpoint:
    resume(args.checkpoint)
    last_epoch = args.checkpoint
    scheduler.last_epoch = last_epoch - 1

  
# context = None
vepoch=0
for epoch in range(last_epoch + 1, args.max_epochs + 1):

    scheduler.step()

    for batch, (data,id_num,name) in enumerate(train_loader):
        batch_t0 = time.time()

        ## init lstm state
        batch_size, input_channels, height, width = data.size()

        encoder_h_1 = (Variable(
            torch.zeros(batch_size, 256, height // 4, width // 4)),
                       Variable(
                           torch.zeros(batch_size, 256, height // 4, width // 4)))
        encoder_h_2 = (Variable(
            torch.zeros(batch_size, 512, height // 8, width // 8)),
                       Variable(
                           torch.zeros(batch_size, 512, height // 8, width // 8)))
        encoder_h_3 = (Variable(
            torch.zeros(batch_size, 512, height // 16, width // 16)),
                       Variable(
                           torch.zeros(batch_size, 512, height // 16, width // 16)))

        decoder_h_1 = (Variable(
            torch.zeros(batch_size, 512, height // 16, width // 16)),
                       Variable(
                           torch.zeros(batch_size, 512, height // 16, width // 16)))
        decoder_h_2 = (Variable(
            torch.zeros(batch_size, 512, height // 8, width // 8)),
                       Variable(
                           torch.zeros(batch_size, 512, height // 8, width // 8)))
        decoder_h_3 = (Variable(
            torch.zeros(batch_size, 256, height // 4, width // 4)),
                       Variable(
                           torch.zeros(batch_size, 256, height // 4, width // 4)))
        decoder_h_4 = (Variable(
            torch.zeros(batch_size, 128, height // 2, width // 2)),
                       Variable(
                           torch.zeros(batch_size, 128, height // 2, width // 2)))


        encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
        encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
        encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

        decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
        decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
        decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
        decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())
        
        patches = Variable(data.cuda())

        solver.zero_grad()

        losses = []

        res = patches - 0.5
        id_num = Variable(id_num.cuda())

        #context = context.cuda()

        #encodedContext = unet(context)
        #encodedContext=  encodedContext.view(args.batch_size,-1)
        #print("encodedContext ",encodedContext.shape)
        #encodedContext =  ff(encodedContext)
        wenc,wdec,wbin = hypernet(id_num,batch_size)
        bp_t0 = time.time()

        for i in range(args.iterations):
            encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                res,wenc,encoder_h_1, encoder_h_2, encoder_h_3,batch_size)

            codes = binarizer(encoded,wbin,batch_size)

            output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                codes,wdec, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4,batch_size)
            res = res - output
            losses.append(res.abs().mean())
       
        bp_t1 = time.time()

        loss = sum(losses) / args.iterations

        loss.backward()

        solver.step()


        batch_t1 = time.time()

        print(
            '[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.
            format(epoch, batch + 1,
                   len(train_loader), loss.data[0], bp_t1 - bp_t0, batch_t1 -
                   batch_t0))
        print(('{:.4f} ' * args.iterations +
               '\n').format(* [l.data[0] for l in losses]))

        index = (epoch - 1) * len(train_loader) + batch
        if index % 10000 == 0 and index != 0:
            vepoch+=1
            save(vepoch)
            #print("scheduled")
            scheduler.step()

        ## save checkpoint every 500 training steps
        if index % 150 == 0 and index != 0:
            save(0, False)

    save(epoch)