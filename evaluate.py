from moduleMetric import msssim, psnr
import argparse
import dataset
import network
import torch.utils.data as data
from torch.autograd import Variable
import torch
import numpy as np
import time
import pickle
from unet import UNet
def as_img_array(image):
	# Iutput: [batch_size, depth, height, width]
	# Output: [batch_size, height, width, depth]
	image = image.clip(0, 1) * 255.0
	return image.astype(np.uint8).transpose(0, 2, 3, 1)


def get_ms_ssim(original, compared):
	return msssim(as_img_array(original), as_img_array(compared))


def get_psnr(original, compared):
	return psnr(as_img_array(original), as_img_array(compared))



def save_codes(name, codes):
	print(codes)
	codes = (codes.astype(np.int8) + 1) // 2
	export = np.packbits(codes.reshape(-1))
	np.savez_compressed(
	  name + '.codes',
	  shape=codes.shape,
	  codes=export)


def save_output_images(name, ex_imgs):
	for i, img in enumerate(ex_imgs):
		save_numpy_array_as_image(
		  '%s_iter%02d.png' % (name, i + 1), 
		  img
		)

def evaluate(original, out_imgs):
	ms_ssims = np.array([get_ms_ssim(original, out_img) for out_img in out_imgs])
	psnrs    = np.array([   get_psnr(original, out_img) for out_img in out_imgs])

	return ms_ssims, psnrs




def finish_batch(args, original, out_imgs,
				 losses, code_batch):

	all_losses, all_msssim, all_psnr = [], [], []
  # for ex_idx, filename in enumerate(filenames):
	  # filename = filename.split('/')[-1]
	  # if args.save_codes:
	  #   save_codes(
	  #     os.path.join(args.out_dir, 'codes', filename),
	  #     code_batch[:, ex_idx, :, :, :]
	  #   )

	  # if args.save_out_img:
	  #   save_output_images(
	  #     os.path.join(args.out_dir, 'images', filename),
	  #     out_imgs[:, ex_idx, :, :, :]
	  #   )

	for ex_idx in range(original.shape[0]):
	  msssim, psnr = evaluate(
		original[None, ex_idx], 
		[out_img[None, ex_idx] for out_img in out_imgs])

	  all_losses.append(losses)
	  all_msssim.append(msssim)
	  all_psnr.append(psnr)

	return all_losses, all_msssim, all_psnr




def forward_model(model, data,context, args,iterations):
	with torch.no_grad():
		encoder, binarizer, decoder,unet = model
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
			torch.zeros(batch_size, 512+512, height // 16, width // 16)),
					   Variable(
						   torch.zeros(batch_size, 512+512, height // 16, width // 16)))
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

		losses = []

		res = patches - 0.5
		# context = pickle.load(open("context.p","rb"))
		context = context.cuda()

		context = unet(context)
		# pickle.dump(context,open("context.p","wb"))

		batch_size, _, height, width = res.size()

		original = res.data.cpu().numpy() + 0.5

		out_img = torch.zeros(1, 3, height, width) + 0.5
		out_imgs = []
		losses = []

		codes = []
		prev_psnr = 0.0
		for i in range(iterations):
			encoder_input = res

			# Encode.
			encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
				encoder_input, encoder_h_1, encoder_h_2, encoder_h_3)

			# Binarize.
			code = binarizer(encoded)
			# if args.save_codes:
			#     codes.append(code.data.cpu().numpy())

			output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
				code,context, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4,i)

			res = res - output
			out_img = out_img + output.data.cpu()
			out_img_np = out_img.numpy().clip(0, 1)

			out_imgs.append(out_img_np)
			losses.append(float(res.abs().mean().data.cpu().numpy()))

		return original, np.array(out_imgs), np.array(losses), np.array(codes)



def run_eval(model, eval_loader, args, output_suffix=''):
  all_losses, all_msssim, all_psnr = [], [], []
  i=0
  txtF = open(txtFile,"wb")
  print("size ",len(eval_loader))
  start_time = time.time()
  for batch, (data,context,name) in enumerate(eval_loader):
	  original, out_imgs, losses, code_batch = forward_model(
		  model, data,context, args,args.iterations)

	  losses, msssim, psnr = finish_batch(
		  args,  original, out_imgs, 
		  losses, code_batch)
	  all_msssim += msssim
	  all_losses += losses
	  all_psnr += psnr
	  i+=1

	  summary ='batch {}  MS-SSIM: '.format(i)+ '\t'.join(['%.5f' % el for el in np.array(all_msssim).mean(axis=0).tolist() ]) +"\n"
	  print(summary)
	  # print("length ",losses)
	  txtF.write(summary.encode())
	  txtF.flush()
	  mean_loss = np.array(all_losses).mean(axis=0)
	  total = mean_loss.sum()
	  summary ='batch {}  LOSSES: '.format(i)+ '\t'.join(['%.5f' % el for el in mean_loss.tolist() ]) +"  SUM: "+str(total)+"\n"
	  print(summary)

	  # if i % 10 == 0:
	  print('\tevaluating iter %d (%f seconds)...' % (i, time.time() - start_time))

  return (np.array(all_losses).mean(axis=0),
		  np.array(all_msssim).mean(axis=0),
		  np.array(all_psnr).mean(axis=0))


# Valid_hypertemp2-5-10_15.p
# batch 150  LOSSES: 0.03249	0.01989	0.01568	0.01344	0.01192	0.01079	0.00991	0.00919	0.00857	0.00805	0.00762	0.00723	0.00689	0.00661	0.00638	0.00619  SUM: 0.1808417229028419


# Valid_hypertemp2-5-10_15.p

# batch 19  LOSSES: 0.03281	0.02021	0.01589	0.01363	0.01207	0.01091	0.01001	0.00927	0.00865	0.00812	0.00767	0.00728	0.00695	0.00665	0.00641	0.00621  SUM: 0.18274526705344518

# Valid_hypertemp2-5-10_15.p 

# 	evaluating iter 18 (618.722547 seconds)...
# batch 19  MS-SSIM: 0.90996	0.96095	0.97685	0.98360	0.98767	0.99017	0.99201	0.99312	0.99408	0.99500	0.99538	0.99586	0.99627	0.99657	0.99681	0.99702

# batch 19  LOSSES: 0.03150	0.01976	0.01547	0.01326	0.01172	0.01059	0.00971	0.00899	0.00839	0.00788	0.00744	0.00706	0.00672	0.00643	0.00619	0.00598  SUM: 0.17706911971171696

# 	evaluating iter 19 (646.466217 seconds)...
# Entire Video Loss   : 0.03150	0.01976	0.01547	0.01326	0.01172	0.01059	0.00971	0.00899	0.00839	0.00788	0.00744	0.00706	0.00672	0.00643	0.00619	0.00598
# Entire Video MS-SSIM: 0.90996	0.96095	0.97685	0.98360	0.98767	0.99017	0.99201	0.99312	0.99408	0.99500	0.99538	0.99586	0.99627	0.99657	0.99681	0.99702
# Entire Video PSNR   : 32.03335	34.69995	36.05317	36.89573	37.55232	38.11272	38.59452	39.02461	39.42710	39.79427	40.12641	40.44940	40.75484	41.04394	41.29677	41.51941

# Valid_hypertemp2-5-10_100.p
# SUM: 0.3152646265722713

# 	evaluating iter 123 (4080.221544 seconds)...
# Entire Video Loss   : 0.04456	0.03225	0.02729	0.02425	0.02197	0.02014	0.01865	0.01737	0.01628	0.01532	0.01447	0.01371	0.01303	0.01245	0.01196	0.01155
# Entire Video MS-SSIM: 0.87182	0.93460	0.95721	0.96793	0.97520	0.97960	0.98307	0.98514	0.98697	0.98849	0.98969	0.99071	0.99162	0.99228	0.99284	0.99331
# Entire Video PSNR   : 31.07253	33.30109	34.49793	35.24158	35.84950	36.34562	36.79018	37.16194	37.50922	37.82709	38.12162	38.40588	38.69562	38.95545	39.18675	39.37486


# 0to20
# batch 25  MS-SSIM: 0.91000	0.96090	0.97686	0.98361	0.98766	0.99017	0.99201	0.99311	0.99407	0.99480	0.99537	0.99586	0.99627	0.99656	0.99681	0.99702
# batch 25  LOSSES: 0.03150	0.01977	0.01547	0.01326	0.01172	0.01059	0.00972	0.00899	0.00839	0.00788	0.00745	0.00706	0.00673	0.00643	0.00619	0.00599  SUM: 0.17713135413825512

# 	evaluating iter 25 (840.387107 seconds)...
# Entire Video Loss   : 0.03150	0.01977	0.01547	0.01326	0.01172	0.01059	0.00972	0.00899	0.00839	0.00788	0.00745	0.00706	0.00673	0.00643	0.00619	0.00599
# Entire Video MS-SSIM: 0.91000	0.96090	0.97686	0.98361	0.98766	0.99017	0.99201	0.99311	0.99407	0.99480	0.99537	0.99586	0.99627	0.99656	0.99681	0.99702
# Entire Video PSNR   : 32.02913	34.70246	36.06322	36.90463	37.55933	38.11728	38.59811	39.02586	39.42843	39.79532	40.12656	40.45034	40.75611	41.04529	41.29959	41.52156

def resume(epoch=None):
	if epoch is None:
		s = 'iter'
		epoch = 0
	else:
		s = 'epoch'
	encoder.load_state_dict(
		torch.load('{}/encoder_{}_{:08d}.pth'.format(args.directory,s, epoch)))
	binarizer.load_state_dict(
		torch.load('{}/binarizer_{}_{:08d}.pth'.format(args.directory,s, epoch)))
	decoder.load_state_dict(
		torch.load('{}/decoder_{}_{:08d}.pth'.format(args.directory,s, epoch)))
	unet.load_state_dict(
		torch.load('{}/unet_{}_{:08d}.pth'.format(args.directory,s, epoch)))
	'''
	encoder.load_state_dict(
		torch.load('checkpoint100_new/encoder_temp.pth'.format(s, epoch)))
	binarizer.load_state_dict(
		torch.load('checkpoint100_new/binarizer_temp.pth'.format(s, epoch)))
	decoder.load_state_dict(
		torch.load('checkpoint100_new/decoder_temp.pth'.format(s, epoch)))
	unet.load_state_dict(
		torch.load('checkpoint100_new/unet_temp.pth'.format(s, epoch)))'''
	# encoder.load_state_dict(
	#     torch.load('checkpoint100_new/encoder_{}_{:08d}.pth'.format(s, epoch)))
	# binarizer.load_state_dict(
	#     torch.load('checkpoint100_new/binarizer_{}_{:08d}.pth'.format(s, epoch)))
	# decoder.load_state_dict(
	#     torch.load('checkpoint100_new/decoder_{}_{:08d}.pth'.format(s, epoch)))
	# unet.load_state_dict(
	#     torch.load('checkpoint100_new/unet_{}_{:08d}.pth'.format(s, epoch)))	
	print("loaded")

parser = argparse.ArgumentParser()
parser.add_argument(
	'--batch-size', '-N', type=int, default=1, help='batch size')
parser.add_argument(
	'--eval', '-f', required=True, type=str, help='folder of training images')
parser.add_argument(
	'--max-epochs', '-e', type=int, default=200, help='max epochs')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
# parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument(
	'--iterations', type=int, default=16, help='unroll iterations')
parser.add_argument(
	'--directory', '-d', required=True, type=str, help='directory of checkpoint')
parser.add_argument('--checkpoint', type=int, help='unroll iterations')
parser.add_argument('--evaluatePickle', '-p', required=True, type=str, help='folder of training images')
args = parser.parse_args()

from torchvision import transforms

eval_transform = transforms.Compose([
	transforms.ToTensor()
])
# evaluatePickle = "0to20.p"
evaluatePickle = args.evaluatePickle
txtFile = "resultTxt/"+ evaluatePickle[:-1]+"txt" +"." +args.directory
eval_set = dataset.ImageFolder(root=args.eval,file_name = evaluatePickle ,train=False)

eval_loader = data.DataLoader(
	dataset=eval_set, batch_size=args.batch_size, shuffle=True, num_workers=0)

print('total images: {}; total batches: {}'.format(
	len(eval_set), len(eval_loader)))



encoder = network.EncoderCell().cuda()
binarizer = network.Binarizer().cuda()
decoder = network.DecoderCell().cuda()
unet = UNet(9,1).cuda()

encoder.eval()
binarizer.eval()
decoder.eval()
unet.eval()

encoder = encoder.cuda()
binarizer = binarizer.cuda()
decoder = decoder.cuda()
unet = unet.cuda()

model = [encoder, binarizer, decoder,unet]
resume()
if args.checkpoint:
	resume(args.checkpoint)

eval_loss, mssim, psnr = run_eval(model, eval_loader, args, output_suffix='')    

eval_name = "Entire Video"
print('%s Loss   : ' % eval_name+ '\t'.join(['%.5f' % el for el in eval_loss.tolist()]))
print('%s MS-SSIM: ' % eval_name+ '\t'.join(['%.5f' % el for el in mssim.tolist()]))
print('%s PSNR   : ' % eval_name+ '\t'.join(['%.5f' % el for el in psnr.tolist()]))