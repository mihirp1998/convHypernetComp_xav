import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
a = open("timedresults.txt").read()
b = a.split("\n")
l = [i.split("\t") for i in b]
print(l)

mssim = np.array(l,dtype=np.float)
# lstm_ssim = np.mean(lstm_ssim, axis=0)
bpp = np.arange(1, 17) / 192 * 24
aucVal = [auc(bpp,i) for i in mssim]
a1 = open("../toderici_kinetics/timedresults.txt").read()
b1 = a1.split("\n")
l1 = [i.split("\t") for i in b1]
mssim1 = np.array(l1,dtype=np.float)
# lstm_ssim = np.mean(lstm_ssim, axis=0)
bpp1 = np.arange(1, 17) / 192 * 24
aucVal1 = [auc(bpp1,i) for i in mssim1]
bpp=[20,40,60,80,100,160,220,260]
plt.rcParams['xtick.major.pad']='8'
plt.rcParams['ytick.major.pad']='8'
# mngr = plt.get_current_fig_manager()
# mngr.window.setGeometry(50,50,960, 640)
# plt.tight_layout()
# plt.rcParams['ytick.major.pad']='30'
# plt.rcParams['ytick.major.pad']='18'
my_xticks=["0-20\nframes","20-40\nframes","40-60\nframes","60-80\nframes","80-100\nframes","140-160\nframes","200-220\nframes","240-260\nframes"]
plt.figure(figsize=(10,7))
plt.xticks(bpp, my_xticks,fontsize=9)
plt.plot(bpp, aucVal, label='hypernet context', marker='o')
plt.plot(bpp, aucVal1, label='toderici kinetic', marker='x')
print(aucVal)
print(aucVal1)
# plt.text(80,1.81,"toderici kinetic auc={:2.4f}")
# plt.text(100,1.8,"jpeg ycbcr 4:2:0 auc={:2.4f}")

plt.xlim(0 , 280)
plt.ylim(1.84, 1.86)
plt.grid()
plt.xlabel('Time Frame Ranges')
plt.ylabel('Area under curve')
plt.legend()
plt.title("Tested over first 260 frames of 10 Videos")
# plt.rcParams.update({'font.size': 22})
plt.savefig("jpegvslstm.png", dpi=200)
plt.show()
