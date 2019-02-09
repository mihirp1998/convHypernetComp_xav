from sklearn.metrics import auc
import numpy as np
import matplotlib.pyplot as plt
hypernetContext = "0.91320 0.95956 0.97403 0.98136 0.98522 0.98840 0.99044 0.99199 0.99323 0.99401 0.99465 0.99519 0.99572 0.99603 0.99633 0.99653"
toderici = "0.88121 0.94379 0.96558 0.97550 0.98130 0.98517 0.98769 0.98982 0.99126 0.99238 0.99329 0.99401 0.99455 0.99498 0.99536 0.9956"
hypernetEmbed = "0.90719 0.95611 0.97194 0.97975 0.98474 0.98766 0.98983 0.99107 0.99233 0.99342 0.99410 0.99466 0.99529 0.99572 0.99596 0.99624"

hc_mssim = hypernetContext.split(" ")
hc_mssim =  [float(i) for i in hc_mssim]

he_mssim = hypernetEmbed.split(" ")
he_mssim =  [float(i) for i in he_mssim]


t_mssim = toderici.split(" ")
t_mssim =  [float(i) for i in t_mssim]

bpp = np.arange(1, 17) / 192 * 24

hc_auc = auc(bpp,hc_mssim)
he_auc = auc(bpp,he_mssim)
t_auc = auc(bpp,t_mssim)
plt.figure(figsize=(10,7))
# plt.figure(figsize=(10,7))
# plt.xticks(bpp, my_xticks,fontsize=9)
plt.plot(bpp, hc_mssim, label='hypernet context', marker='o')
plt.plot(bpp, he_mssim, label='hypernet embed', marker='+')
plt.plot(bpp, t_mssim, label='toderici kinetic', marker='x')

# plt.text(1,0.9,"toderici mssim - "+hypernet)
# plt.text(1,0.91,"hypernet mssim - "+toderici)
# for i in range(len(t_mssim)):
#   plt.annotate("{:2.3f}".format(h_mssim[i]),(bpp[i]-0.05,h_mssim[i]+0.0002))


plt.text(1,0.92,"toderici kinetic auc={:2.4f}".format(t_auc),fontsize=12)
plt.text(1,0.93,"hypernet embed auc={:2.4f}".format(he_auc),fontsize=12)
plt.text(1,0.94,"hypernet context auc={:2.4f}".format(hc_auc),fontsize=12)

plt.xlim(0., 2.)
plt.ylim(0.86, 1)
plt.grid()
plt.xlabel('bit per pixel')
plt.ylabel('MS-SSIM')
plt.legend()
plt.title("Shorter Vids -Tested over first 15frames of 100 videos")
# plt.rcParams.update({'font.size': 22})
plt.savefig("short_hypervstoderici.png", dpi=200)
plt.show()