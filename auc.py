import numpy as np
from numpy import genfromtxt
from sklearn.metrics import auc
import matplotlib.pyplot as plt
my_data = genfromtxt('test/lstm_ssim.csv', delimiter=',')
y = my_data[:,:16]
print(y.shape)
m = np.mean(y,axis=0)
print(m)
# m = "0.87746	0.93982	0.96346	0.97410	0.98024	0.98423	0.98685	0.98902	0.99047	0.99165	0.99263	0.99338	0.99396	0.99442	0.99482	0.99512"
# l = m.split("\t")
# print(len(l))
# print(l)
# m = [float(i) for i in l]
x = np.array(range(1,17))*0.125
aucval= auc(x,m)

plt.plot(x,m)

plt.xlabel('bits per pixel')

plt.ylabel('mssim')

plt.title('toderici kinetics data')

plt.text(1,0.9,"auc={}".format(aucval))

plt.grid(True)
plt.savefig("todkin.png")
plt.show()