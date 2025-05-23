import scipy.io as sio
import os   
data_mat = sio.loadmat("Cssd Hsi/data/PaviaU/PaviaU.mat")
print(data_mat['paviaU'].shape)