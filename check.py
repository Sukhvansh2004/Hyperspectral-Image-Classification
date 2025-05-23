import scipy.io as sio
import os   
data_mat = sio.loadmat("/Users/ashmit22/Hyperspectral-Image-Classification/Cssd Hsi/archive/PaviaU.mat")
print(data_mat['paviaU'].shape())