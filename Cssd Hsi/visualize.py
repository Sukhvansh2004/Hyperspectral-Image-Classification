"""
Visualize Indian Pines hyperspectral data and ground truth labels.
Generates a false-color RGB composite and overlays the label map.
"""
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def visualize_hsi(data_path, label_path, bands=(50,30,10)):
    """
    data_path: path to .mat file containing HxWxS data as 'data'
    label_path: path to .mat file containing HxW labels as 'labels'
    bands: tuple of three band indices for RGB composite
    """
    # Load .mat files
    data_mat = scipy.io.loadmat(data_path)
    label_mat = scipy.io.loadmat(label_path)
    # Dynamically get variable names
    data = data_mat[[k for k in data_mat.keys() if not k.startswith('__')][0]]
    labels = label_mat[[k for k in label_mat.keys() if not k.startswith('__')][0]]

    # Create RGB composite
    rgb = np.stack([data[:,:,bands[0]], data[:,:,bands[1]], data[:,:,bands[2]]], axis=2)
    # Normalize for display
    rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    # Plot
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title('False-Color RGB Composite')
    plt.imshow(rgb_norm)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title('Ground Truth Labels')
    cmap = plt.get_cmap('jet', np.unique(labels).size)
    plt.imshow(labels, cmap=cmap, vmin=labels.min(), vmax=labels.max())
    plt.colorbar(ticks=np.unique(labels))
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    # Default paths
    data_p = 'data\PaviaU\data.mat' #os.path.join('data', 'IndianPines', 'data.mat')
    lbl_p  = 'data\PaviaU\labels.mat' #os.path.join('data', 'IndianPines', 'labels.mat')
    visualize_hsi(data_p, lbl_p)