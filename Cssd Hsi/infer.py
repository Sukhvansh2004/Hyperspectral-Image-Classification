import argparse
import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import HSIDataset, split_dataset
from models import CSSDModel
from utils import load_checkpoint


def infer_full_image(args, bands=(30, 20, 10)):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # Load full data cube and labels
    data_mat = scipy.io.loadmat(os.path.join(args.data_path, args.data))
    label_mat = scipy.io.loadmat(os.path.join(args.data_path, args.label_file))
    data_var = [k for k in data_mat.keys() if not k.startswith('__')][0]
    labels_var = [k for k in label_mat.keys() if not k.startswith('__')][0]
    cube = data_mat[data_var]  # H x W x S
    gt = label_mat[labels_var]  # H x W

    H, W, S = cube.shape
    # Initialize prediction map
    pred_map = np.zeros((H, W), dtype=int)

    # Prepare model
    model = CSSDModel(args.spectral_dim, args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    load_checkpoint(model, optimizer, args.ckpt, device)
    model.eval()

    # Iterate over all labeled pixels
    patch_size = args.patch_size
    pad = patch_size // 2
    padded = np.pad(cube, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    coords = [(i, j) for i in range(H) for j in range(W) if gt[i, j] > 0]

    with torch.no_grad():
        for i, j in coords:
            patch = padded[i:i+patch_size, j:j+patch_size, :]
            patch = np.transpose(patch, (2,0,1))[None, None, ...]  # 1,1,S,ps,ps
            tensor = torch.from_numpy(patch).float().to(device)
            y_main, _ = model(tensor)
            pred = y_main.argmax(dim=1).item() + 1  # back to 1-based
            pred_map[i, j] = pred

    # Create RGB composite of full cube
    rgb = np.stack([cube[:, :, bands[0]], cube[:, :, bands[1]], cube[:, :, bands[2]]], axis=2)
    rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min())

    # Plot original composite and predicted map
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.title('False-Color RGB Composite')
    plt.imshow(rgb_norm)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title('Predicted Label Map')
    cmap = plt.get_cmap('jet', args.num_classes)
    plt.imshow(pred_map, cmap=cmap, vmin=1, vmax=args.num_classes)
    plt.colorbar(ticks=range(1, args.num_classes+1))
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True,
                        help='Folder containing data.mat and labels.mat')
    parser.add_argument('--data', required=True)
    parser.add_argument('--label-file', default='labels.mat')
    parser.add_argument('--spectral-dim', type=int, default=200)
    parser.add_argument('--num-classes', type=int, default=16)
    parser.add_argument('--ckpt', type=str, default='best.pth')
    parser.add_argument('--patch-size', type=int, default=5)
    parser.add_argument('--bands', type=int, nargs=3, default=[30,20,10],
                        help='Band indices for RGB composite')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    infer_full_image(args, bands=tuple(args.bands))