import argparse
import torch
import numpy as np
from datasets import HSIDataset, split_dataset
from torch.utils.data import DataLoader
from models import CSSDModel
from utils import load_checkpoint, accuracy, AverageMeter

def test(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    full = HSIDataset(args.data_path, spectral_dim=args.spectral_dim, data=args.data)
    _, _, test_set = split_dataset(full, args.train_ratio, args.val_ratio)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = CSSDModel(args.spectral_dim, args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    load_checkpoint(model, optimizer, args.ckpt, device)
    model.eval()

    # initialize confusion matrix
    num_classes = args.num_classes
    conf_mat = np.zeros((num_classes, num_classes), dtype=int)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds, _ = model(x)
            pred_labels = preds.argmax(dim=1)

            # update confusion matrix
            for t, p in zip(y.view(-1), pred_labels.view(-1)):
                conf_mat[t.item(), p.item()] += 1

    # compute metrics
    total = conf_mat.sum()
    oa = np.trace(conf_mat) / total                      # Overall Accuracy
    per_class_acc = np.diag(conf_mat) / conf_mat.sum(1)   # per-class accuracy
    aa = np.nanmean(per_class_acc)                       # Average Accuracy

    # Cohen's kappa
    row_marginals = conf_mat.sum(axis=1)
    col_marginals = conf_mat.sum(axis=0)
    pe = (row_marginals * col_marginals).sum() / (total**2)
    kappa = (oa - pe) / (1 - pe)

    # print results
    print(f"Overall Accuracy (OA):    {oa*100:.2f}%")
    print(f"Average Accuracy (AA):    {aa*100:.2f}%")
    print(f"Cohen's Kappa (κ):        {kappa:.4f}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',   required=True)
    parser.add_argument('--spectral-dim', type=int, default=200)
    parser.add_argument('--num-classes',  type=int, default=16)
    parser.add_argument('--batch-size',   type=int, default=32)
    parser.add_argument('--gpu',          type=int, default=0)
    parser.add_argument('--ckpt',        type=str, default='best.pth')
    parser.add_argument('--train-ratio', type=float, default=0.6)
    parser.add_argument('--val-ratio',   type=float, default=0.2)
    parser.add_argument('--data',        type=str,   default='data')
    args = parser.parse_args()
    test(args)
