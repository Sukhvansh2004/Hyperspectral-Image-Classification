import argparse
import torch
from datasets import HSIDataset, split_dataset
from models import CSSDModel
from utils import load_checkpoint

def infer(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    full = HSIDataset(args.data_path, spectral_dim=args.spectral_dim, data=args.data)
    _,_,test_set = split_dataset(full, args.train_ratio, args.val_ratio)
    model = CSSDModel(args.spectral_dim, args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    load_checkpoint(model, optimizer, args.ckpt, device)
    sample, label = test_set[args.index]
    model.eval()
    with torch.no_grad():
        pred,_ = model(sample.unsqueeze(0).to(device))
    print(f"GT: {label}, Pred: {pred.argmax(1).item()}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--spectral-dim', type=int, default=200)
    parser.add_argument('--num-classes', type=int, default=16)
    parser.add_argument('--ckpt', type=str, default='best.pth')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--train-ratio', type=float, default=0.6)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data', type=str, default='data')
    args = parser.parse_args()
    infer(args)
