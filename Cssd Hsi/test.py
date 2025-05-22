import argparse
import torch
from datasets import HSIDataset, split_dataset
from torch.utils.data import DataLoader
from models import CSSDModel
from utils import load_checkpoint, accuracy, AverageMeter

def test(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    full = HSIDataset(args.data_path, spectral_dim=args.spectral_dim)
    _,_,test_set = split_dataset(full, args.train_ratio, args.val_ratio)
    loader = DataLoader(test_set, batch_size=args.batch_size)
    model = CSSDModel(args.spectral_dim, args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    load_checkpoint(model, optimizer, args.ckpt, device)
    acc_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            preds,_ = model(x)
            acc_meter.update(accuracy(preds.argmax(1), y), x.size(0))
    print(f"Test Accuracy: {acc_meter.avg:.4f}")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--spectral-dim', type=int, default=200)
    parser.add_argument('--num-classes', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='best.pth')
    parser.add_argument('--train-ratio', type=float, default=0.6)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    args = parser.parse_args()
    test(args)