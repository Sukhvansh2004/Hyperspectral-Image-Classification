import argparse
import torch
from datasets import HSIDataset, split_dataset
from torch.utils.data import DataLoader
from models import CSSDModel, cssd_loss
from utils import set_seed, save_checkpoint, accuracy, AverageMeter

def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    full = HSIDataset(args.data_path, spectral_dim=args.spectral_dim, data=args.data)
    train_set, val_set, test_set = split_dataset(full, args.train_ratio, args.val_ratio)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size)
    model = CSSDModel(args.spectral_dim, args.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        loss_meter = AverageMeter()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            y_main,y_aux = model(x)
            loss = cssd_loss(y_main,y_aux,y, model.epsilon, model.tau)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_meter.update(loss.item(), x.size(0))
        print(f"Epoch {epoch}: Loss={loss_meter.avg:.4f}")
        model.eval()
        acc_meter = AverageMeter()
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                preds,_ = model(x)
                acc_meter.update(accuracy(preds.argmax(1), y), x.size(0))
        print(f"Val Acc={acc_meter.avg:.4f}")
        if acc_meter.avg > best:
            best = acc_meter.avg
            save_checkpoint({'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}, args.ckpt)
    # test
    from utils import load_checkpoint
    load_checkpoint(model, optimizer, args.ckpt, device)
    acc_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            preds,_ = model(x)
            acc_meter.update(accuracy(preds.argmax(1), y), x.size(0))
    print(f"Test Acc={acc_meter.avg:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--spectral-dim', type=int, default=200)
    parser.add_argument('--num-classes', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train-ratio', type=float, default=0.6)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='best.pth')
    parser.add_argument('--data', type=str, default='data')
    args = parser.parse_args()
    train(args)