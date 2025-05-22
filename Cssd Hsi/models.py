import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeFE(nn.Module):
    def __init__(self, in_channels=1, spectral_dim=None):
        super().__init__()
        assert spectral_dim is not None
        self.bn = nn.BatchNorm3d(in_channels)
        self.dilate = nn.Conv3d(in_channels, 64, kernel_size=(3,1,1), dilation=(3,1,1), stride=(3,1,1), padding=(2,0,0))
        self.spectral_len = spectral_dim // 3
        self.wide = nn.Conv3d(64, 256, kernel_size=(self.spectral_len,1,1), groups=2)
        
    def forward(self, x):
        x = self.bn(x)
        x = self.dilate(x)
        x = self.wide(x)
        return x.squeeze(2)

class SpeFR(nn.Module):
    """
    Spectral Feature Refiner:
    - learnable structuring element -> dilation -> attention -> max
    """
    def __init__(self, channels, se_size=(3,3)):
        super().__init__()
        w, h = se_size
        self.w, self.h = w, h
        # structuring element parameters
        self.SE = nn.Parameter(torch.randn(w * h))
        # attention conv: operate over blocks as depth dimension
        self.attn_conv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        # x: B, C, W, H
        B, C, W, H = x.shape
        # expand spatial to 3D: B, C, 1, W, H
        x3d = x.unsqueeze(2)
        # compute SE weights
        se = torch.sigmoid(self.SE)  # shape (w*h,)
        # pad spatial dims
        pad_w, pad_h = self.w // 2, self.h // 2
        x_pad = F.pad(x3d, (pad_h, pad_h, pad_w, pad_w))
        # extract blocks: B, C, D, 1, W, H where D = w*h
        blocks = x_pad.unfold(3, self.w, 1).unfold(4, self.h, 1)
        blocks = blocks.contiguous().view(B, C, self.w * self.h, 1, W, H)
        # apply structuring element weights
        weighted = blocks * se.view(1, 1, -1, 1, 1, 1)
        # remove singleton dim for attention: B, C, D, W, H
        weighted3d = weighted.squeeze(3)
        # compute attention per block
        attn = torch.sigmoid(self.attn_conv(weighted3d))  # B, C, D, W, H
        # apply attention
        weighted_attended = weighted3d * attn
        # aggregate max over block dimension D
        out, _ = weighted_attended.max(dim=2)
        return out  # shape B, C, W, H
    
class SpaFE(nn.Module):
    def __init__(self, in_channels=256, mid_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
    def forward(self, x): return self.conv(x)

class SAPClassifier(nn.Module):
    def __init__(self, in_channels=64, num_classes=10):
        super().__init__()
        self.conv_cls = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
    def forward(self, x_spefe, x_spafe):
        logits = self.conv_cls(x_spafe)
        B,K,W,H = logits.shape
        c_feat = x_spefe[:,:,W//2,H//2].unsqueeze(-1).unsqueeze(-1)
        spe_flat = x_spefe.view(B,256,-1)
        c_flat = c_feat.view(B,256,1)
        cos = F.cosine_similarity(spe_flat, c_flat, dim=1).view(B,1,W,H)
        weighted = logits * cos
        out = weighted.view(B,K,-1).mean(dim=2)
        return out, logits

class CSSDModel(nn.Module):
    def __init__(self, spectral_dim, num_classes, epsilon=0.1, tau=4.0):
        super().__init__()
        self.spefe = SpeFE(1, spectral_dim)
        self.spafr = SpeFR(256)
        self.spafe = SpaFE(256,64)
        self.sap = SAPClassifier(64, num_classes)
        self.aux = nn.Conv2d(256, num_classes, kernel_size=1)
        self.epsilon, self.tau = epsilon, tau
    def forward(self, x):
        x_spefe = self.spefe(x)
        x_spefr = self.spafr(x_spefe)
        x_spafe = self.spafe(x_spefr)
        y_main, _ = self.sap(x_spefe, x_spafe)
        c = x_spefe[:,:,x_spefe.size(2)//2, x_spefe.size(3)//2].unsqueeze(-1).unsqueeze(-1)
        y_aux = self.aux(c).view(x.size(0), -1)
        return y_main, y_aux

def cssd_loss(y_main, y_aux, labels, epsilon, tau):
    K = y_main.size(1)
    y = F.one_hot(labels, K).float()
    p_aux = F.softmax(y_aux / tau, dim=1)
    y_tilde = epsilon * p_aux + (1 - epsilon) * y
    loss_main = -(y_tilde * F.log_softmax(y_main,1)).sum(1).mean()
    loss_aux  = -(y_tilde * F.log_softmax(y_aux,1)).sum(1).mean()
    return loss_main + loss_aux