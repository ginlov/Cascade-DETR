from torch import nn
from transformer import TransformerEncoder, TransformerDecoder, Attention

import torch

def make_backbone(backbone_name, pretrained=True, batch_norm_freeze=True, resolution_increase=False):
    if backbone_name == 'resnet50':
        from torchvision.models import resnet50
        backbone = resnet50(pretrained=pretrained)
        if batch_norm_freeze:
            for param in backbone.parameters():
                if isinstance(param, nn.BatchNorm2d):
                    param.requires_grad = False
        if resolution_increase:
            # Modify the last layer to increase resolution
            backbone.fc = nn.Conv2d(backbone.fc.in_features, 256, kernel_size=1)
    elif backbone_name == 'resnet101':
        from torchvision.models import resnet101
        backbone = resnet101(pretrained=pretrained)
        if batch_norm_freeze:
            for param in backbone.parameters():
                if isinstance(param, nn.BatchNorm2d):
                    param.requires_grad = False
        if resolution_increase:
            # Modify the last layer to increase resolution
            backbone.fc = nn.Conv2d(backbone.fc.in_features, 256, kernel_size=1)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    return backbone


class DETR(nn.Module):
    def __init__(
            self, 
            backbone_name='resnet50', 
            pretrained=True, 
            batch_norm_freeze=True, 
            resolution_increase=False,
            N=6,
            train_positional_encoding=True,):
        super(DETR, self).__init__()
        self.backbone = make_backbone(backbone_name, pretrained, batch_norm_freeze, resolution_increase)
        self.q_positional_encoding = nn.Parameter(torch.randn(1, 256, 50, 50)) if train_positional_encoding else nn.Parameter(torch.zeros(1, 256, 50, 50), requires_grad=False)
        self.kv_positional_encoding = nn.Parameter(torch.randn(1, 256, 50, 50)) if train_positional_encoding else nn.Parameter(torch.zeros(1, 256, 50, 50), requires_grad=False)
        self.encoder = TransformerEncoder(d_model=256, nhead=8, num_layers=N)
        self.decoder = TransformerDecoder(d_model=256, nhead=8, num_layers=N)
        self.ffn = nn.Linear(256, 256)

    def forward(self, src, tgt):
        pass
        return None
    
class CascadeDETR(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, num_classes):
        super(CascadeDETR, self).__init__()

    def forward(self, src, tgt):
        pass
        return None
    