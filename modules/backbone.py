from torchvision.models import resnet101, resnet50
from torchvision.models.resnet import ResNet50_Weights, ResNet101_Weights
from torch import nn

import torch

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
    
class Backbone(nn.Module):
    """
    A simple backbone that uses FrozenBatchNorm2d.
    """
    
    def __init__(self, backbone_name, pretrained=False, batch_norm_freeze=False, resolution_increase=False):
        super(Backbone, self).__init__()

        if backbone_name == 'resnet50':
            if batch_norm_freeze:
                backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, norm_layer=FrozenBatchNorm2d, replace_stride_with_dilation=[False, False, True])
            else:
                backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1, replace_stride_with_dilation=[False, False, True])
        elif backbone_name == 'resnet101':
            if batch_norm_freeze:
                backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, norm_layer=FrozenBatchNorm2d,replace_stride_with_dilation=[False, False, True])
            else:
                backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1, replace_stride_with_dilation=[False, False, True])
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        backbone = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*backbone)

        if resolution_increase:
            # Modify the last layer to increase resolution
            self.backbone.fc = torch.nn.Conv2d(self.backbone.fc.in_features, 256, kernel_size=1)

        if backbone_name in ['resnet50', 'resnet101']:
            self.num_channels = 2048

    def forward(self, x):
        x = self.backbone(x)
        return x
    
def make_backbone(args):
    return Backbone(
        backbone_name=args.backbone_name,
        pretrained=args.pretrained,
        batch_norm_freeze=args.batch_norm_freeze,
        resolution_increase=args.resolution_increase
    )