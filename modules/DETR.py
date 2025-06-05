from torch import nn
from .transformer import Transformer, TransformerEncoder, TransformerDecoder
from .positional_encoder import PositionEmbeddingSine, PositionEmbeddingLearned
from .backbone import FrozenBatchNorm2d
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from torch.nn import functional as F

import torch

def build_position_encoding(args):
    """
    The idea is borrowed from the facebookresearch/DETR repository.
    """
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding

class DETR(nn.Module):
    def __init__(
            self, 
            backbone: nn.Module,
            pos_encoder: nn.Module,
            transformer: Transformer,
            num_queries: int = 100,
            num_classes: int = 91
        ):
        super(DETR, self).__init__()
        self.hdim = transformer.d_model
        self.backbone = backbone
        self.pos_encoder = pos_encoder
        self.transformer = transformer
        self.num_classes = num_classes

        self.input_proj = nn.Conv2d(backbone.num_channels, self.hdim, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, self.hdim) # Learnable query positional encoding

        self.class_embed = nn.Linear(self.hdim, num_classes + 1)  # +1 for the no-object class   
        self.bbox_embed = nn.Linear(self.hdim, 4)     ## TODO Change this to MLP

    def forward(self, img: NestedTensor):
        if isinstance(img, (list, torch.Tensor)):
            img = nested_tensor_from_tensor_list(img)
        img, mask = img.decompose() # TODO make backbone accept NestedTensor and compute mask inside backbone
        src = self.backbone(img)
        mask = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
        pos_embed = self.pos_encoder(NestedTensor(src, mask))

        hs = self.transformer(self.input_proj(src), mask=mask, query_embed=self.query_embed.weight, pos_embed=pos_embed)[0]
        class_logits = self.class_embed(hs)[-1] # Match decoder output since it may return imediate outputs
        bbox_preds = self.bbox_embed(hs).sigmoid()[-1] # Match decoder output since it may return imediate outputs
        return {'pred_logits': class_logits, 'pred_boxes': bbox_preds}
    
class CascadeDETR(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, num_classes):
        super(CascadeDETR, self).__init__()

    def forward(self, src, tgt):
        pass
        return None
    