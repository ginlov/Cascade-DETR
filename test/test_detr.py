import sys
import os

# Add the modules directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../modules')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../util')))

import unittest
import torch
from torch import nn
from transformer import Transformer
from positional_encoder import PositionEmbeddingSine
from DETR import DETR

class DummyBackbone(nn.Module):
    def __init__(self, num_channels=3, out_channels=256):
        super().__init__()
        self.num_channels = num_channels
        self.conv = nn.Conv2d(num_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W)
        return self.conv(x)

class DummyPositionalEncoder(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.hdim = hdim

    def forward(self, x):
        # x: (B, C, H, W)
        # Return a dummy positional encoding of shape (B, hdim, H, W)
        B, C, H, W = x.shape
        return torch.zeros(B, self.hdim, H, W, device=x.device, dtype=x.dtype)

class TestDETR(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.img_h = 32
        self.img_w = 32
        self.num_classes = 20
        self.d_model = 256
        self.num_queries = 10

        self.backbone = DummyBackbone(num_channels=3, out_channels=self.d_model)
        self.pos_encoder = DummyPositionalEncoder(self.d_model)
        self.transformer = Transformer(d_model=self.d_model, nhead=8, num_encoder_layers=2, num_decoder_layers=2)
        self.detr = DETR(
            backbone=self.backbone,
            pos_encoder=self.pos_encoder,
            transformer=self.transformer,
            num_queries=self.num_queries,
            num_classes=self.num_classes
        )

    def test_forward_shape(self):
        img = torch.randn(self.batch_size, 3, self.img_h, self.img_w)
        mask = torch.zeros(self.batch_size, self.img_h, self.img_w, dtype=torch.bool)
        class_logits, bbox_preds = self.detr(img, mask)
        # class_logits: (batch_size, num_queries, num_classes+1)
        # bbox_preds: (batch_size, num_queries, 4)
        self.assertEqual(class_logits.shape[0], self.batch_size)
        self.assertEqual(class_logits.shape[1], self.num_queries)
        self.assertEqual(class_logits.shape[2], self.num_classes + 1)
        self.assertEqual(bbox_preds.shape[0], self.batch_size)
        self.assertEqual(bbox_preds.shape[1], self.num_queries)
        self.assertEqual(bbox_preds.shape[2], 4)

if __name__ == '__main__':
    unittest.main()