import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from math import sqrt

from utils.masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False ):
        super(self, FullAttention).__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def forward(self, queries, keys, values, att_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, values)
