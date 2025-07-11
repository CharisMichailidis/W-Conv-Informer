import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
from encoder import DilatedConvLayer, ConvLayer, GaussianTransformLayerC3, EncoderLayer, Encoder
from attn import FullAttention, ProbAttention, AttentionLayer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

attention = ProbAttention( mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False)

encoder = Encoder(attn_layers=attention)

# Generate a random input
batch_size = 8
seq_len = 50
d_model = 512
x = torch.rand(batch_size, seq_len, d_model)  # [B, L, D]

output = encoder(x)

print(f'output.shape:{output.shape}')

