# Attention Layer using wavelets

import torch
import torch.nn as nn
import pywt
import numpy as np

class CDWDecomposedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, scales, wavelet='gaus1'):
        super(CDWDecomposedMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by num_heads"

        # Initialize linear projections for query, key, and value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        #self.out_proj = nn.Linear(seq_len * embed_dim, embed_dim)

        # Define wavelet parameters
        self.scales = scales
        self.wavelet = wavelet

    def  cwt_transform(self, x, scales):
        """
        Applies continuous wavelet transform to the input x using a Gaussian wavelet at specified scales.
        Args:
            x (tensor): Input tensor of shape (batch_size, seq_length, embed_dim)
            scales (list): Scales for the CWT, determining frequency resolution
        Returns:
            Tensor of shape (len(scales), batch_size, seq_length, embed_dim) representing CWT-transformed data
        """
        batch_size, seq_len, embed_dim = x.shape
        transformed_signals = []

        # Apply CWT across each sequence independently
        for b in range(batch_size):
            cwt_effs = []
            for d in range(embed_dim):
                # Perform CWT with specified scales and wavelet
                #coeffs, _ = pywt.cwt(x[b, :, d].detach().cpu().numpy(), scales=[1, 3, 6], wavelet=self.wavelet)
                coeffs, _ = pywt.cwt(x[b, :, d].detach().cpu().numpy(), scales=scales, wavelet=self.wavelet)
                cwt_effs.append(coeffs)
            # Stack the CWT coefficients for each embedding dimension
            transformed_signals.append(np.stack(cwt_effs, axis=-1))  # Shape: (len(scales), seq_length, embed_dim)

        # Convert back to tensor
        transformed = torch.tensor(np.stack(transformed_signals, axis=1)).to(x.device)

        return transformed
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()

        # Apply CWT for the specified scales
        cwt_outputs = self.cwt_transform(x, self.scales)

        multi_scale_attention_outputs = []
        # Project to query, key, and value for each scale
        for i in range(len(self.scales)):
            cwt_x = cwt_outputs[i]    # Shape: (batch_size, seq_length, embed_dim)
            query = self.query_proj(cwt_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            print(f'query_shape:{query.shape}')         #    query_shape:torch.Size([2, 4, 10, 2])
            key = self.key_proj(cwt_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            print(f'key_shape:{key.shape}')
            value = self.value_proj(cwt_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            print(f'value_shape:{value.shape}')

            # Compute scaled dot-product attention for each scale
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.num_heads ** 0.5)   #  attention_scores_shape:torch.Size([2, 4, 10, 10]) 
            print(f'attention_scores_shape:{attention_scores.shape}')
            attention_weights = nn.functional.softmax(attention_scores, dim=-1)    #   attention_weights_shape:torch.Size([2, 4, 10, 10])
            print(f'attention_weights_shape:{attention_weights.shape}')
            attention_output = torch.matmul(attention_weights, value).transpose(1, 2)   # torch.Size([2, 10, 4, 2])
            print(f'attention_output_shape:{attention_output.shape}')  #   attention_output_shape:torch.Size([2, 4, 10, 2])

            # Collect attention outputs for each scale
            multi_scale_attention_outputs.append(attention_output)

        # Concatenate the attention outputs from each scale (frequency band)
        print(f'multi_scale_attention_outputs.len:{len(multi_scale_attention_outputs)}')

        #combined_attention = torch.cat(multi_scale_attention_outputs, dim=-1).transpose(1, 2)
        combined_attention = torch.stack(multi_scale_attention_outputs, dim=-1)  # combined_attention.shape:torch.Size([2, 10, 4, 2, 4])
        #print(f'combined_attention.shape:{combined_attention.shape}')
        #print(multi_scale_attention_outputs[0])

        mean_attention_output = torch.mean(combined_attention, dim=-1)

        #combined_attention = combined_attention.contiguous().view(batch_size, seq_len, embed_dim)
        #mean_atention_output = mean_attention_output.view(batch_size * seq_len , embed_dim)  # :torch.Size([2, 10, 8])
        output = mean_attention_output.view(batch_size , seq_len , embed_dim)  # :torch.Size([2, 10, 8])
        #print(f'mean_attention_output.shape:{mean_atention_output.shape}')  

        # Final output projection
        #output = self.out_proj(combined_attention)
        #output = self.out_proj(mean_attention_output)
        #output = output.view(batch_size, seq_len, embed_dim)
        #print(f'output.shape:{output.shape}')

        return output
    

# Testing the custom CWT-Decomposed Attention Layer
batch_size = 2
seq_len = 10
embed_dim = 8
num_heads = 4
scales = [1, 2, 4, 8]    # Different scales for different frequency bands

x  = torch.rand(batch_size, seq_len, embed_dim)
cwt_attention = CDWDecomposedMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, scales=scales)
output = cwt_attention(x)
print(output)
 

                




