import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GaussianTransformLayerC(nn.Module):
    def __init__(self, wavelet='gaus1'):
        super(GaussianTransformLayerC, self).__init__()
        self.wavelet = wavelet    # e.g., 'gaus1', 'gaus2' for Gaussian wavelets of different orders

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        wavelet_coeffs = []

        for i in range(batch_size):
            for c in range(channels):
                #coeffs = pywt.cwt(x[i, :, c].detach().cpu().numpy(), scales=[1, 2, 3], wavelet=self.wavelet)[0]
                coeffs = pywt.cwt(x[i, :, c].detach().cpu().numpy(), scales=[1, 3, 6], wavelet=self.wavelet)[0]
                #coeffs = pywt.cwt(x[i, :, c].detach().cpu().numpy(), scales=[1, 5, 10], wavelet=self.wavelet)[0]
                coeffs_tensor = torch.tensor(coeffs, dtype=torch.float32).to(x.device)  # Convert to tensor
                wavelet_coeffs.append(coeffs_tensor.unsqueeze(0)).to(x.device)  # Add batch dimension


        # Stack and reshape the wavelet coefficients for batch and channel processing
        stacked_coeffs = torch.stack(wavelet_coeffs)     # Shape: (batch_size * channels, 1, scales, seq_len)
        #x_wavelet = stacked_coeffs.view(batch_size, channels, coeffs.shape[0], seq_len)  # Reshape to (batch_size, channels, scales, seq_len)
        x_wavelet = stacked_coeffs.view(batch_size, -1, seq_len)   # Reshape tp (batch_size, channels * scales, seq_len)

        #return x_wavelet.transpose(2, 3)   # Final Shape (batch_size, channels, seq_len, scales)
        return x_wavelet.transpose(1, 2)
    
class GaussianTransformLayerC2(nn.Module):
    def __init__(self, wavelet='gaus1'):
        super(GaussianTransformLayerC2, self).__init__()
        self.wavelet = wavelet    # e.g., 'gaus1', 'gaus2' for Gaussian wavelets of different orders

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        #print(f'batch_size:{batch_size}')
        #print(f'seq_len:{seq_len}')
        #print(f'channels:{channels}')
        wavelet_coeffs = []

        for i in range(batch_size):
            channel_coeffs = []
            for c in range(channels):
                #coeffs, _ = pywt.cwt(x[i, :, c].detach().cpu().numpy(), scales=[1, 2, 3], wavelet=self.wavelet)
                coeffs, _ = pywt.cwt(x[i, :, c].detach().cpu().numpy(), scales=[1, 3, 6], wavelet=self.wavelet)

                # Resample CWT output to match sequence length (average or sum along scales)
                coeffs_resampled = np.mean(coeffs, axis=0)             # Shape: (seq_len,)
                #print(f'coeffs_resampled.shape:{coeffs_resampled.shape}')
                coeffs_tensor = torch.tensor(coeffs_resampled, dtype=torch.float32)
                channel_coeffs.append(coeffs_tensor.unsqueeze(1))      # Add channel dimension
                #print(f'channel_coeffs.length:{len(channel_coeffs)}')


                #coeffs_tensor = torch.tensor(coeffs, dtype=torch.float32).to(x.device)  # Convert to tensor
                #wavelet_coeffs.append(coeffs_tensor.unsqueeze(0))  # Add batch dimension

            # Concatenate coefficients for all channels in the batch
            batch_coeffs = torch.cat(channel_coeffs, dim=1)   # Shape: (seq_len, channels)
            #print(f'batch_coeffs.shape:{batch_coeffs.shape}')
            wavelet_coeffs.append(batch_coeffs)
            #print(f'wavelet_coeffs.length:{len(wavelet_coeffs)}')


       
        x_wavelet = torch.stack(wavelet_coeffs).to(device)  #              # Shape: (batch_size, seq_len, channels)
        #print(f'x_wavelet.shape:{x_wavelet.shape}')
        

       
        return x_wavelet
    

"""class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)"""


class DecoderLayer(nn.Module):
    #def __init__(self, self_attention, cross_attention, d_model, d_ff=None,dropout=0.1, activation="relu"):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None, dropout= 0.1 ,activation='relu', wavelet='gaus1'  ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        #self.wavelet_transform = GaussianTransformLayerC2(wavelet=wavelet)
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        
        #x = self.wavelet_transform(x)
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    #def __init__(self, layers, norm_layer=None):
    def __init__(self, layers, norm_layer=None, wavelet='gaus1'):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)      # DecoderLayers 
        self.norm = norm_layer
        #self.wavelet_transform = GaussianTransformLayerC2(wavelet) 
        

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        #x = self.wavelet_transform(x)
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)
        #print(f'decoder output:{x.shape}')

        return x