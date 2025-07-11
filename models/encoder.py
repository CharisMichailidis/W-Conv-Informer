import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class TemporalConv1D(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        super(TemporalConv1D, self).__init__()
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv1d(output_channels, output_channels, kernel_size, stride, padding),
            nn.ReLU(),
          
            )
        
    
    
    def forward(self, x):
        # Input shape : [batch_size, input_channels, seq_len]
        return self.conv1d_layers(x)


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)                    
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x
    
class DilatedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(DilatedConvLayer, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)   # input.shape = (batch_size, in_channels, sequence_length)
    
    def forward(self, x):
        return F.relu(self.conv1d(x))
    




class GaussianTransformLayerC3(nn.Module):
    def __init__(self, wavelet='gaus1'):
        super(GaussianTransformLayerC3, self).__init__()
        self.wavelet = wavelet    # e.g., 'gaus1', 'gaus2' for Gaussian wavelets of different orders

    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        wavelet_coeffs = []

        for i in range(batch_size):
            channel_coeffs = []
            for c in range(channels):
                #coeffs, _ = pywt.cwt(x[i, :, c].detach().cpu().numpy(), scales=[1, 2, 3], wavelet=self.wavelet)
                coeffs, _ = pywt.cwt(x[i, :, c].detach().cpu().numpy(), scales=[1, 3, 6], wavelet=self.wavelet)  # (num_scales, seq_length)

                # Resample CWT output to match sequence length (average or sum along scales)
                coeffs_tensor = torch.tensor(coeffs, dtype=torch.float32, device=device)
                channel_coeffs.append(coeffs_tensor.unsqueeze(1))      # Add channel dimension  (num_scales, seq_length) --->  (num_scales, 1, seq_length).
                



            # Concatenate coefficients for all channels in the batch
            batch_coeffs = torch.cat(channel_coeffs, dim=1)   # Shape: (num_scales, seq_len, channels)
            wavelet_coeffs.append(batch_coeffs)   



        x_wavelet = torch.stack(wavelet_coeffs).transpose(2, 3)
       
        

        return x_wavelet





class EncoderLayer(nn.Module):
    
    def __init__(self, attention, d_model, d_ff, dropout= 0.1 ,activation='relu', wavelet='gaus1'  ):
   
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv_temp_poss = TemporalConv1D(input_channels=d_model, output_channels=d_model, kernel_size=1)
        self.wavelet_transform = GaussianTransformLayerC3(wavelet=wavelet)
        self.conv1 = DilatedConvLayer(d_model, d_ff , dilation=2)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation=='relu' else F.gelu

    def forward(self, x, attn_mask=None):
        # Wavelet Transform for multi-scale feature extraction
        x = self.wavelet_transform(x)
        

        scale_attention_outputs=[]
        scale_attention = []
        for s in range(x.shape[1]):
            scale_x = x[:, s, :, :]
    
            scale_x = self.conv_temp_poss(scale_x.transpose(-1, 1))      #   x [B, L, D]  -----> [B, D, L]
            scale_x = scale_x.transpose(-1, 1)                           #     [B, D, L]  -----> [B, L, D]


            new_scale_x, attn = self.attention(scale_x, scale_x, scale_x, attn_mask=attn_mask) 
    

        
            new_scale_x = scale_x + self.dropout(new_scale_x)  # Check this change it is important


            # Residual connection and normalization for this scale
            scale_x = scale_x + self.dropout(new_scale_x)
            scale_x = self.norm1(scale_x)

            # Append the processed scale output and attention weights to lists
            scale_attention_outputs .append(scale_x.unsqueeze(1))  # # Add scale dimension back
            scale_attention.append(attn)

        x = torch.cat(scale_attention_outputs, dim=1)

        x = x.mean(dim=1)  # Shape: [batch_size, seq_len, channels]


        # Dilated Convolution and Feedforward Network
        y = x.transpose(-1, 1)   # [B, D, L] for Conv1d
        y = self.conv1(y)         # Dilated Conv
        y = self.dropout(self.activation(self.conv2(y)))
        y = y.transpose(-1, 1)    # [B, L, D] back for LayerNorm

        # Residual connection and normalization
        return self.norm2(x + y), attn

            


class Encoder(nn.Module):   #       Encoder_arxikos
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, wavelet='gaus1'  ):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm = norm_layer
        self.wavelet_transform = GaussianTransformLayerC3(wavelet) 
        

    def forward(self, x, attn_mask=None):
        
        
        attns = []
        if self.conv_layers is not None:

            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        
        if self.norm is not None:
            x = self.norm(x)

        
        return x, attns

 
class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(self, EncoderStack).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x):
        x_stack = []
        attns = []
        for  i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_lens = x.shape[1] // (2**i_len)
            x_s, attn = encoder(x[:, -inp_lens:, :])
            x_stack.append(x_s)
            attns.append(attn)

        x_stack = torch.cat(x_stack, dim=-2)

        return x_stack, attns



 
