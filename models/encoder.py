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
    


class WaveletTransformLayer(nn.Module):
    def __init__(self, wavelet='db1'):
        super(WaveletTransformLayer, self).__init__()
        self.wavelet = wavelet
        # Optonal projection layer to reduce channels back to d_model after wavelet expansion
        #self.projection = nn.Linear(d_model * 2, d_model)
        #self.projection = nn.Linear()

    def forward(self, x):
        batch_size, seq_len, channels = x.shape    # x.shape = [32, 48, 512]
        wavelet_coeffs = []
        
        for i in range(batch_size):  # batch_size = 32
            for c in range(channels): # channels = d_model    
                coeffs = pywt.dwt(x[i, :, c].detach().cpu().numpy(), self.wavelet)   # seq_len = 48
                cA, cD = coeffs   # cA.shape , cD.shape = (24,)
                #print(f'cA.shape{cA.shape}')
                #print(f'cD.shape{cD.shape}')
                wavelet_coeffs.append(torch.tensor(cA).unsqueeze(0).to(x.device))   #  unsqueeze(0) to (1, 24).
                wavelet_coeffs.append(torch.tensor(cD).unsqueeze(0).to(x.device))

        #return torch.stack(wavelet_coeffs).view(batch_size, -1, seq_len // 2)
        #print(f'wavelet_coeffs.length:{len(wavelet_coeffs)}')    #  wavelet_coeffs.length:32768 = 32 * 2 * 512
        #x_wavelet = torch.stack(wavelet_coeffs).view(batch_size, channels * 2, -1)   #  x_wavelet.shape:torch.Size([32, 1024, 24])
        k = torch.stack(wavelet_coeffs)   # #  k.shape:torch.Size([32768, 1, 24])
        #print(f'k.shape:{k.shape}')   
        x_wavelet = torch.stack(wavelet_coeffs).view(batch_size, channels , -1)       # x_wavelet.shape:torch.Size([32, 512, 48])
        #print(f'x_wavelet.shape:{x_wavelet.shape}')  
        # Project back to d_model channels if necessary
        
        #return self.projection(x_wavelet.transpose(1, 2)).transpose(1, 2)
        return x_wavelet.transpose(1, 2)

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


        # Stack and reshape the wavelet coefficients for batch and channel processing
        #stacked_coeffs = torch.stack(wavelet_coeffs)     # Shape: (batch_size * channels, 1, scales, seq_len)
        #x_wavelet = stacked_coeffs.view(batch_size, channels, coeffs.shape[0], seq_len)  # Reshape to (batch_size, channels, scales, seq_len)
        #x_wavelet = stacked_coeffs.view(batch_size, -1, seq_len)   # Reshape tp (batch_size, channels * scales, seq_len)

        x_wavelet = torch.stack(wavelet_coeffs).to(device)  #              # Shape: (batch_size, seq_len, channels)
        #print(f'x_wavelet.shape:{x_wavelet.shape}')
        

        #return x_wavelet.transpose(2, 3)   # Final Shape (batch_size, channels, seq_len, scales)
        #return  x_wavelet.transpose(1, 2)
        return x_wavelet

class GaussianTransformLayerC3(nn.Module):
    def __init__(self, wavelet='gaus1'):
        super(GaussianTransformLayerC3, self).__init__()
        self.wavelet = wavelet    # e.g., 'gaus1', 'gaus2' for Gaussian wavelets of different orders

    def forward(self, x):
        #print(f'x.shape:{x.shape}')
        batch_size, seq_len, channels = x.shape
        #print(f'batch_size:{batch_size}')
        #print(f'seq_len:{seq_len}')
        #print(f'channels:{channels}')
        wavelet_coeffs = []

        for i in range(batch_size):
            channel_coeffs = []
            for c in range(channels):
                #coeffs, _ = pywt.cwt(x[i, :, c].detach().cpu().numpy(), scales=[1, 2, 3], wavelet=self.wavelet)
                coeffs, _ = pywt.cwt(x[i, :, c].detach().cpu().numpy(), scales=[1, 3, 6], wavelet=self.wavelet)  # (num_scales, seq_length)

                # Resample CWT output to match sequence length (average or sum along scales)
                #coeffs_resampled = np.mean(coeffs, axis=0)             # Shape: (seq_len,)
                #print(f'coeffs_resampled.shape:{coeffs_resampled.shape}')
                #coeffs_tensor = torch.tensor(coeffs_resampled, dtype=torch.float32)
                coeffs_tensor = torch.tensor(coeffs, dtype=torch.float32, device=device)
                channel_coeffs.append(coeffs_tensor.unsqueeze(1))      # Add channel dimension  (num_scales, seq_length) --->  (num_scales, 1, seq_length).
                #print(f'channel_coeffs.length:{len(channel_coeffs)}')


                #coeffs_tensor = torch.tensor(coeffs, dtype=torch.float32).to(x.device)  # Convert to tensor
                #wavelet_coeffs.append(coeffs_tensor.unsqueeze(0))  # Add batch dimension

            # Concatenate coefficients for all channels in the batch
            batch_coeffs = torch.cat(channel_coeffs, dim=1)   # Shape: (num_scales, seq_len, channels)
            #print(f'batch_coeffs.shape:{batch_coeffs.shape}')
            wavelet_coeffs.append(batch_coeffs)   
            #print(f'wavelet_coeffs.length:{len(wavelet_coeffs)}')


        # Stack and reshape the wavelet coefficients for batch and channel processing
        #stacked_coeffs = torch.stack(wavelet_coeffs)     # Shape: (batch_size * channels, 1, scales, seq_len)
        #x_wavelet = stacked_coeffs.view(batch_size, channels, coeffs.shape[0], seq_len)  # Reshape to (batch_size, channels, scales, seq_len)
        #x_wavelet = stacked_coeffs.view(batch_size, -1, seq_len)   # Reshape tp (batch_size, channels * scales, seq_len)

        #x_wavelet = torch.stack(wavelet_coeffs).to(device)  #              # Shape: (batch_size, num_scales, seq_len, channels)
        x_wavelet = torch.stack(wavelet_coeffs).transpose(2, 3)
        #print(f'x_wavelet.shape:{x_wavelet.shape}')
        

        #return x_wavelet.transpose(2, 3)   # Final Shape (batch_size, channels, seq_len, scales)
        #return  x_wavelet.transpose(1, 2)
        return x_wavelet

batch_size = 8
seq_len = 10
num_scales = 3
channels = 5

x = torch.rand(batch_size, seq_len, channels)

model = GaussianTransformLayerC3(wavelet='gaus1')

output = model(x)   # output.shape = [batch_size, num_scales, channels, seq_len]

#print(output.shape)


class GaussianTransformLayer(nn.Module):
    def __init__(self, wavelet='haar'):
        super(GaussianTransformLayer, self).__init__()
        self.wavelet = wavelet  # Gaussian wavelets, e.g., 'gaus1', 'gaus2'

    def forward(self, x):
        batch_size, seq_len, channels = x.shape  # x.shape = [batch_size, seq_len, channels]
        wavelet_coeffs = []

        for i in range(batch_size):
            for c in range(channels):
                # Perform discrete wavelet transform (DWT) for Gaussian wavelets
                cA, cD = pywt.dwt(x[i, :, c].detach().cpu().numpy(), self.wavelet)
                
                # Combine cA and cD for each signal as needed
                coeffs = torch.tensor(cA, dtype=torch.float32).to(x.device)  # Only using cA for simplicity
                coeffs_tensor = torch.cat((torch.tensor(cA), torch.tensor(cD)), dim=0).to(x.device)
                wavelet_coeffs.append(coeffs_tensor.unsqueeze(0))  # Add batch dimension

        # Stack and reshape the wavelet coefficients for batch and channel processing
        stacked_coeffs = torch.stack(wavelet_coeffs)  # Shape: (batch_size * channels, 1, combined coefficients)
        x_wavelet = stacked_coeffs.view(batch_size, channels, -1)  # Reshape as needed

        return x_wavelet.transpose(1, 2)  # Final shape: (batch_size, seq_len, channels)



        


"""class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn"""


"""
Encoder Layer (EncoderLayer): It integrates wavelet-based
multi-scale features and a dilated convolution-based feedforward mechanism.
This setup aims to capture both local and global temporal features effectively.
"""

"""
class EncoderLayer(nn.Module):
    #def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", lstm_hidden_dim=None, num_lstm_layers=10):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.lstm_hidden_dim = lstm_hidden_dim if lstm_hidden_dim is not None else d_model
        self.lstm = nn.LSTM(d_model, self.lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True, bidirectional=False)

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        lstm_out, _ = self.lstm(x)
        #new_x, attn = self.attention(x, x, x,attn_mask = attn_mask)
        new_x, attn = self.attention(lstm_out, lstm_out, lstm_out, attn_mask = attn_mask)
        new_x = new_x + lstm_out

        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm2(x+y), attn
"""

class EncoderLayerM(nn.Module):
    #def __init__(self, attention, d_model, d_ff, dropout= 0.1 ,activation='relu', wavelet='db1'  ):
    def __init__(self, attention, d_model, d_ff, dropout= 0.1 ,activation='relu', wavelet='gaus1'  ):
        super(EncoderLayerM, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        #self.wavelet_transform = WaveletTransformLayer(wavelet=wavelet)
        self.wavelet_transform = GaussianTransformLayerC2(wavelet=wavelet)
        #self.lstm = nn.LSTM()
        self.conv1 = DilatedConvLayer(d_model, d_ff , dilation=2)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation=='relu' else F.gelu

    def forward(self, x, attn_mask=None):
        # Wavelet Transform for multi-scale feature extraction
        x = self.wavelet_transform(x)

        """
        for s in range(x.shape[1]):
            scale_x = x[:, s, :, :]
            new_scale_x, attn = attn_layer(scale_x, scale_x, scale_x, attn_mask=attn_mask)  

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
        ........ etc

            
        """
    

        # Attention mechanism
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)    #  attn_mask ????????
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # Dilated Convolution and Feedforward Network
        y = x.transpose(-1, 1)   # [B, D, L] for Conv1d
        y = self.conv1(y)         # Dilated Conv
        y = self.dropout(self.activation(self.conv2(y)))
        y = y.transpose(-1, 1)    # [B, L, D] back for LayerNorm

        # Residual connection and normalization
        return self.norm2(x + y), attn

class EncoderLayer(nn.Module):
    #def __init__(self, attention, d_model, d_ff, dropout= 0.1 ,activation='relu', wavelet='db1'  ):
    def __init__(self, attention, d_model, d_ff, dropout= 0.1 ,activation='relu', wavelet='gaus1'  ):
    #def __init__(self, attention, d_model, d_ff, dropout= 0.1 ,activation='relu', wavelet='gaus1' , lstm_hidden_dim=None, num_lstm_layers = 50 ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # LSTM Before Attention
        #self.lstm_hidden_dim = lstm_hidden_dim if lstm_hidden_dim is not None else d_model
        #self.lstm = nn.LSTM(d_model, self.lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True, bidirectional=False)
        #self.wavelet_transform = WaveletTransformLayer(wavelet=wavelet)
        self.conv_temp_poss = TemporalConv1D(input_channels=d_model, output_channels=d_model, kernel_size=1)
        self.wavelet_transform = GaussianTransformLayerC3(wavelet=wavelet)
        self.conv1 = DilatedConvLayer(d_model, d_ff , dilation=2)
        #self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
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
            #print(f'scale_x.shape:{scale_x.shape}')
            #lstm_out, _ = self.lstm(scale_x)
            scale_x = self.conv_temp_poss(scale_x.transpose(-1, 1))      #   x [B, L, D]  -----> [B, D, L]
            scale_x = scale_x.transpose(-1, 1)                           #     [B, D, L]  -----> [B, L, D]


            new_scale_x, attn = self.attention(scale_x, scale_x, scale_x, attn_mask=attn_mask) 
            #new_scale_x , attn = self.attention(lstm_out, lstm_out, lstm_out, attn_mask=attn_mask)

            #new_scale_x = lstm_out + self.dropout(new_scale_x)
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

batch_size = 8
seq_len = 10
num_scales = 3
channels = 5

#x = torch.rand(batch_size, seq_len, channels)

#model = GaussianTransformLayerC3(wavelet='gaus1')

#model = EncoderLayer(attention, d_model, d_ff, dropout= 0.1 ,activation='relu', wavelet='gaus1' )

#output = model(x)   # output.shape = [batch_size, num_scales, channels, seq_len]

#print(output.shape)
        

            


"""class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
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
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns"""




class Encoder(nn.Module):   #       Encoder_arxikos
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, wavelet='gaus1'  ):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm = norm_layer
        #self.wavelet_transform = WaveletTransformLayer(wavelet) # Global Wavelet Transform
        #self.wavelet_transform = GaussianTransformLayerC2(wavelet) 
        self.wavelet_transform = GaussianTransformLayerC3(wavelet) 
        

    def forward(self, x, attn_mask=None):
        
        #print(f'x_enc.shape:{x.shape}')
        #x = self.wavelet_transform(x)
        #print(f'x_enc_after_wav.shape:{x.shape}')
        attns = []
        #for s in range(x.shape[1]):
            # scale_x = x[:, s, :, :]
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

"""
  
class Encoder(nn.Module):    # The one we used in the Results former Encoder for 2 Wavelets.
    #def __init__(self, attn_layers, conv_layers=None, norm_layer=None, wavelet='gaus1'  ): lstm_hidden_dim=None, num_lstm_layers = 50
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, wavelet='gaus1', lstm_hidden_dim=None, num_lstm_layers=50  ):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm = norm_layer
        #self.wavelet_transform = WaveletTransformLayer(wavelet) # Global Wavelet Transform
        #self.wavelet_transform = GaussianTransformLayerC2(wavelet) 
        self.wavelet_transform = GaussianTransformLayerC3(wavelet)
        # LSTM Before Attention
        #self.lstm_hidden_dim = lstm_hidden_dim if lstm_hidden_dim is not None else d_model
        #self.lstm = nn.LSTM(d_model, self.lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True, bidirectional=False)
        

    def forward(self, x, attn_mask=None):
        
        #print(f'x_enc.shape:{x.shape}')
        x = self.wavelet_transform(x)                     #  if we want wavelet transformer before the entrance in EncoderLayers
        #print(f'x_enc_after_wav.shape:{x.shape}')
        attns = []
        scale_outputs = []
        for s in range(x.shape[1]):
            scale_x = x[:, s, :, :]
            #print(f'scale_x.shape:{scale_x.shape}')
            if self.conv_layers is not None:

                for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):

                    # Before the input to go to attn we feed it to a LSTM network, extra code
                    #lstm_out, _ = self.lstm(scale_x)   # Fix: Extract only LSTM output
                    

                    scale_x, attn = attn_layer(scale_x, attn_mask=attn_mask)   # previous model without LSTM 
                    #scale_x, attn = attn_layer(lstm_out, attn_mask=attn_mask)
                    #scale_x = scale_x + lstm_out

                    scale_x = conv_layer(scale_x)
                    attns.append(attn)
                scale_x, attn = self.attn_layers[-1](scale_x, attn_mask=attn_mask)
                attns.append(attn)
            else:
                for attn_layer in self.attn_layers:
                    scale_x, attn = attn_layer(scale_x, attn_mask=attn_mask)
                    attns.append(attn)
            scale_outputs.append(scale_x.unsqueeze(1))   # Add the scales dim back

        # Concatenate processed scales and compute mean over scale dimension
        x = torch.cat(scale_outputs, dim = 1)
        x = x.mean(dim=1)  # Average over scales

        
        if self.norm is not None:
            x = self.norm(x)
            
        #print(f'encoder output:{x.shape}')

        
        return x, attns

"""

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



"""
In PyTorch, the dimensions of a 3D tensor [B, L, D] generally represent:

B: Batch size (the number of samples in a batch).
L: Sequence length (the number of time steps or the length of each sequence).
D: Feature dimension (the number of features per time step)
"""

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=2, batch_first=True)

    def forward(self, query, key, value, attn_mask=None):
        attn_output, attn_weights =  self.attention(query, key, value, attn_mask=attn_mask)

        return attn_output, attn_weights

# Initialize the EncoderLayer with SelfAttention, DilatedConvLayer, and WaveletTransformLayer
d_model = 16
e_layers = 3
"""
attn_layers = [
    EncoderLayer(SelfAttention(d_model=d_model), d_model=d_model, d_ff=64, dropout=0.1,activation='relu', wavelet='db1') 
    for _ in range(e_layers)
]
"""

# Initialize the Encoder using these layers
#encoder = Encoder(attn_layers=attn_layers)

# Generate a random input
batch_size = 8
seq_len = 50
x = torch.rand(batch_size, seq_len, d_model)  # [B, L, D]

# Pass through encoder
#output, attns = encoder(x)

#print("Encoder Output Shape:", output.shape)  # Expected: [batch_size, seq_len, d_model]
#print("Attention Weights Shape (for each layer):", [attn.shape for attn in attns])

#wav = WaveletTransformLayer(wavelet='db1')
#utput = wav(input)
