import torch
import torch.nn as nn
import torch.nn.functional as F

#from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
import pywt


# ------------------- Dilated Convolution Layer --------------------
class DilatedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(DilatedConvLayer, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
    
    def forward(self, x):
        return F.relu(self.conv1d(x))

# ------------------- Wavelet Transform Layer -----------------------
class WaveletTransformLayer(nn.Module):
    def __init__(self, wavelet='db1'):
        super(WaveletTransformLayer, self).__init__()
        self.wavelet = wavelet

    def forward(self, x):
        batch_size, channels, seq_len = x.shape
        wavelet_coeffs = []

        for i in range(batch_size):
            for c in range(channels):
                # Perform 1D discrete wavelet transform (DWT) on each time series
                coeffs = pywt.dwt(x[i, c].cpu().numpy(), self.wavelet)
                cA, cD = coeffs  # Approximation and Detail coefficients
                wavelet_coeffs.append(torch.tensor(cA).unsqueeze(0).to(x.device))  # Approximation part
                wavelet_coeffs.append(torch.tensor(cD).unsqueeze(0).to(x.device))  # Detail part

        return torch.stack(wavelet_coeffs).view(batch_size, channels * 2, seq_len)



class LSTMTemporalFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=50, bidirectional=False):  #bidirectional=False)
        super(LSTMTemporalFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=False, bidirectional = bidirectional)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bideractional = bidirectional

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)   # lstm_out shape: [batch_size, seq_len, hidden_dim * num_directions = d_model * 1 (on that case)]
        return  lstm_out 


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

         # LSTM Temporal Feature Extractor
        self.lstm_feature_extractor = LSTMTemporalFeatureExtractor(input_dim=enc_in, hidden_dim=d_model)


        """
        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        """

        # Encoding
        self.enc_embedding = DataEmbedding(d_model, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        # Feature Extractor
        lstm_out = self.lstm_feature_extractor(x_enc)         # [batch_size, seq_len, d_model]

        # Embedding
        #enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.enc_embedding(lstm_out, x_mark_enc)    # changed to be suitable for the lstm feature extractor
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        self.lstm_feature_extractor = LSTMTemporalFeatureExtractor(input_dim=enc_in, hidden_dim=d_model)

        """
        # Encoding
        #self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        #self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        """


        # Encoding
        self.enc_embedding = DataEmbedding(d_model, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        # LSTM Feature Extraction
        lstm_out = self.lstm_feature_extractor(x_enc)  # [batch_size, seq_len, d_model]

        #enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.enc_embedding(lstm_out, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
