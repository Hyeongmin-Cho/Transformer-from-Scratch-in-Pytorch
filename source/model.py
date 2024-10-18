import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import re
import random
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
import math
import copy

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model


    def calculate_angle(self, seq_len):
        pos = torch.arange(seq_len).unsqueeze(1)
        div = torch.exp(-1 * (torch.arange(0, self.d_model, 2) / self.d_model) * np.log(10000))
        angle = pos * div
        return angle

    def pos_encoding(self, shape):
        seq_len = shape[1] # get seq_len
        angles = self.calculate_angle(seq_len)
        sin, cos = torch.sin(angles), torch.cos(angles)
        pos_enc = torch.zeros(shape, requires_grad=False)
        pos_enc[:, :, 0::2] = sin
        pos_enc[:, :, 1::2] = cos
        return pos_enc

    def forward(self, x):
        return x + self.pos_encoding(x.shape).to(x.device)
    

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)

    def forward(self, x):
        output = self.embedding(x)
        output = self.pos_enc(output)
        return output
    
    
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, 'd_model \% num_heads is not Zero.'
        self.attn_dim = d_model // num_heads

        self.query_fc = nn.Linear(d_model, self.attn_dim * self.num_heads)
        self.key_fc = nn.Linear(d_model, self.attn_dim * self.num_heads)
        self.value_fc = nn.Linear(d_model, self.attn_dim * self.num_heads)
        self.out_fc = nn.Linear(self.attn_dim * self.num_heads, d_model)

    def split_head(self, x, fc):
        # x: (batch, seq_len, d_model)
        # fc: (d_model, attn_dim * num_heads)
        n_batch = x.shape[0]
        output = fc(x) # (batch, seq_len, attn_dim * num_heads)
        output = output.reshape(n_batch, -1, self.num_heads, self.attn_dim) # (batch, seq_len, num_heads, attn_dim)
        output = output.transpose(1, 2) # (batch, num_heads, seq_len, attn_dim)
        return output

    def scaled_dot_product_attention(self, WQ, WK, WV, attn_mask):
        # WQ, WK, WV: (batch, num_heads, seq_len, attn_dim)
        attn_dim = WQ.shape[-1]
        logits = torch.matmul(WQ, WK.transpose(-1, -2)) # (batch, num_heads, seq_len, seq_len)
        logits = logits / np.sqrt(attn_dim) # Scaling
        
        if attn_mask is not None:
            logits = logits.masked_fill(attn_mask == 0, -1e9)

        attn_score = F.softmax(logits, dim=-1) # (batch, num_heads, seq_len, seq_len)
        output = torch.matmul(attn_score, WV) # (batch, num_heads, seq_len, attn_dim)
        return output


    def forward(self, query, key, value, mask=None):
        n_batch = query.shape[0]
        WQ = self.split_head(query, self.query_fc)
        WK = self.split_head(key, self.key_fc)
        WV = self.split_head(value, self.value_fc)

        scaled_attn = self.scaled_dot_product_attention(WQ, WK, WV, mask) # (batch, num_heads, seq_len, attn_dim)
        scaled_attn = scaled_attn.transpose(1, 2) # (batch, seq_len, num_heads, attn_dim)
        scaled_attn = scaled_attn.reshape(n_batch, -1, self.d_model)

        output = self.out_fc(scaled_attn) # (batch, seq_len, d_model)

        return output

def create_padding_mask(query, key, pad_idx=1):
    # query, key: (n_batch, seq_len) 임베딩 레이어 통과 전 생성
    query_seq_len, key_seq_len = query.shape[1], key.shape[1]

    query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3) # (n_batch, 1, query_seq_len, 1)
    query_mask = query_mask.repeat(1, 1, 1, key_seq_len) # (n_batch, 1, query_seq_len, key_seq_len)

    key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2) # (n_batch, 1, 1, key_seq_len)
    key_mask = key_mask.repeat(1, 1, query_seq_len, 1) # (n_batch, 1, query_seq_len, key_seq_len)

    mask = query_mask & key_mask
    mask.requires_grad = False
    return mask

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention = MultiheadAttention(d_model, num_heads)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)


    def forward(self, x, mask):
        # Multi-Head Self-Attention 후 Dropout, Residual Connection, Layer Normalization
        attn = self.attention(query=x, key=x, value=x, mask=mask)
        attn = F.dropout(attn, self.dropout)
        attn = F.layer_norm(x+attn, normalized_shape=(self.d_model,))

        #Position-wise Feed-forward 후 Dropout, Residual Connection, Layer Normalization
        output = F.relu(self.fc1(attn))
        output = self.fc2(output)
        output = F.dropout(output, self.dropout)
        output = F.layer_norm(attn+output, normalized_shape=(self.d_model,))

        return output
    
class Encoder(nn.Module):
    def __init__(self, encoder_layer, n_layers):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(n_layers)])

    def forward(self, x, mask):
        output = x
        for layer in self.layers:
            output = layer(output, mask)
        return output
    
def create_lookahead_mask(query, key, pad_idx=1):
    # query, key: (n_batch, seq_len) 임베딩 레이어 통과 전 생성
    query_seq_len, key_seq_len = query.shape[1], key.shape[1]
    lower_triangular_matrix = np.tril(np.ones((query_seq_len, key_seq_len)))
    mask = torch.Tensor(lower_triangular_matrix).type(torch.bool)
    mask.requires_grad = False
    mask = mask.to(query.device)
    return mask

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention1 = MultiheadAttention(d_model, num_heads) # for Self-Attention
        self.attention2 = MultiheadAttention(d_model, num_heads) # Encoder-Decoder Attention
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x, enc_out, mask, look_ahead_mask):
        # Multi-Head Self-Attention 후 Dropout, Residual Connection, Layer Normalization
        attn1 = self.attention1(query=x, key=x, value=x, mask=look_ahead_mask)
        attn1 = F.dropout(attn1, self.dropout)
        attn1 = F.layer_norm(x+attn1, normalized_shape=(self.d_model,))


        # Multi-Head Attention (Enc-Dec) 후 Dropout, Residual Connection, Layer Normalization
        attn2 = self.attention2(query=attn1, key=enc_out, value=enc_out, mask=mask)
        attn2 = F.dropout(attn2, self.dropout)
        attn2 = F.layer_norm(attn1+attn2, normalized_shape=(self.d_model,))

        # Position-wise Feed-forward 후 Dropout, Residual Connection, Layer Normalization
        output = F.relu(self.fc1(attn2))
        output = F.dropout(self.fc2(output), self.dropout)
        output = F.layer_norm(attn2+output, normalized_shape=(self.d_model,))

        return output
    
class Decoder(nn.Module):
    def __init__(self, decoder_layer, n_layers):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(n_layers)])

    def forward(self, x, enc_out, mask, look_ahead_mask):
        output = x
        for layer in self.layers:
            output = layer(output, enc_out, mask, look_ahead_mask)
        return output
    
    
class Transformer(nn.Module):
    def __init__(self, enc_embedder, dec_embedder, encoder, decoder, classifier):
        super(Transformer, self).__init__()
        self.enc_embedder = enc_embedder
        self.dec_embedder = dec_embedder
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier

    def create_padding_mask(self,query, key, pad_idx=1):
        # query, key: (n_batch, seq_len) 임베딩 레이어 통과 전 생성
        query_seq_len, key_seq_len = query.shape[1], key.shape[1]

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3) # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len) # (n_batch, 1, query_seq_len, key_seq_len)

        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2) # (n_batch, 1, 1, key_seq_len)
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1) # (n_batch, 1, query_seq_len, key_seq_len)

        mask = query_mask & key_mask
        mask = mask.type(torch.int).to(query.device)
        mask.requires_grad = False
        return mask

    def create_lookahead_mask(self, query, key, pad_idx=1):
        # query, key: (n_batch, seq_len) 임베딩 레이어 통과 전 생성
        query_seq_len, key_seq_len = query.shape[1], key.shape[1]
        lower_triangular_matrix = np.tril(np.ones((query_seq_len, key_seq_len)))
        mask = torch.Tensor(lower_triangular_matrix).type(torch.int).to(query.device)
        mask.requires_grad = False
        return mask

    def encode(self, x, mask):
        output = self.encoder(self.enc_embedder(x), mask)
        return output

    def decode(self, x, enc_out, mask, look_ahead_mask):
        output = self.decoder(self.dec_embedder(x), enc_out, mask, look_ahead_mask)
        return output

    def forward(self, x_enc, x_dec):
        enc_mask = self.create_padding_mask(x_enc, x_enc)
        dec_mask = self.create_padding_mask(x_dec, x_enc)
        look_ahead_mask = self.create_lookahead_mask(x_dec, x_dec) & self.create_padding_mask(x_dec, x_dec)

        enc_out = self.encode(x_enc, enc_mask)
        z = self.decode(x_dec, enc_out, dec_mask, look_ahead_mask) # (batch, seq_len, d_model)
        y = F.log_softmax(self.classifier(z), dim=-1) # (batch, seq_len, vocab_size)
        return y