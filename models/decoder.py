import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.modules import TransformerDecoderLayer
# from torch.nn.modules.activation import MultiheadAttention
from torch.nn import Linear, Dropout
from torch.nn import LayerNorm

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model=768, dim_feedforward=3072, nhead=12, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        '''
        Input:
            tgt: target input ids
            memory: hidden states of source input ids
            tgt_mask: mask special target token ids, default is None 
            memory_mask: mask special input token ids, default is None
            tgt_key_padding_mask: mask [PAD] in target token ids
            memory_key_padding_mask: mask [PAD] in memory token ids
        '''
        attn_dists = []
        for i in range(self.num_layers):
            # self_attention
            tgt2 = self.self_attn(tgt, tgt, tgt,  attn_mask=tgt_mask, 
                                    key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            # Model saves attention weights from multi-head-attn
            tgt2, attn_dist = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

            attn_dists.append(attn_dist)    # add attention weights to attn_dists

            # feed forward neural network
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
            tgt = tgt + self.dropout3(tgt2)
            tgt = self.norm3(tgt)

        return tgt, attn_dists


