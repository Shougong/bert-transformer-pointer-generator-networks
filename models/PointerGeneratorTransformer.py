import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_

from .model_utils import PositionalEncoding, Embedding, get_subsequent_mask, _generate_square_subsequent_mask
# from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer
# from torch.nn.modules import TransformerDecoder
from .decoder import TransformerDecoder

from transformers import BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings

import copy
import logging

class PointerGeneratorTransformer(nn.Module):
    def __init__(self, rank=0, src_vocab_size=128, tgt_vocab_size=128,
                 inv_vocab=None, pad_id=0,
                 embedding_dim=768, fcn_hidden_dim=3072,
                 num_heads=12, num_layers=8, dropout=0.1,
                 max_len=200):
        super(PointerGeneratorTransformer, self).__init__()
        
        self.rank = rank
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        
        # Encoder layers
        self.config = BertConfig.from_pretrained('bert-base-chinese')
        self.encoder = BertModel.from_pretrained('bert-base-chinese')
        
        # Source and target embeddings
        # using word embeddings from pre-trained bert
        self.tgt_embed = copy.deepcopy(self.encoder.embeddings)

        # Decoder layers
        # self.decoder_layer = TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dim_feedforward=fcn_hidden_dim, dropout=dropout)
        self.decoder = TransformerDecoder(num_layers)

        # Final linear layer + softmax. for probability over target vocabulary
        # self.last_linear = nn.Linear(self.embedding_dim, self.tgt_vocab_size)
        self.p_vocab = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Linear(self.embedding_dim, self.tgt_vocab_size),
            nn.Softmax(dim=-1))

        self.p_gen = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, 1),
            nn.Sigmoid())
        
        # Initialize masks
        self.src_mask = None
        self.tgt_mask = None
        self.mem_mask = None

    def encode(self, src_input_ids, src_attention_masks):
        """
        Applies embedding, positional encoding and then runs the transformer encoder on the source
        :param src: source tokens batch
        :param src_attention_masks: source padding mask
        :return: memory- the encoder hidden states
        """
        # Pass the source to the encoder
        memory = self.encoder(src_input_ids, attention_mask=src_attention_masks)[0]
        return memory

    def decode(self, memory, tgt, src, tgt_key_padding_mask=None, memory_key_padding_mask=None, has_mask=True):
        """
        Applies embedding, positional encoding on target  and then runs the transformer encoder on the memory and target.
        Also creates square subsequent mask for teacher learning.
        """
        # Create target mask for transformer if no appropriate one was created yet, created of size (T, T)
        tgt_mask = get_subsequent_mask(tgt)

        # Target embedding and positional encoding, changes dimension (N, T) -> (N, T, E) -> (T, N, E)
        tgt_embed = self.tgt_embed(tgt).transpose(0, 1)
        
        # Get output of decoder and attention weights. decoder Dimensions stay the same
        decoder_output, attention = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask,
                                                memory_mask=None,
                                                tgt_key_padding_mask=tgt_key_padding_mask,
                                                memory_key_padding_mask=memory_key_padding_mask)
        decoder_output = decoder_output.transpose(0, 1)
        
        # hidden states of src input, [bz, seq_len, embed_dim]
        hidden_states = memory.transpose(0, 1)
        
        # context vector, [bz, seq_len, embed_dim]
        context_vectors = torch.matmul(attention[-1], hidden_states)
        vocab_dist = self.p_vocab(torch.cat((decoder_output, context_vectors), dim=-1))

        # total_states for p_gen, total_states => [bz, seq_len, 1]
        total_states = torch.cat((context_vectors, decoder_output, tgt_embed.transpose(0, 1)), dim=-1)
        p_gen_prob = self.p_gen(total_states)

        one_hot = torch.zeros(src.size(0), src.size(1), self.src_vocab_size, device=src.device)
        one_hot = one_hot.scatter_(dim=-1, index=src.unsqueeze(-1), value=1)

        p_copy_vocab = torch.matmul(attention[-1], one_hot)
        
        p = torch.add(p_gen_prob * vocab_dist, p_copy_vocab * (1.0 - p_gen_prob))
        return p

    def forward(self, src_input_ids, src_attention_masks, tgt_input_ids, tgt_attention_masks):
        """Take in and process masked source/target sequences.
		"""

        # using pre-trained bert as encoder, shape of output => [batch size, seq_len, embed_dim]
        memory = self.encode(src_input_ids, src_attention_masks).transpose(0, 1)
        # change padding mask to match the form of torch.nn.Transformer
        src_key_padding_masks = ((1 - src_attention_masks) > 0)
        tgt_key_padding_masks = ((1 - tgt_attention_masks) > 0)
        # Applies embedding, positional encoding on target  and then runs the transformer encoder on the memory and target.
        output = self.decode(memory, tgt_input_ids, src_input_ids, tgt_key_padding_masks, src_key_padding_masks)
        return output