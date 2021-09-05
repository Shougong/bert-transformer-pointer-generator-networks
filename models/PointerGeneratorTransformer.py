import sys

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from .model_utils import PositionalEncoding, _generate_square_subsequent_mask, Embedding
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer
from .decoder import TransformerDecoder
from transformers import BertModel

import copy
import logging

class PointerGeneratorTransformer(nn.Module):
    def __init__(self, rank=0, src_vocab_size=128, tgt_vocab_size=128,
                 embedding_dim=768, fcn_hidden_dim=768,
                 num_heads=8, num_layers=6, dropout=0.2,
                 max_len=200, src_to_tgt_vocab_conversion_matrix=None):
        super(PointerGeneratorTransformer, self).__init__()
        
        self.rank = rank
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.src_to_tgt_vocab_conversion_matrix = src_to_tgt_vocab_conversion_matrix
        # self.pos_encoder = PositionalEncoding(embedding_dim, max_seq_len=max_len)

        # Encoder layers
        self.encoder = BertModel.from_pretrained('bert-base-chinese')
        
        # Source and target embeddings
        # using word embeddings from pre-trained bert
        self.tgt_embed = copy.deepcopy(self.encoder.embeddings)

        # Decoder layers
        self.decoder = TransformerDecoder(num_layers)

        # Final linear layer + softmax. for probability over target vocabulary
        self.p_vocab = nn.Sequential(
            nn.Linear(self.embedding_dim, self.tgt_vocab_size),
            nn.Softmax(dim=-1))

        # P_gen, probability of generating output
        self.p_gen = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, 1),
            nn.Sigmoid())

        # Context vector
        self.c_t = None

        # Initialize weights of model
        self._reset_parameters()

    def _reset_parameters(self):
        """ Initiate parameters in the transformer model. """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def encode(self, src_input_ids, src_token_type_ids, src_attention_masks):
        """
        Applies embedding, positional encoding and then runs the transformer encoder on the source
        :param src: source tokens batch
        :param src_attention_masks: source padding mask
        :return: memory- the encoder hidden states
        """
        # Pass the source to the encoder
        memory = self.encoder(src_input_ids, token_type_ids=src_token_type_ids, attention_mask=src_attention_masks)[0]
        return memory

    def decode(self, memory, tgt, src, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Applies embedding, positional encoding on target  and then runs the transformer encoder on the memory and target.
        Also creates square subsequent mask for teacher learning.
        :param memory: The encoder hidden states
        :param tgt: Target tokens batch
        :param tgt_key_padding_mask: target padding mask
        :param memory_key_padding_mask: memory padding mask
        :param has_mask: Whether to use square subsequent mask for teacher learning
        :return: decoder output
        """
        # Create target mask for transformer if no appropriate one was created yet, created of size (T, T)

        # Target embedding and positional encoding, changes dimension (N, T) -> (N, T, E) -> (T, N, E)
        tgt_embed = self.tgt_embed(tgt).transpose(0, 1)
        
        # Get output of decoder and attention weights. decoder Dimensions stay the same
        decoder_output, attention = self.decoder(tgt_embed, memory,
                                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                                 memory_key_padding_mask=memory_key_padding_mask)
        
        # Get target vocabulary distribution, (T, N, E) -> (T, N, tgt_vocab_size)
        # logits
        p_vocab = self.p_vocab(decoder_output)

        # ---Compute Pointer Generator probability---
        # Get hidden states of source (easier/more understandable computation). (S, N, E) -> (N, S, E)
        hidden_states = memory.transpose(0, 1)
        # compute context vectors. (N, T, S) x (N, S, E) -> (N, T, E)
        context_vectors = torch.matmul(attention[-1], hidden_states).transpose(0, 1)
        total_states = torch.cat((context_vectors, decoder_output, tgt_embed), dim=-1)
        # Get probability of generating output. (N, T, 3*E) -> (N, T, 1)
        p_gen = self.p_gen(total_states)
        # Get probability of copying from input. (N, T, 1)
        p_copy = 1 - p_gen

        # Get representation of src tokens as one hot encoding
        one_hot = torch.zeros(src.size(0), src.size(1), self.src_vocab_size, device=src.device)
        one_hot = one_hot.scatter_(dim=-1, index=src.unsqueeze(-1), value=1)
        # p_copy from source is sum over all attention weights for each token in source
        p_copy_src_vocab = torch.matmul(attention[-1], one_hot)
        
        # convert representation of token from src vocab to tgt vocab
        # src vocab equals to tgt vocab
        p_copy_tgt_vocab = p_copy_src_vocab.transpose(0, 1)
        # Compute final probability
        p = torch.add(p_vocab * p_gen, p_copy_tgt_vocab * p_copy)
        
        # Change back batch and sequence dimensions, from (T, N, tgt_vocab_size) -> (N, T, tgt_vocab_size)
        return torch.log(p.transpose(0, 1))

    # def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None,
    #             memory_key_padding_mask=None, has_mask=True):
    def forward(self, src_input_ids, src_token_type_ids, src_attention_masks, 
                tgt_input_ids, tgt_token_type_ids, tgt_attention_masks, has_mask=True):
        """Take in and process masked source/target sequences.

		Args:
			src: the sequence to the encoder (required).
			tgt: the sequence to the decoder (required).
			src_mask: the additive mask for the src sequence (optional).
			tgt_mask: the additive mask for the tgt sequence (optional).
			memory_mask: the additive mask for the encoder output (optional).
			src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

		Shape:
			- src: :math:`(S, N, E)`. Starts as (N, S) and changed after embedding
			- tgt: :math:`(T, N, E)`. Starts as (N, T) and changed after embedding
			- src_mask: :math:`(S, S)`.
			- tgt_mask: :math:`(T, T)`.
			- memory_mask: :math:`(T, S)`.
			- src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

			Note: [src/tgt/memory]_mask should be filled with
			float('-inf') for the masked positions and float(0.0) else. These masks
			ensure that predictions for position i depend only on the unmasked positions
			j and are applied identically for each sequence in a batch.
			[src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
			that should be masked with float('-inf') and False values will be unchanged.
			This mask ensures that no information will be taken from position i if
			it is masked, and has a separate mask for each sequence in a batch.

			- output: :math:`(T, N, E)`.

			Note: Due to the multi-head attention architecture in the transformer model,
			the output sequence length of a transformer is same as the input sequence
			(i.e. target) length of the decode.

			where S is the source sequence length, T is the target sequence length, N is the
			batch size, E is the feature number

		Examples:
			output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
		"""

        # using pre-trained bert as encoder, shape of output => [batch size, seq_len, embed_dim]
        memory = self.encode(src_input_ids, src_token_type_ids, src_attention_masks).transpose(0, 1)
        # change attention mask to match the form of torch.nn.Transformer
        src_attention_masks = ((1 - src_attention_masks) > 0)
        tgt_attention_masks = ((1 - tgt_attention_masks) > 0)
        # Applies embedding, positional encoding on target  and then runs the transformer encoder on the memory and target.
        output = self.decode(memory, tgt_input_ids, src_input_ids, tgt_attention_masks, src_attention_masks)
        return output