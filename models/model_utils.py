import math
import torch
import torch.nn as nn

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device), diagonal=1).bool()
    return subsequent_mask

""" MASKS UTILS """
def _generate_subsequent_mask(src_sz, tgt_sz):
    mask = (torch.triu(torch.ones(src_sz, tgt_sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def _generate_square_subsequent_mask(sz):
    return _generate_subsequent_mask(sz, sz)

def padding_trg(trg_ids, trg_ground_ids, trg_key_padding_mask, max_len):
    trg_mask = [1] * len(trg_ids)
    trg_padding = [0] * (max_len - len(trg_ids))
    trg_ids += trg_padding
    trg_ground_ids += trg_padding
    trg_mask += trg_padding
    return trg_ids, trg_ground_ids, trg_mask

""" EMBEDDING UTILS """
def Embedding(pretrained_embeddings):
    """ Generates embeddings for tokens in vocabulary
        Weights initialized with mean=0 and std=sqrt(embedding_dim)"""
    m = nn.Embedding.from_pretrained(weight)
    return m


""" POSITIONAL ENCODING UTILS """
class PositionalEncoding(nn.Module):
    """ Adds positional encoding to sequences """
    def __init__(self, embedding_dim, dropout=0.1, max_seq_len=100):
        """ Initializes a seq_len x 1 x embedding_dim positional encoding matrix"""
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ Adds positional encoding to the input.
            Input of dimensions (seq_len x batch_sz x embedding_dim).
            Adds positional encoding matrix (seq_len x 1 x embedding_dim) to every individual example in batch """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)