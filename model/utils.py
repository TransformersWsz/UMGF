import torch
import numpy as np
import torch.nn as nn
import copy
import math
from torch.nn import functional as F


def Linear(inputdim, outputdim, bias=True, uniform=True):
    linear = nn.Linear(inputdim, outputdim, bias)
    if uniform:
        nn.init.xavier_uniform_(linear.weight)
    else:
        nn.init.xavier_normal_(linear.weight)
    if bias:
        nn.init.constant_(linear.bias, 0.0)
    return linear

def Embedding(num_embeddings, embedding_dim):
    m = nn.Embedding(num_embeddings, embedding_dim)
    # fairseq and thumt
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m

def prepare_sources(data, padid, share_vocab=True):
    assert data.ndimension() == 2

    if share_vocab:
        # remove <init>
        data = data[:, 1:]

    masks = (data != padid).unsqueeze(1)
    # bytetensor
    return data, masks

def prepare_targets(data, tgtpadid):
    tgt_input = data[:, :-1]
    tgt_input_mask = (tgt_input != tgtpadid).unsqueeze(1).byte()

    subsequent_mask = make_subsequent_mask(tgt_input_mask.size(2))
    subsequent_mask = subsequent_mask.to(data.device)
    tgt_input_mask = tgt_input_mask & subsequent_mask

    tgt_output = data[:, 1:]
    n_tokens = (tgt_output != tgtpadid).detach().sum()
    return tgt_input, tgt_output, tgt_input_mask, n_tokens

def make_subsequent_mask(length, search=False):
    if search:
        shape = (1, length)
    else:
        shape = (length, length)
    subsequent_mask = torch.tril(torch.ones(shape, dtype=torch.uint8))
    subsequent_mask = subsequent_mask.unsqueeze(0)
    return subsequent_mask

def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_num, d_model, dropout=0.0, v=1, output=1, relu=0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % head_num == 0
        self.d_k = d_model // head_num
        self.head = head_num
        self.output = output

        self.userelu = relu

        self.linears = clone(Linear(d_model, d_model), 2)
        if v:
            self.linears.append(Linear(d_model, d_model))
        if output:
            self.linears.append(Linear(d_model, d_model))

        self.v = v
        self.output = output

        self.attn = None
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # when searching, target mask is not needed
        if mask is not None:
            # b 1 t -> b 1 1 t -> b head t t
            mask = mask.unsqueeze(1).expand_as(scores)
            scores.masked_fill_(mask == 0, -1e9)
        p_att = F.softmax(scores, -1)
        if self.dropout:
            p_att = self.dropout(p_att)
        return torch.matmul(p_att, v)

    def forward(self, query, key, value, mask=None):
        # q k v : B T H
        nbatches = query.size(0)
        # b head t dim
        if self.v:
            if self.userelu:
                query, key, value = [F.relu(l(x)).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)
                                 for l, x in zip(self.linears, (query, key, value))]
            else:
                query, key, value = [l(x).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)
                                 for l, x in zip(self.linears, (query, key, value))]
        else:
            if self.userelu:
                query, key = [F.relu(l(x)).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)
                          for l, x in zip(self.linears, (query, key))]
            else:
                query, key = [l(x).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)
                          for l, x in zip(self.linears, (query, key))]
            value = value.view(nbatches, -1, self.head, self.d_k).transpose(1, 2)

        x = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.head * self.d_k)
        if self.output:
            x = self.linears[-1](x)
        # returen b t dim
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = Linear(d_model, d_ff)
        self.w_2 = Linear(d_ff, d_model)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.w_1(x), inplace=True)
        if self.dropout:
            h = self.dropout(h)
        return self.w_2(h)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # compute once in log space
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                             * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # b t dim
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: batch t dim
        # word embedding + position embedding
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))

class SublayerConnectionv2(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnectionv2, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_trans):
        return self.norm(x + self.dropout(x_trans))

class GatedConnection(nn.Module):
    def __init__(self, size, dropout):
        super(GatedConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.gate = Linear(size*2, 1)
        self.norm = nn.LayerNorm(size)
    def forward(self, x, y):
        #y = sublayer(x)
        y = self.dropout(y)
        g = torch.sigmoid(self.gate(torch.cat((x, y), -1)))
        return self.norm(g * y + x)