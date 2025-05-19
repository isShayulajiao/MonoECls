import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, config, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # Ensure that d_model can be divided by h
        self.d_k = d_model // h  # Each head's dimension
        self.h = h
        self.d_model = d_model
        self.linears = self.clones(nn.Linear(d_model, d_model), 3)
        self.dropout = nn.Dropout(p=dropout)
        self.config = config

    def forward(self, query, key, value, device, mask=None):
        nbatches = query.size(0)  # batch_size
        seq_len = 1

        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        residual = query  # For residual connection

        query, key, value = [
            l(x).view(nbatches, seq_len, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        attn_weight = torch.matmul(query, key.transpose(-1, -2))

        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask == 0, -float('inf'))



        attn_weight = self.dropout(F.softmax(attn_weight, dim=-1))


        context = torch.matmul(attn_weight, value)

        context = context.transpose(1, 2).contiguous().view(nbatches, seq_len, -1)

        # Add residual connection
        context = self.dropout(context)
        residual = residual.view(nbatches, seq_len, -1)
        output = context + residual

        return output

    def clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
