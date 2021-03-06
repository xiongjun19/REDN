import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter

from .base_model import SentenceRE


def swish(x):
    return x * F.sigmoid(x)


class PARAM(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id, num_token_labels, subject_1=False, use_cls=True):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.use_cls = use_cls
        self.subject_1 = subject_1
        self.num_token_labels = num_token_labels
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        hidden_size = self.sentence_encoder.hidden_size
        self.fc = nn.Linear(hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

        self.subject_output_fc = nn.Linear(hidden_size, self.num_token_labels)
        # self.bias = Parameter(torch.zeros((num_class, seq_len, seq_len)))
        LAYER = 1  
        for l in range(LAYER):
            setattr(self, f"query_{l}", nn.Linear(hidden_size, hidden_size))
            setattr(self, f"key_{l}", nn.Linear(hidden_size, hidden_size))
            setattr(self, f"value_{l}", nn.Linear(hidden_size, hidden_size))

        self.layer_num = LAYER

        self.attn_score = MultiHeadAttention(input_size=hidden_size,
                                             output_size=num_class * hidden_size,
                                             num_heads=num_class)
        self.c_lin = nn.Linear(num_class, 1) 
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(hidden_size, 1e-12) 

    def infer(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score

    def forward(self, token, att_mask):
        # import ipdb; ipdb.set_trace()
        rep, hs, atts = self.sentence_encoder(token, att_mask)  # (B, H)
        if self.subject_1:
            subject_output = hs[-1]  # BS * SL * HS
        else:
            subject_output = hs[-2]  # BS * SL * HS
        subject_output_logits = self.subject_output_fc(subject_output)  # BS * SL * NTL

        if self.use_cls:
            subject_output = subject_output + rep.view(-1, 1, rep.shape[-1])

        query = subject_output
        value = subject_output
        key = hs[-1]

        for l in range(self.layer_num):
            q = getattr(self, f"query_{l}")
            k = getattr(self, f"key_{l}")
            v = getattr(self, f"value_{l}")
            f = swish
            org_val = value
            query = q(query)
            key = k(key)
            value = v(value)
            query = self.dropout(query)
            key = self.dropout(key)
            value = self.dropout(value)
            query = f(query)
            key = f(key)
            value = f(value)
            att = self.attn_score(key, query)  # BS * NTL * SL * SL
            value = self._update_val(att, value, org_val, att_mask)  
            query = value

        score = self.attn_score(key, query)  # BS * NTL * SL * SL
        score = score.sigmoid()
        return score, subject_output_logits, atts[-1]

    def _update_val(self, att, value, org_val, att_mask):
        # import ipdb; ipdb.set_trace()
        att_mask = att_mask[:, None, None, :]
        att_mask = (1 - att_mask) * -1000.0
        att = att_mask + att
        attention_probs = nn.Softmax(dim=-1)(att) # TODO adding att_mask
        # attention_probs = self.dropout(attention_probs)
        val_expand = value.unsqueeze(1)
        new_val = torch.matmul(attention_probs, val_expand)
        new_val = new_val.permute([0, 2, 3, 1])  
        new_val = self.dropout(new_val)
        new_val = self.c_lin(new_val)
        new_val = swish(new_val)
        new_val = new_val.squeeze(-1)
        value = org_val + new_val # TODO adding layer_norm
        value = self.norm(value)
        return value


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_size, output_size, num_heads, output_attentions=False):
        super(MultiHeadAttention, self).__init__()
        self.output_attentions = output_attentions
        self.num_heads = num_heads
        self.d_model_size = input_size

        self.depth = int(output_size / self.num_heads)

        self.Wq = torch.nn.Linear(input_size, output_size)
        self.Wk = torch.nn.Linear(input_size, output_size)
        self.dropout = torch.nn.Dropout(0.2)
        self.dim_heads = 8
        self.dim_lin = torch.nn.Linear(self.dim_heads, 1)

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)  # BS * SL * NH * H
        x = x.permute([0, 2, 1, 3])  # BS * NH * SL * H
        xx = x.reshape([x.shape[0], x.shape[1], x.shape[2], self.dim_heads, -1])
        xx = xx.permute([0, 1, 3, 2, 4])
        return xx

    def forward(self, k, q):  # BS * SL * HS
        batch_size = q.shape[0]

        q = self.Wq(q)  # BS * SL * OUT
        k = self.Wk(k)  # BS * SL * OUT

        # q = F.dropout(q, 0.8, training=self.training)
        # k = F.dropout(k, 0.8, training=self.training)

        q = self.split_into_heads(q, batch_size)  # BS * NH * SL * H
        k = self.split_into_heads(k, batch_size)  # BS * NH * SL * H

        attn_score = torch.matmul(q, k.permute(0, 1, 2, 4, 3))
        attn_score = attn_score / np.sqrt(k.shape[-1])
        attn_score = attn_score.permute([0, 1, 3, 4, 2])
        attn_score = self.dropout(attn_score)
        attn_score = self.dim_lin(attn_score).squeeze(-1)
        return attn_score
