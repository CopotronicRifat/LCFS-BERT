# -*- coding: utf-8 -*-
# file: lcfs_bert.py
# author: yangheng <yangheng@m.scnu.edu.cn>
# Copyright (C) 2019. All Rights Reserved.

import torch
import torch.nn as nn
import copy
import numpy as np

from pytorch_transformers.modeling_bert import BertPooler, BertSelfAttention, BertConfig


class PointwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_hid, d_inner_hid=None, d_out=None, dropout=0):
        super(PointwiseFeedForward, self).__init__()
        if d_inner_hid is None:
            d_inner_hid = d_hid
        if d_out is None:
            d_out = d_inner_hid
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_inner_hid, d_out, 1)  # position-wise
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return output


class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_tensor = torch.tensor(
            np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_len), dtype=np.float32),
            dtype=torch.float32,
        ).to(self.opt.device)
        SA_out, att = self.SA(inputs, zero_tensor)

        SA_out = self.tanh(SA_out)
        return SA_out, att


class LCFS_BERT(nn.Module):
    def __init__(self, model, opt):
        super(LCFS_BERT, self).__init__()
        if "bert" in opt.pretrained_bert_name:
            hidden = model.config.hidden_size
        elif "xlnet" in opt.pretrained_bert_name:
            hidden = model.config.d_model
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.hidden = hidden
        sa_config = BertConfig(hidden_size=self.hidden, output_attentions=True)

        self.bert_spc = model
        self.bert_g_sa = SelfAttention(sa_config, opt)
        self.bert_g_pct = PointwiseFeedForward(self.hidden)

        self.opt = opt
        self.bert_local = copy.deepcopy(model)
        self.bert_local_sa = SelfAttention(sa_config, opt)
        self.bert_local_pct = PointwiseFeedForward(self.hidden)

        self.dropout = nn.Dropout(opt.dropout)
        self.bert_sa = SelfAttention(sa_config, opt)
        self.mask_scalar = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        # self.mean_pooling_double = nn.Linear(hidden * 2, hidden)
        self.mean_pooling_double = PointwiseFeedForward(hidden * 2, hidden, hidden)
        self.bert_pooler = BertPooler(sa_config)
        self.dense = nn.Linear(hidden, opt.polarities_dim)

    def compute_attention_scores(self, inputs):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2]
        aspect_indices = inputs[3]
        spc_out = self.bert_spc(text_bert_indices, bert_segments_ids)
        spc_att = spc_out[-1][-1]

        attention_scores = spc_att.sum(dim=1).mean(
            dim=-1
        )  # This is just an example, adjust as needed

        return attention_scores
        


    def compute_attention_and_aspect_relevance(self, text_indices, aspect_indices):
        # Compute attention scores
        attention_scores = self.compute_attention_scores(
            (text_indices, None, text_indices, aspect_indices)
        )

        # Compute embeddings for text and aspects
        text_embeddings = self.bert_spc(text_indices)[0]
        aspect_embeddings = self.bert_spc(aspect_indices)[0]

        # Compute similarity scores
        similarity_scores = torch.matmul(text_embeddings, aspect_embeddings.transpose(-1, -2))

        # The similarity_scores tensor might have incorrect dimensions here.
        # You need to ensure that it has dimensions [batch_size, seq_len, num_aspects]
        # or similar, so that each token has a single similarity score per aspect term.

        # Apply scaling factor beta and softmax to compute aspect relevance

        exp_scores = torch.exp(beta * similarity_scores)

        # Ensure that softmax is applied in such a way that each token gets a single relevance score
        aspect_relevance = exp_scores / exp_scores.sum(dim=-1, keepdim=True)

        return attention_scores, aspect_relevance

    def feature_dynamic_mask(self, text_local_indices, aspect_indices, distances_input=None):
        # Ensure tensors are on CPU
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()

        # Convert distances to numpy if it's a tensor
        if isinstance(distances_input, torch.Tensor):
            distances_input = distances_input.cpu().numpy()

        # Initialize masked text raw indices
        masked_text_raw_indices = np.ones(
            (text_local_indices.size(0), self.opt.max_seq_len, self.hidden),
            dtype=np.float32,
        )

        # Compute attention scores and aspect relevance
        attention_scores, aspect_relevance = self.compute_attention_and_aspect_relevance(
            text_local_indices, aspect_indices
        )

        # Iterate over each batch
        for batch_i in range(text_local_indices.size(0)):
            mean_attention_score = attention_scores[batch_i].mean().item()
            for token_i in range(text_local_indices.size(1)):
                # Compute dynamic threshold for each token
                aspect_relevance_score = aspect_relevance[batch_i, token_i].item() if aspect_relevance[batch_i, token_i].numel() == 1 else aspect_relevance[batch_i, token_i].squeeze().item()
                tau = self.alpha * mean_attention_score + self.gamma * aspect_relevance_score

                # Get the attention score for the current token
                attention_score = attention_scores[batch_i, token_i].item()

                # Apply masking based on the dynamic threshold
                if attention_score < tau:
                    masked_text_raw_indices[batch_i][token_i] = np.zeros(self.hidden, dtype=np.float32)

        return torch.tensor(masked_text_raw_indices).to(self.opt.device)


    def feature_dynamic_weighted(
        self, text_local_indices, aspect_indices, distances_input=None
    ):
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        if distances_input is not None:
            distances_input = distances_input.cpu().numpy()
        masked_text_raw_indices = np.ones(
            (text_local_indices.size(0), self.opt.max_seq_len, self.opt.bert_dim),
            dtype=np.float32,
        )  # batch x seq x dim
        mask_len = self.opt.SRD
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            if distances_input is None:
                asp_len = np.count_nonzero(asps[asp_i]) - 2
                try:
                    asp_begin = np.argwhere(texts[text_i] == asps[asp_i][2])[0][0]
                    asp_avg_index = (asp_begin * 2 + asp_len) / 2  # central position
                except:
                    continue
                distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
                for i in range(1, np.count_nonzero(texts[text_i]) - 1):
                    srd = abs(i - asp_avg_index) + asp_len / 2
                    if srd > self.opt.SRD:
                        distances[i] = 1 - (srd - self.opt.SRD) / np.count_nonzero(
                            texts[text_i]
                        )
                    else:
                        distances[i] = 1
                for i in range(len(distances)):
                    masked_text_raw_indices[text_i][i] = (
                        masked_text_raw_indices[text_i][i] * distances[i]
                    )
            else:
                distances_i = distances_input[text_i]  # distances of batch i-th
                for i, dist in enumerate(distances_i):
                    if dist > mask_len:
                        distances_i[i] = 1 - (dist - mask_len) / np.count_nonzero(
                            texts[text_i]
                        )
                    else:
                        distances_i[i] = 1

                for i in range(len(distances_i)):
                    masked_text_raw_indices[text_i][i] = (
                        masked_text_raw_indices[text_i][i] * distances_i[i]
                    )

        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)

    def forward(self, inputs, output_attentions=False):
        text_bert_indices = inputs[0]
        bert_segments_ids = inputs[1]
        text_local_indices = inputs[2]  # Raw text without adding aspect term
        aspect_indices = inputs[3]  # Raw text of aspect
        distances = inputs[4]
        # distances = None

        spc_out = self.bert_spc(text_bert_indices, bert_segments_ids)
        bert_spc_out = spc_out[0]
        spc_att = spc_out[-1][-1]
        # bert_spc_out = self.bert_g_sa(bert_spc_out)
        # bert_spc_out = self.dropout(bert_spc_out)
        # bert_spc_out = self.bert_g_pct(bert_spc_out)
        # bert_spc_out = self.dropout(bert_spc_out)

        bert_local_out = self.bert_local(text_local_indices)[0]
        # bert_local_out = self.bert_local_sa(bert_local_out)
        # bert_local_out = self.dropout(bert_local_out)
        # bert_local_out = self.bert_local_pct(bert_local_out)
        # bert_local_out = self.dropout(bert_local_out)

        if self.opt.local_context_focus == "cdm":
            masked_local_text_vec = self.feature_dynamic_mask(
                text_local_indices, aspect_indices, distances
            ) + 1e-6 * self.mask_scalar
            bert_local_out = torch.mul(bert_local_out, masked_local_text_vec)

        elif self.opt.local_context_focus == "cdw":
            weighted_text_local_features = self.feature_dynamic_weighted(
                text_local_indices, aspect_indices, distances
            )
            bert_local_out = torch.mul(bert_local_out, weighted_text_local_features)

        # bert_local_out = self.bert_local_sa(bert_local_out)
        out_cat = torch.cat((bert_local_out, bert_spc_out), dim=-1)
        mean_pool = self.mean_pooling_double(out_cat)
        self_attention_out, local_att = self.bert_sa(mean_pool)
        pooled_out = self.bert_pooler(self_attention_out)
        dense_out = self.dense(pooled_out)
        if output_attentions:
            return (dense_out, spc_att, local_att)
        return dense_out
