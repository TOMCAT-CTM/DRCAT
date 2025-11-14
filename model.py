# model.py
# -*- coding: utf-8 -*-
"""
模型架构定义 (DRCAT)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChemistryModule(nn.Module):
    """将每个压力层的化学特征嵌入到高维空间"""
    def __init__(self, n_features_in, emb_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features_in, emb_size * 2),
            nn.LayerNorm(emb_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * 2, emb_size)
        )
    def forward(self, x):
        return self.net(x)

class GraphConstructor(nn.Module):
    """动态构建压力层之间的邻接矩阵（关系图）"""
    def __init__(self, num_nodes, k, dim, alpha=3.0):
        super().__init__()
        self.num_nodes, self.k, self.alpha = num_nodes, k, alpha
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, dim))
    def forward(self):
        A = F.relu(torch.tanh(self.alpha * (self.node_embeddings @ self.node_embeddings.T)))
        if self.k is not None and self.k < self.num_nodes:
            top_k_vals, _ = torch.topk(A, self.k, dim=1)
            kth_vals = top_k_vals[:, -1].view(-1, 1)
            mask = (A >= kth_vals).float()
            A = A * mask
        return A

class SageLayer(nn.Module):
    """一个结合了图神经网络和自注意力的编码器层"""
    def __init__(self, d_model, n_head, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.gnn_prop = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, adj):
        x = self.norm1(x + self.dropout(self.gnn_prop(torch.einsum('bnc,nm->bmc', x, adj))))
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm2(x + self.dropout(attn_out))
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x

class DRCAT(nn.Module):
    """
    DRCAT model for predicting OH and estimate uncertainty
    """
    def __init__(self, n_chem_features, n_levels, emb_size, nhead, num_layers, dim_feedforward, dropout, gnn_dim, k):
        super().__init__()
        self.n_levels = n_levels
        self.chemistry_encoder = ChemistryModule(n_chem_features, emb_size)
        self.positional_encoder = nn.Embedding(n_levels, emb_size)
        self.graph_constructor = GraphConstructor(num_nodes=n_levels, k=k, dim=gnn_dim)
        self.encoder_layers = nn.ModuleList([SageLayer(emb_size, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.ssa_oh_head = nn.Linear(emb_size, 1)
        self.mls_oh_head = nn.Linear(emb_size + 1, 2)
        self.init_weights()
        
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, x_seq, mask=None, ssa_phys_pred=None):
        batch_size = x_seq.shape[0]
        if mask is not None:
            x_seq = x_seq * mask
        x_flat = x_seq.view(-1, x_seq.shape[-1])
        chem_embedding = self.chemistry_encoder(x_flat).view(batch_size, self.n_levels, -1)
        positions = torch.arange(self.n_levels, device=x_seq.device).expand(batch_size, -1)
        x = chem_embedding + self.positional_encoder(positions)
        adj = self.graph_constructor()
        x_encoded = x
        for layer in self.encoder_layers:
            x_encoded = layer(x_encoded, adj)
        if ssa_phys_pred is None:
            return self.ssa_oh_head(x_encoded).squeeze(-1)
        else:
            head_input = torch.cat([x_encoded, ssa_phys_pred], dim=-1)
            mls_out = self.mls_oh_head(head_input)
            return mls_out[..., 0], mls_out[..., 1]