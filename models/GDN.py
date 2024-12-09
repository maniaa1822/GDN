import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
import math

from .graph_layer import GraphLayer as GraphLayer

def get_batch_edge_index_og(org_edge_index, batch_num, node_num):
    """Create batched edge indices."""
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    """Create batched edge indices more efficiently using vectorization."""
    edge_index = org_edge_index.clone().detach()
    
    # Create offset tensor [0, node_num, 2*node_num, ...]
    offset = torch.arange(batch_num, device=edge_index.device) * node_num
    
    # Add offset to edge_index for each batch
    # edge_index shape: [2, E], offset shape: [B]
    # Result shape: [2, E, B]
    batch_edge_index = edge_index.unsqueeze(-1) + offset.unsqueeze(0).unsqueeze(0)
    
    # Reshape to final form [2, E*B]
    batch_edge_index = batch_edge_index.reshape(2, -1)
    
    return batch_edge_index.long()

def compute_temporal_correlation(x, window_size):
    """
    Compute temporal correlation between sensors using sliding window.
    Args:
        x: Input tensor [batch_size, num_nodes, window_size]
        window_size: Size of sliding window
    Returns:
        Correlation matrix [num_nodes, num_nodes]
    """
    batch_size, num_nodes, _ = x.shape
    x_reshaped = x.transpose(0, 1).reshape(num_nodes, -1)
    
    # Center the data
    x_centered = x_reshaped - x_reshaped.mean(dim=1, keepdim=True)
    
    # Compute correlation matrix
    cov = torch.mm(x_centered, x_centered.t())
    std = torch.sqrt(torch.diagonal(cov) + 1e-8)
    corr = cov / (torch.outer(std, std) + 1e-8)
    
    return corr

class OutLayer(nn.Module):
    """Output layer implementing MLP."""
    def __init__(self, in_num, node_num, layer_num, inter_num=512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            if i == layer_num-1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out

class GNNLayer(nn.Module):
    """Graph neural network layer wrapper."""
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index

        out = self.bn(out)
        return self.relu(out)

class GDN(nn.Module):
    """Graph Deviation Network main class."""
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20):
        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets
        edge_index = edge_index_sets[0]

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1) 
            for i in range(edge_set_num)
        ])

        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num=out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index):
        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            
            batch_edge_index = self.cache_edge_index_sets[i]
            
            all_embeddings = self.embedding(torch.arange(node_num, device=device))

            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
            cos_ji_mat = cos_ji_mat / (normed_mat + 1e-8)

            temporal_corr = compute_temporal_correlation(
                x.view(batch_num, node_num, -1), 
                all_feature
            )
            use_temporal_corr = True  # Set this flag to use temporal correlation

            if use_temporal_corr:
                alpha = 0.7  # Hyperparameter to balance similarities
                combined_sim = alpha * cos_ji_mat + (1 - alpha) * temporal_corr
            else:
                combined_sim = cos_ji_mat

            topk_indices_ji = torch.topk(combined_sim, self.topk, dim=-1)[1]

            self.learned_graph = topk_indices_ji

            gated_i = torch.arange(0, node_num, device=device).view(-1,1).repeat(1, self.topk).flatten().unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)
            
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        indexes = torch.arange(0, node_num, device=device)
        out = torch.mul(x, self.embedding(indexes))
        
        out = out.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)

        return out
