import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F

from .graph_layer import GraphLayer


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out



class GNNLayer(nn.Module):
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
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        device = get_device()

        edge_index = edge_index_sets[0]


        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)


        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])


        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)

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
            
            all_embeddings = self.embedding(torch.arange(node_num).to(device))

            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
            cos_ji_mat = cos_ji_mat / normed_mat

            dim = weights.shape[-1]
            topk_num = self.topk

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            self.learned_graph = topk_indices_ji

            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings)

            
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)


        indexes = torch.arange(0,node_num).to(device)
        out = torch.mul(x, self.embedding(indexes))
        
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)
   

        return out
class TemporalAttention(nn.Module):
    """Temporal attention module to capture time-based relationships."""
    def __init__(self, in_channels, window_size):
        super().__init__()
        self.window_size = window_size
        
        # Temporal query/key/value projections
        self.query_proj = nn.Linear(in_channels, in_channels)
        self.key_proj = nn.Linear(in_channels, in_channels)
        self.value_proj = nn.Linear(in_channels, in_channels)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.sqrt(torch.FloatTensor([in_channels])))
        
    def forward(self, x):
        # x shape: [batch_size, num_nodes, window_size, channels]
        batch_size, num_nodes, window_size, channels = x.size()
        
        # Project to query/key/value
        q = self.query_proj(x)  # [batch, nodes, window, channels]
        k = self.key_proj(x)    # [batch, nodes, window, channels]
        v = self.value_proj(x)  # [batch, nodes, window, channels]
        
        # Compute temporal attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.temperature + 1e-6)
        attention = F.softmax(scores, dim=-1)  # [batch, nodes, window, window]
        
        # Apply attention to values
        out = torch.matmul(attention, v)  # [batch, nodes, window, channels]
        
        return out, attention

class EnhancedGDN(nn.Module):
    """GDN with both spatial and temporal attention."""
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, 
                 input_dim=10, out_layer_num=1, topk=20, window_size=10):
        super().__init__()
        
        self.edge_index_sets = edge_index_sets
        self.node_num = node_num
        self.dim = dim
        self.window_size = window_size
        
        # Node embeddings (as before)
        self.embedding = nn.Embedding(node_num, dim)
        
        # Spatial attention (GNN layers as before)
        self.spatial_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+dim, heads=1)
            for _ in range(len(edge_index_sets))
        ])
        
        # Add temporal attention
        self.temporal_attention = TemporalAttention(dim, window_size)
        
        # Fusion layer to combine spatial and temporal features
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        # Output layer
        self.out_layer = OutLayer(dim, node_num, out_layer_num, 
                                inter_num=out_layer_inter_dim)
        
        # Initialize other parameters as before
        self.init_params()
    
    def forward(self, data, edge_index):
        batch_size, node_num, seq_len = data.shape
        
        # Reshape input for temporal attention
        x_temporal = data.view(batch_size, node_num, -1, self.dim)
        temporal_out, temporal_attention = self.temporal_attention(x_temporal)
        
        # Spatial attention (existing GNN logic)
        x_spatial = data.view(-1, seq_len).contiguous()
        spatial_features = []
        
        for i, edge_index in enumerate(self.edge_index_sets):
            if isinstance(edge_index, torch.Tensor):
                edge_index = get_batch_edge_index(edge_index, batch_size, node_num)
            
            out = self.spatial_layers[i](x_spatial, edge_index,
                                       embedding=self.embedding.weight)
            spatial_features.append(out)
        
        spatial_out = torch.cat(spatial_features, dim=1)
        spatial_out = spatial_out.view(batch_size, node_num, -1)
        
        # Combine temporal and spatial features
        temporal_out = temporal_out.view(batch_size, node_num, -1)
        combined = torch.cat([spatial_out, temporal_out], dim=-1)
        fused = self.fusion(combined)
        
        # Final prediction
        out = self.out_layer(fused)
        return out.view(batch_size * node_num)

    def init_params(self):
        """Initialize network parameters."""
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))