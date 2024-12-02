import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
import math

from .graph_layer import GraphLayer

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    """Create batched edge indices."""
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()

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

class GDN_OG(nn.Module):
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
            cos_ji_mat = cos_ji_mat / normed_mat

            
            dim = weights.shape[-1]
            topk_num = self.topk

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            self.learned_graph = topk_indices_ji

            gated_i = torch.arange(0, node_num, device=device).view(-1,1).repeat(1, topk_num).flatten().unsqueeze(0)
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


class GDN(nn.Module):
    """Graph Deviation Network with representation-level fusion of supervised and unsupervised learning."""
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, 
                 input_dim=10, out_layer_num=1, topk=20):
        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets
        self.node_num = node_num
        self.dim = dim
        self.topk = topk
        
        # Shared embedding layer for node representations
        self.embedding = nn.Embedding(node_num, dim)
        self.bn_outlayer_in = nn.BatchNorm1d(dim)

        # Number of edge sets (typically 1 for fully connected graph)
        edge_set_num = len(edge_index_sets)
        
        # Forecasting pathway (unsupervised)
        self.forecast_gnn = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+dim, heads=1)
            for i in range(edge_set_num)
        ])
        
        # Detection pathway (supervised)
        self.detection_gnn = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+dim, heads=1)
            for i in range(edge_set_num)
        ])

        # Output layers for each pathway
        self.forecast_layer = OutLayer(
            dim*edge_set_num,
            node_num,
            out_layer_num,
            inter_num=out_layer_inter_dim
        )
        
        # Classifier for detection pathway
        self.classifier = nn.Sequential(
            nn.Linear(dim * edge_set_num, dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # Representation-level fusion module
        fusion_input_dim = dim * edge_set_num * 2  # Concatenated representations from both pathways
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # Cache for computed edge indices
        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None
        
        self.dp = nn.Dropout(0.2)
        
        self.init_params()

    def init_params(self):
        """Initialize network parameters."""
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, edge_index):
        """
        Forward pass incorporating both pathways and representation fusion.
        
        Args:
            data: Input time series data [batch_size, num_nodes, features]
            edge_index: Graph edge indices
            
        Returns:
            During training:
                final_output: Fused anomaly predictions
                forecast_out: Forecasting pathway predictions
                detection_out: Detection pathway predictions
            During inference:
                final_output: Fused anomaly predictions
        """
        x = data.clone().detach()
        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        # Process through both pathways to get representations
        forecast_features = []
        detection_features = []

        for i, edge_index in enumerate(self.edge_index_sets):
            # Cache and prepare edge indices
            if self.cache_edge_index_sets[i] is None:
                self.cache_edge_index_sets[i] = get_batch_edge_index(
                    edge_index, batch_num, node_num).to(device)
                
            batch_edge_index = self.cache_edge_index_sets[i]
            
            # Get node embeddings
            all_embeddings = self.embedding(torch.arange(node_num, device=device))
            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            # Learn graph structure
            weights = weights_arr.view(node_num, -1)
            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), 
                                    weights.norm(dim=-1).view(1,-1))
            cos_ji_mat = cos_ji_mat / normed_mat

            # Get top-k connections
            topk_indices_ji = torch.topk(cos_ji_mat, self.topk, dim=-1)[1]
            self.learned_graph = topk_indices_ji

            # Create edge indices for attention
            gated_i = torch.arange(0, node_num, device=device).view(-1,1)\
                          .repeat(1, self.topk).flatten().unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
            
            batch_gated_edge_index = get_batch_edge_index(
                gated_edge_index, batch_num, node_num).to(device)

            # Process through both pathways
            forecast_out = self.forecast_gnn[i](
                x, batch_gated_edge_index,
                node_num=node_num*batch_num,
                embedding=all_embeddings
            )
            detection_out = self.detection_gnn[i](
                x, batch_gated_edge_index,
                node_num=node_num*batch_num,
                embedding=all_embeddings
            )

            forecast_features.append(forecast_out)
            detection_features.append(detection_out)

        # Combine features from each pathway
        forecast_rep = torch.cat(forecast_features, dim=1)
        detection_rep = torch.cat(detection_features, dim=1)
        
        forecast_rep = forecast_rep.view(batch_num, node_num, -1)
        detection_rep = detection_rep.view(batch_num, node_num, -1)

        # Generate pathway-specific outputs
        forecast_out = self.process_forecast(forecast_rep)
        detection_out = self.classifier(detection_rep.mean(dim=1))

        # Fuse representations for final prediction
        forecast_rep_pooled = forecast_rep.mean(dim=1)
        detection_rep_pooled = detection_rep.mean(dim=1)
        
        combined_rep = torch.cat([forecast_rep_pooled, detection_rep_pooled], dim=1)
        final_output = self.fusion(combined_rep)

        if self.training:
            return final_output, forecast_out, detection_out
        else:
            return final_output

    def process_forecast(self, features):
        """Process features for forecasting pathway."""
        indexes = torch.arange(0, self.node_num, device=features.device)
        forecast_out = torch.mul(features, self.embedding(indexes))
        
        forecast_out = forecast_out.permute(0, 2, 1)
        forecast_out = F.relu(self.bn_outlayer_in(forecast_out))
        forecast_out = forecast_out.permute(0, 2, 1)
        
        forecast_out = self.dp(forecast_out)
        forecast_out = self.forecast_layer(forecast_out)
        return forecast_out.view(-1, self.node_num)
    @staticmethod
    def focal_loss(pred, target, gamma=2.0, alpha=0.25):
        # Clip predictions to prevent numerical instability
        pred = torch.clamp(pred, 1e-7, 1-1e-7)
        
        # Calculate BCE loss
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Calculate focal term
        pt = torch.exp(-bce)
        focal_term = (1-pt)**gamma
        
        # Apply alpha balancing
        if alpha is not None:
            alpha_term = target * alpha + (1-target) * (1-alpha)
            focal_term = alpha_term * focal_term
            
        return (focal_term * bce).mean()

    def loss_function(self, final_output, forecast_out, detection_out, 
                 true_values, labels, alpha=0.5, gamma=2.0):
        # Unsupervised forecasting loss
        forecast_loss = F.mse_loss(forecast_out, true_values)
        
        # Add small epsilon to avoid numerical instability
        detection_out = torch.clamp(detection_out, 1e-7, 1-1e-7)
        final_output = torch.clamp(final_output, 1e-7, 1-1e-7)
        
        # Supervised detection loss with focal loss
        detection_loss = GDN.focal_loss(detection_out.squeeze(), labels, 
                                  gamma=gamma) * 10000
    
        # Final prediction loss with focal loss
        final_loss = GDN.focal_loss(final_output.squeeze(), labels, gamma=gamma)
        
        # Combine losses with weighting
        total_loss = (1 - alpha) * forecast_loss + \
                    alpha * (detection_loss + final_loss)
        
        return total_loss, forecast_loss, detection_loss