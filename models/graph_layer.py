import torch
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
import math
class GraphLayer_og(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True,inter_dim=-1, **kwargs):
        # Initialize with 'add' aggregation
        super().__init__(aggr='add', node_dim=0, 
                         #**kwargs
                         )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Linear transformation for input features
        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention parameters for features and embeddings
        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        # Optional bias parameter
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        
        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias)



    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        """Forward pass of the layer."""
        
        # Handle single/tuple input
        if isinstance(x, torch.Tensor):
            x = self.lin(x)
            x = (x, x)
        else:
            # Apply linear transformation
            x = (self.lin(x[0]), self.lin(x[1]))

        # Process edge indices
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))

        # Propagate messages
        out = self.propagate(edge_index, 
                           x=x,
                           embedding=embedding,
                           edges=edge_index,
                           return_attention_weights=return_attention_weights,
                           #size=None
                           )

        # Apply concatenation or mean
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # Add bias if present
        if self.bias is not None:
            out = out + self.bias

        # Handle attention weight return if requested
        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i,
                embedding,
                edges,
                return_attention_weights):
        """Constructs messages for each edge."""
        
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

    
        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

        
        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)
        
        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)
        
        # Convert edge_index_i to tensor if it isn't already
        edge_index_i = torch.tensor(edge_index_i) if isinstance(edge_index_i, int) else edge_index_i
        size_i = torch.tensor(size_i) if isinstance(size_i, (int, list)) else size_i
        alpha = alpha.view(-1, self.heads, 1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
        
class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, **kwargs):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.eps = 1e-6  # Small epsilon for numerical stability
        
        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))
        
        # Initialize temperature with a reasonable starting value
        init_temp = math.sqrt(self.out_channels)
        self.log_temperature = Parameter(torch.log(torch.Tensor([init_temp])))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def forward(self, x, edge_index, embedding=None, node_num=None, return_attention_weights=False):
        if isinstance(x, torch.Tensor):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        # Add assertions for debugging
        if embedding is not None:
            assert embedding.size(-1) == self.out_channels, f"Embedding dim {embedding.size(-1)} != out_channels {self.out_channels}"

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x[1].size(self.node_dim))

        out = self.propagate(edge_index, x=x, embedding=embedding, 
                           edges=edge_index, size=None,
                           return_attention_weights=return_attention_weights)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha = self.__alpha__
            self.__alpha__ = None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, index, ptr, size_t,
                embedding, edges, return_attention_weights):
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None:
            embedding_i = embedding[index]
            embedding_j = embedding[edges[0]]
            
            # Normalize embeddings for stability
            embedding_i = F.normalize(embedding_i, p=2, dim=-1)
            embedding_j = F.normalize(embedding_j, p=2, dim=-1)
            
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

            cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
            cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

            # Get clamped temperature value
            temperature = torch.exp(self.log_temperature).clamp(min=0.1, max=100.0)
            
            # Compute attention scores with scaled dot product
            key_i = F.normalize(key_i, p=2, dim=-1)
            key_j = F.normalize(key_j, p=2, dim=-1)
            
            att_i = (key_i * cat_att_i).sum(-1)
            att_j = (key_j * cat_att_j).sum(-1)
            
            alpha = (att_i + att_j) / (temperature + self.eps)

        alpha = alpha.view(-1, self.heads, 1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Apply softmax with numerical stability
        alpha = softmax(alpha, index)
        
        if torch.isnan(alpha).any():
            print("NaN in attention weights!")
            alpha = torch.ones_like(alpha) / alpha.size(0)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.view(-1, self.heads, 1)

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.att_em_i)
        zeros(self.att_em_j)
        
        # Initialize temperature to reasonable value
        with torch.no_grad():
            self.log_temperature.fill_(math.log(math.sqrt(self.out_channels)))