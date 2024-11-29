import torch
import numpy as np
from torch_geometric.data import Data

# Import your GraphLayer implementation
from models.graph_layer import GraphLayer

def test_graph_layer():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test parameters
    in_channels = 16
    out_channels = 32
    num_nodes = 10
    heads = 2
    
    # Create random input features
    x = torch.randn(num_nodes, in_channels)
    
    # Create a simple edge index (connecting each node to next node)
    edge_index = torch.tensor([[i for i in range(num_nodes-1)],
                             [i+1 for i in range(num_nodes-1)]], dtype=torch.long)
    
    # Create random node embeddings
    embedding = torch.randn(num_nodes, out_channels)
    
    # Initialize the layer
    layer = GraphLayer(in_channels, out_channels, heads=heads)
    
    # Test 1: Basic forward pass without embeddings
    print("\nTest 1: Without embeddings")
    try:
        out = layer(x, edge_index)
        print(f"Output shape: {out.shape}")
        print("Expected shape:", (num_nodes, out_channels * heads))
        print("Test 1 passed!")
    except Exception as e:
        print("Test 1 failed:", str(e))
    
    # Test 2: Forward pass with embeddings
    print("\nTest 2: With embeddings")
    try:
        out = layer(x, edge_index, embedding=embedding)
        print(f"Output shape: {out.shape}")
        print("Expected shape:", (num_nodes, out_channels * heads))
        print("Test 2 passed!")
    except Exception as e:
        print("Test 2 failed:", str(e))
    
    # Test 3: Test attention weights
    print("\nTest 3: Testing attention weights")
    try:
        out, (edge_index, attention_weights) = layer(x, edge_index, 
                                                   embedding=embedding,
                                                   return_attention_weights=True)
        print(f"Attention weights shape: {attention_weights.shape}")
        print(f"Edge index shape: {edge_index.shape}")
        print("Test 3 passed!")
    except Exception as e:
        print("Test 3 failed:", str(e))
    
    # Test 4: Test with different batch sizes
    print("\nTest 4: Testing with batch")
    try:
        batch_size = 3
        batch_x = x.unsqueeze(0).repeat(batch_size, 1, 1)
        batch_edge_index = edge_index.clone()
        batch_embedding = embedding.unsqueeze(0).repeat(batch_size, 1, 1)
        
        out = layer(batch_x, batch_edge_index, embedding=batch_embedding)
        print(f"Batch output shape: {out.shape}")
        print(f"Expected batch shape: {(batch_size * num_nodes, out_channels * heads)}")
        print("Test 4 passed!")
    except Exception as e:
        print("Test 4 failed:", str(e))

if __name__ == "__main__":
    test_graph_layer()
