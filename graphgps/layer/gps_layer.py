import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch, k_hop_subgraph, degree, sort_edge_index, to_undirected, subgraph, to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE
from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from mamba_ssm import Mamba
from typing import List
import random

import numpy as np
import torch
from torch import Tensor


def permute_nodes_within_identity(identities):
    unique_identities, inverse_indices = torch.unique(identities, return_inverse=True)
    node_indices = torch.arange(len(identities), device=identities.device)
    
    masks = identities.unsqueeze(0) == unique_identities.unsqueeze(1)
    
    # Generate random indices within each identity group using torch.randint
    permuted_indices = torch.cat([
        node_indices[mask][torch.randperm(mask.sum(), device=identities.device)] for mask in masks
    ])
    return permuted_indices

def sort_rand_gpu(pop_size, num_samples, neighbours):
    # Randomly generate indices and select num_samples in neighbours
    idx_select = torch.argsort(torch.rand(pop_size, device=neighbours.device))[:num_samples]
    neighbours = neighbours[idx_select]
    return neighbours

def augment_seq(edge_index, batch, num_k = -1):
    unique_batches = torch.unique(batch)
    # Initialize list to store permuted indices
    permuted_indices = []
    mask = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()
        for k in indices_in_batch:
            neighbours = edge_index[1][edge_index[0]==k]
            if num_k > 0 and len(neighbours) > num_k:
                neighbours = sort_rand_gpu(len(neighbours), num_k, neighbours)
            permuted_indices.append(neighbours)
            mask.append(torch.zeros(neighbours.shape, dtype=torch.bool, device=batch.device))
            permuted_indices.append(torch.tensor([k], device=batch.device))
            mask.append(torch.tensor([1], dtype=torch.bool, device=batch.device))
    permuted_indices = torch.cat(permuted_indices)
    mask = torch.cat(mask)
    return permuted_indices.to(device=batch.device), mask.to(device=batch.device)

def lexsort(
    keys: List[Tensor],
    dim: int = -1,
    descending: bool = False,
) -> Tensor:
    r"""Performs an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, returns an array of integer indices that
    describe their sort order.
    The last key in the sequence is used for the primary sort order, the
    second-to-last key for the secondary sort order, and so on.

    Args:
        keys ([torch.Tensor]): The :math:`k` different columns to be sorted.
            The last key is the primary sort key.
        dim (int, optional): The dimension to sort along. (default: :obj:`-1`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)
    """
    assert len(keys) >= 1

    out = keys[0].argsort(dim=dim, descending=descending, stable=True)
    for k in keys[1:]:
        index = k.gather(dim, out)
        index = index.argsort(dim=dim, descending=descending, stable=True)
        out = out.gather(dim, index)
    return out

def dynamic_sampling_ratio(degree, max_degree, min_bound=(0.3, 0.1), max_bound=(0.9, 0.4)):
    normalized_degree = degree / max_degree

    low_bound = min_bound[1] + (max_bound[1] - min_bound[1]) * normalized_degree
    high_bound = min_bound[0] + (max_bound[0] - min_bound[0]) * normalized_degree
    
    low_bound = max(0.1, min(low_bound, 0.9))
    high_bound = max(low_bound, min(high_bound, 0.9))
    
    sampling_ratio = torch.rand(1).item() * (high_bound - low_bound) + low_bound
    
    return sampling_ratio

def sample_random_subgraphs_from_k_hop(node_idx, degree, max_degree, num_hops, edge_index, num_samples=10, num_nodes=None):
    sampled_subgraphs = []

    subgraph_node_indices, subgraph_edge_indices, _, edge_mask = k_hop_subgraph(
        node_idx, num_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes
    )

    for _ in range(num_samples):
        sampling_ratio = dynamic_sampling_ratio(degree, max_degree)

        # Randomly sample a subset of nodes from the k-hop subgraph
        num_sub_nodes = int(len(subgraph_node_indices) * sampling_ratio)
        sampled_nodes = subgraph_node_indices[torch.randperm(subgraph_node_indices.size(0))[:num_sub_nodes]]
        
        # Get the corresponding edge indices for the sampled subgraph
        sampled_subgraph_edge_index, sampled_edge_mask = subgraph(sampled_nodes, subgraph_edge_indices, relabel_nodes=True)
        
        sampled_subgraphs.append((sampled_nodes, sampled_subgraph_edge_index, edge_mask, sampled_edge_mask))

    return sampled_subgraphs

def calculate_max_k(degrees, min_k=1, max_k=5, scale_factor=1.0):
    normalized_degrees = (degrees - degrees.min()) / (degrees.max() - degrees.min())
    
    k_values = max_k - (normalized_degrees * (max_k - min_k) * scale_factor)
    
    k_values = torch.clamp(k_values, min=min_k, max=max_k)
    
    k_values = torch.round(k_values).long()
    
    return k_values

def encode_subgraphs(subgraphs, x, edge_attr, model):
    subgraph_embeddings = []
    for subgraph in subgraphs:
        sampled_nodes = subgraph[0]
        sampled_subgraph_edge_index = subgraph[1]
        edge_mask = subgraph[2]
        sampled_edge_mask = subgraph[3]
        sampled_edge_attr = edge_attr[edge_mask][sampled_edge_mask]

        subgraph_data = Data(x=x[sampled_nodes], edge_index=sampled_subgraph_edge_index, edge_attr=sampled_edge_attr)

        subgraph_embedding = model(subgraph_data)
        
        subgraph_embeddings.append(subgraph_embedding)

    return torch.mean(torch.stack(subgraph_embeddings), dim=0)


def permute_within_batch(batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)
    
    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()
        
        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]
        
        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)
    
    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices

# Heuristics
def _heuristic_degree(batch):
    h = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
    return [h]
def _heuristic_degree_bidirectional(batch):
    h = _heuristic_degree(batch)[0]
    return [h, -h]
def _heuristic_eigencentrality(batch):
    return [batch.EigCentrality]
def _heuristic_eigencentrality_bidirectional(batch):
    h = _heuristic_eigencentrality(batch)[0]
    return [h, -h]
def _heuristic_RWSE(batch):
    return [-torch.sum(batch.pestat_RWSE, dim=1)]
def _heuristic_RWSE_bidirectional(batch):
    h = _heuristic_RWSE(batch)[0]
    return [h, -h]
def _heuristic_global_pe_1(batch):
    return [batch.EigVecs[:, 1]]
def _heuristic_global_pe_1_bidirectional(batch):
    h = _heuristic_global_pe_1(batch)[0]
    return [h, -h]
def _heuristic_global_pe_2(batch):
    return [batch.EigVecs[:, 1], batch.EigVecs[:, 2]]
def _heuristic_global_pe_2_bidirectional(batch):
    hs = _heuristic_global_pe_2(batch)
    return hs + [-h for h in hs]
def _heuristic_global_pe_m1(batch):
    return [batch.EigVecs[:, -1]]
def _heuristic_global_pe_m1_bidirectional(batch):
    hs = _heuristic_global_pe_m1(batch)
    return hs + [-h for h in hs]

heuristic_fns = {
    'degree': (_heuristic_degree, 1),
    'degree_bidirectional': (_heuristic_degree_bidirectional, 2),
    'eigencentrality': (_heuristic_eigencentrality, 1),
    'eigencentrality_bidirectional': (_heuristic_eigencentrality_bidirectional, 2),
    'RWSE': (_heuristic_RWSE, 1),
    'RWSE_bidirectional': (_heuristic_RWSE_bidirectional, 2),
    'global_pe_1': (_heuristic_global_pe_1, 1),
    'global_pe_2': (_heuristic_global_pe_2, 2),
    'global_pe_m1': (_heuristic_global_pe_m1, 1),
    'global_pe_1_bidirectional': (_heuristic_global_pe_1_bidirectional, 2),
    'global_pe_2_bidirectional': (_heuristic_global_pe_2_bidirectional, 4),
    'global_pe_m1_bidirectional': (_heuristic_global_pe_m1_bidirectional, 2),
}

class MeanModel(nn.Module):
    def __init__(self, d_model, expand):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, expand*d_model),
            nn.ReLU(),
            nn.Linear(expand*d_model, d_model)
        )

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1)
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        x = self.mlp(x)
        x = x.reshape(x_shape)
        x = ((x.sum(dim=-2, keepdim=True) / mask.sum(dim=-2, keepdim=True))*mask)
        x = x.expand(x.shape)
        return x

class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads,
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, mamba_permute_iterations=0, mamba_noise=0.0, mamba_buckets_num=0, mamba_heuristics=['degree']):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.NUM_BUCKETS = 3
        self.mamba_permute_iterations = mamba_permute_iterations
        self.mamba_noise = mamba_noise
        self.mamba_buckets_num = mamba_buckets_num
        self.mamba_heuristics = mamba_heuristics

        # Local message-passing model.
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   nn.ReLU(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=16, # dim_h,
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'Transformer':
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        elif 'MeanL65' in global_model_type:
            self.self_attn = MeanModel(d_model=dim_h, # Model dimension d_model
                        expand=2,    # Block expansion factor
                    )
        elif 'Subgraph_Mamba_L65' in self.global_model_type:
            self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,    # Local convolution width
                    expand=1,    # Block expansion factor
                )
        elif 'RandomWalk_Mamba_L65' in self.global_model_type:
            # Initialize the Mamba model for self.self_attn
            self.self_attn = Mamba(d_model=dim_h,  # Input dimension
                                d_state=16,     # State dimension (adjust as needed)
                                d_conv=4,       # Convolution dimension (adjust as needed)
                                expand=1)       # Expansion factor (adjust as needed)

            # Initialize the MLP with one hidden layer
            expand_factor = 2  # Adjust the expansion factor as needed
            self.mlp = nn.Sequential(
                nn.Linear(dim_h, dim_h * expand_factor),  # Input to hidden layer
                nn.ReLU(),                                # Activation function
                nn.Linear(dim_h * expand_factor, dim_h)   # Hidden layer to output
            )
        elif 'MambaL65' in global_model_type:
            num_models = max(1, sum((heuristic_fns[h][1] for h in mamba_heuristics)))
            self.self_attn = torch.nn.ModuleList()
            for i in range(num_models):
                self.self_attn.append(Mamba(d_model=dim_h, # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,    # Local convolution width
                    expand=1,    # Block expansion factor
                ))
        elif 'Mamba' in global_model_type:
            if global_model_type.split('_')[-1] == '2':
                self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                        d_state=8,  # SSM state expansion factor
                        d_conv=4,    # Local convolution width
                        expand=2,    # Block expansion factor
                    )
            elif global_model_type.split('_')[-1] == '4':
                self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                        d_state=4,  # SSM state expansion factor
                        d_conv=4,    # Local convolution width
                        expand=4,    # Block expansion factor
                    )
            elif global_model_type.split('_')[-1] == 'Multi':
                self.self_attn = []
                for i in range(4):
                    self.self_attn.append(Mamba(d_model=dim_h, # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,    # Local convolution width
                    expand=1,    # Block expansion factor
                    ))
            elif global_model_type.split('_')[-1] == 'SmallConv':
                self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                        d_state=16,  # SSM state expansion factor
                        d_conv=2,    # Local convolution width
                        expand=1,    # Block expansion factor
                    )
            elif global_model_type.split('_')[-1] == 'SmallState':
                self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                        d_state=8,  # SSM state expansion factor
                        d_conv=4,    # Local convolution width
                        expand=1,    # Block expansion factor
                    )
            else:
                self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                        d_state=16,  # SSM state expansion factor
                        d_conv=4,    # Local convolution width
                        expand=1,    # Block expansion factor
                    )
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            # self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        if self.layer_norm:
            # self.norm2 = pygnn.norm.LayerNorm(dim_h)
            self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection
        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(Batch(batch=batch,
                                                   x=h,
                                                   edge_index=batch.edge_index,
                                                   edge_attr=batch.edge_attr,
                                                   pe_EquivStableLapPE=es_data))
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.equivstable_pe:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr,
                                               batch.pe_EquivStableLapPE)
                else:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            if self.global_model_type in ['Transformer', 'Performer', 'BigBird', 'Mamba', 'MeanL65']:
                h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)

            elif self.global_model_type == 'RandomWalk_Mamba_L65':
                num_walks = ... # TODO (evan) Define the number of walks per node (num_walk)
                walk_length = ... # TODO (evan) Define the length of each walk (walk_len)
                walks_embeddings = []
                for node_idx in range(batch.x.size(0)):  # Loop over each node in the batch
                    node_walks_embeddings = []
                    for _ in range(num_walks):
                        # Perform a random walk for the current node
                        walk_nodes = [node_idx]
                        current_node = node_idx
                        for _ in range(walk_length - 1):
                            neighbors = edge_index[1][edge_index[0] == current_node]
                            if len(neighbors) > 0:
                                current_node = neighbors[random.randint(0, len(neighbors) - 1)].item()
                                walk_nodes.append(current_node)
                            else:
                                break
                        walk_nodes.reverse()  # Reverse the list to start with the initial node
                        # Create embeddings using Mamba for the node-edge states (you need to define how to extract these states)
                        # For simplicity, assuming `mamba_model` is an instance of the Mamba model you want to use
                        node_edge_states = ... # TODO (evan) Extract or construct node-edge states from walk_nodes
                        walk_embedding = self.self_attn(node_edge_states)  # Assuming self.self_attn is a Mamba model here
                        node_walks_embeddings.append(walk_embedding)
                    # Concatenate all walk embeddings for the current node and append to the walks_embeddings list
                    walks_embeddings.append(torch.cat(node_walks_embeddings, dim=0))
                # Concatenate embeddings from all nodes to form a batch-wide embedding tensor
                batch_embedding = torch.cat(walks_embeddings, dim=0)
                # Process the batch_embedding through an MLP as specified
                mlp_output = self.mlp(batch_embedding)
                h_attn = mlp_output
            elif self.global_model_type == 'Subgraph_Mamba_L65':
                degrees = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                max_degree = torch.max(degrees)
                max_k_values = calculate_max_k(degrees)
                subgraph_encodings = []
                node_subgraph_embeddings = []
                for node_idx in range(batch.x.size(0)):  # Loop over each node in the batch
                    # Sample subgraphs for different k values
                    subgraph_encodings = []
                    max_k = max_k_values[node_idx]
                    degree = degrees[node_idx]
                    for k in range(1, max_k + 1):  # 'max_k' is the maximum hop you consider
                        subgraphs = sample_random_subgraphs_from_k_hop(node_idx, degree, max_degree, k, batch.edge_index, num_nodes=batch.x.size(0))
                        subgraph_encoding = encode_subgraphs(subgraphs, batch.x, batch.edge_attr, self.local_model)
                        subgraph_encodings.append(subgraph_encoding)
                    
                    # Merge the subgraph encodings for the current node into a single representation
                    # This could involve averaging, concatenating, or another aggregation method
                    node_representation = torch.cat(subgraph_encodings, dim=0)
                    node_subgraph_embeddings.append(node_representation)

                # Initialize lists to hold padded sequences and their masks
                padded_sequences = []
                sequence_masks = []

                # Maximum length will be determined dynamically
                max_sequence_length = 0
                for node_idx in range(batch.x.size(0)):
                    sequence = []
                    # Global context
                    for idx, encodings in enumerate(node_subgraph_embeddings):
                        if idx != node_idx:
                            sequence.append(encodings)
                    random.shuffle(sequence)  # Shuffle global context
                    # Local context and node's own features
                    sequence.append(node_subgraph_embeddings[node_idx])
                    sequence.append(batch.x[node_idx].unsqueeze(0))  # Ensure it's 2D
                    
                    # Convert list of tensors into a single tensor (sequence_tensor)
                    sequence_tensor = torch.cat(sequence, dim=0)
                    max_sequence_length = max(max_sequence_length, sequence_tensor.size(0))
                    
                    # Store the sequence tensor for now; padding will be done in the next step
                    padded_sequences.append(sequence_tensor)
                    # Create and store the mask indicating the real length of the sequence
                    sequence_masks.append(torch.ones(sequence_tensor.size(0), dtype=torch.bool))

                # Step 2: Pad sequences to the max length found
                for i in range(len(padded_sequences)):
                    padding_length = max_sequence_length - padded_sequences[i].size(0)
                    padded_sequences[i] = F.pad(padded_sequences[i], pad=(0, 0, 0, padding_length), mode='constant', value=0)
                    sequence_masks[i] = F.pad(sequence_masks[i], pad=(0, padding_length), mode='constant', value=False)

                # Convert lists to tensors
                padded_sequences_tensor = torch.stack(padded_sequences)
                sequence_masks_tensor = torch.stack(sequence_masks)

                h_attn = self.self_attn(padded_sequences_tensor)[sequence_masks_tensor]
            
            elif self.global_model_type == 'MambaL65':
                # NOTE(guillem): This should include all of the Mamba variants below, but in a much more concise way!
                # (and also our new code)

                # NOTE(guillem): naming conventions of graph-mamba's global_mode_type:
                #   permute = hybrid with 5 replaced by 1 for averaging during inference:
                #       basically they randomly permute the orders before applying any heuristic
                #   multi (doesn't seem to be used):
                #       different mamba layers running in parallel
                #   noise:
                #       adds noise to the heuristics
                #   bucket:
                #       colors each node randomly and runs mamba for the sequences of nodes in each color
                #   cluster:
                #       similar to bucket but instead of random colors, it permutes nodes within a cluster
                #   degree, eigen, RWSE:
                #       different heuristics that are used

                

                permute_iterations = self.mamba_permute_iterations
                if batch.split == 'train':
                    permute_iterations = 1
                noise = self.mamba_noise
                buckets_num = self.mamba_buckets_num

                heuristics = self.mamba_heuristics
                heuristic_values = sum((heuristic_fns[h][0](batch) for h in heuristics), [])
                heuristic_noise_maginitudes = [noise*torch.std(h) for h in heuristic_values]
                mamba_arr = []
                for _ in range(max(permute_iterations, 1)):
                    for j, heuristic_ in enumerate(heuristic_values):
                        if noise > 0.0:
                            heuristic_noise = heuristic_noise_maginitudes[j]*torch.randn_like(heuristic_).to(heuristic_.device)
                            heuristic = heuristic_ + heuristic_noise
                        else:
                            heuristic = heuristic_
                        
                        if buckets_num > 1:
                            indices_arr, emb_arr = [], []
                            bucket_assign = torch.randint_like(heuristic, 0, self.NUM_BUCKETS).to(heuristic.device)
                            for i in range(buckets_num):
                                ind_i = (bucket_assign==i).nonzero().squeeze()
                                h_ind_perm_sort = lexsort([heuristic[ind_i], batch.batch[ind_i]])
                                h_ind_perm_i = ind_i[h_ind_perm_sort]
                                h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                                h_dense = self.self_attn[j](h_dense)[mask]
                                indices_arr.append(h_ind_perm_i)
                                emb_arr.append(h_dense)
                            h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                            h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]

                        elif permute_iterations > 1:
                            h_ind_perm = permute_within_batch(batch.batch)
                            h_ind_perm_1 = lexsort([heuristic[h_ind_perm], batch.batch[h_ind_perm]])
                            h_ind_perm = h_ind_perm[h_ind_perm_1]
                            h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                            h_ind_perm_reverse = torch.argsort(h_ind_perm)
                            h_attn = self.self_attn[j](h_dense)[mask][h_ind_perm_reverse]

                        else:
                            h_ind_perm = lexsort([heuristic, batch.batch])
                            h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                            h_ind_perm_reverse = torch.argsort(h_ind_perm)
                            h_attn = self.self_attn[j](h_dense)[mask][h_ind_perm_reverse]

                        mamba_arr.append(h_attn)

                h_attn = sum(mamba_arr) / len(mamba_arr)
            
            elif self.global_model_type == 'MeanL65':
                h_attn = self.self_attn(h_dense, mask)[mask]    
            
            elif self.global_model_type == 'Mamba':
                h_attn = self.self_attn(h_dense)[mask]                

            elif self.global_model_type == 'Mamba_Permute':
                h_ind_perm = permute_within_batch(batch.batch)
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
            
            elif self.global_model_type == 'Mamba_Degree':
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                # indcies that sort by batch and then deg, by ascending order
                h_ind_perm = lexsort([deg, batch.batch])
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
            
            elif self.global_model_type == 'Mamba_Hybrid':
                if batch.split == 'train':
                    h_ind_perm = permute_within_batch(batch.batch)
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        h_ind_perm = permute_within_batch(batch.batch)
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif 'Mamba_Hybrid_Degree' == self.global_model_type:
                if batch.split == 'train':
                    h_ind_perm = permute_within_batch(batch.batch)
                    #h_ind_perm = permute_nodes_within_identity(batch.batch)
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                    h_ind_perm_1 = lexsort([deg[h_ind_perm], batch.batch[h_ind_perm]])
                    h_ind_perm = h_ind_perm[h_ind_perm_1]
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    if self.global_model_type.split('_')[-1] == 'Multi':
                        h_attn_list = []
                        for mod in self.self_attn:
                            mod = mod.to(h_dense.device)
                            h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                            h_attn_list.append(h_attn) 
                        h_attn = sum(h_attn_list) / len(h_attn_list)
                    else:
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        #h_ind_perm = permute_nodes_within_identity(batch.batch)
                        h_ind_perm = permute_within_batch(batch.batch)
                        deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                        h_ind_perm_1 = lexsort([deg[h_ind_perm], batch.batch[h_ind_perm]])
                        h_ind_perm = h_ind_perm[h_ind_perm_1]
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        if self.global_model_type.split('_')[-1] == 'Multi':
                            h_attn_list = []
                            for mod in self.self_attn:
                                mod = mod.to(h_dense.device)
                                h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
                                h_attn_list.append(h_attn) 
                            h_attn = sum(h_attn_list) / len(h_attn_list)
                        else:
                            h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        #h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5
            
            elif 'Mamba_Hybrid_Degree_Noise' == self.global_model_type:
                if batch.split == 'train':
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    #deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    # Potentially use torch.rand_like?
                    #deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    #deg_noise = torch.randn(deg.shape).to(deg.device)
                    deg_noise = torch.rand_like(deg).to(deg.device)
                    h_ind_perm = lexsort([deg+deg_noise, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    for i in range(5):
                        #deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                        #deg_noise = torch.randn(deg.shape).to(deg.device)
                        deg_noise = torch.rand_like(deg).to(deg.device)
                        h_ind_perm = lexsort([deg+deg_noise, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5
                        
            elif 'Mamba_Hybrid_Degree_Noise_Bucket' == self.global_model_type:
                if batch.split == 'train':
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    #deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    deg_noise = torch.rand_like(deg).to(deg.device)
                    #deg_noise = torch.randn(deg.shape).to(deg.device)
                    deg = deg + deg_noise
                    indices_arr, emb_arr = [],[]
                    bucket_assign = torch.randint_like(deg, 0, self.NUM_BUCKETS).to(deg.device)
                    for i in range(self.NUM_BUCKETS):
                        ind_i = (bucket_assign==i).nonzero().squeeze()
                        h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    deg_ = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    for i in range(5):
                        #deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                        deg_noise = torch.rand_like(deg_).to(deg_.device)
                        #deg_noise = torch.randn(deg.shape).to(deg.device)
                        deg = deg_ + deg_noise
                        indices_arr, emb_arr = [],[]
                        bucket_assign = torch.randint_like(deg, 0, self.NUM_BUCKETS).to(deg.device)
                        for i in range(self.NUM_BUCKETS):
                            ind_i = (bucket_assign==i).nonzero().squeeze()
                            h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif 'Mamba_Hybrid_Noise' == self.global_model_type:
                if batch.split == 'train':
                    deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(batch.batch.device)
                    indices_arr, emb_arr = [],[]
                    bucket_assign = torch.randint_like(deg_noise, 0, self.NUM_BUCKETS).to(deg_noise.device)
                    for i in range(self.NUM_BUCKETS):
                        ind_i = (bucket_assign==i).nonzero().squeeze()
                        h_ind_perm_sort = lexsort([deg_noise[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    deg = batch.batch.to(torch.float)
                    for i in range(5):
                        deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(batch.batch.device)
                        indices_arr, emb_arr = [],[]
                        bucket_assign = torch.randint_like(deg_noise, 0, self.NUM_BUCKETS).to(deg_noise.device)
                        for i in range(self.NUM_BUCKETS):
                            ind_i = (bucket_assign==i).nonzero().squeeze()
                            h_ind_perm_sort = lexsort([deg_noise[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5
            
            elif 'Mamba_Hybrid_Noise_Bucket' == self.global_model_type:
                if batch.split == 'train':
                    deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(batch.batch.device)
                    h_ind_perm = lexsort([deg_noise, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    deg = batch.batch.to(torch.float)
                    for i in range(5):
                        deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(batch.batch.device)
                        h_ind_perm = lexsort([deg_noise, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == 'Mamba_Eigen':
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                centrality = batch.EigCentrality
                if batch.split == 'train':
                    # Shuffle within 1 STD
                    centrality_noise = torch.std(centrality)*torch.rand(centrality.shape).to(centrality.device)
                    # Order by batch, degree, and centrality
                    h_ind_perm = lexsort([centrality+centrality_noise, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        centrality_noise = torch.std(centrality)*torch.rand(centrality.shape).to(centrality.device)
                        h_ind_perm = lexsort([centrality+centrality_noise, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5
            
            elif 'Mamba_Eigen_Bucket' == self.global_model_type:
                centrality = batch.EigCentrality
                if batch.split == 'train':
                    centrality_noise = torch.std(centrality)*torch.rand(centrality.shape).to(centrality.device)
                    indices_arr, emb_arr = [],[]
                    bucket_assign = torch.randint_like(centrality, 0, self.NUM_BUCKETS).to(centrality.device)
                    for i in range(self.NUM_BUCKETS):
                        ind_i = (bucket_assign==i).nonzero().squeeze()
                        h_ind_perm_sort = lexsort([(centrality+centrality_noise)[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        centrality_noise = torch.std(centrality)*torch.rand(centrality.shape).to(centrality.device)
                        indices_arr, emb_arr = [],[]
                        bucket_assign = torch.randint_like(centrality, 0, self.NUM_BUCKETS).to(centrality.device)
                        for i in range(self.NUM_BUCKETS):
                            ind_i = (bucket_assign==i).nonzero().squeeze()
                            h_ind_perm_sort = lexsort([(centrality+centrality_noise)[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == 'Mamba_RWSE':
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                RWSE_sum = torch.sum(batch.pestat_RWSE, dim=1)
                if batch.split == 'train':
                    # Shuffle within 1 STD
                    RWSE_noise = torch.std(RWSE_sum)*torch.randn(RWSE_sum.shape).to(RWSE_sum.device)
                    # Sort in descending order
                    # Nodes with more local connections -> larger sum in RWSE
                    # Nodes with more global connections -> smaller sum in RWSE
                    h_ind_perm = lexsort([-RWSE_sum+RWSE_noise, batch.batch])
                    # h_ind_perm = lexsort([-RWSE_sum+RWSE_noise, deg, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    # Sort in descending order
                    # Nodes with more local connections -> larger sum in RWSE
                    # Nodes with more global connections -> smaller sum in RWSE
                    # h_ind_perm = lexsort([-RWSE_sum, deg, batch.batch])
                    mamba_arr = []
                    for i in range(5):
                        RWSE_noise = torch.std(RWSE_sum)*torch.randn(RWSE_sum.shape).to(RWSE_sum.device)
                        h_ind_perm = lexsort([-RWSE_sum+RWSE_noise, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5
            
            elif self.global_model_type == 'Mamba_Cluster':
                h_ind_perm = permute_within_batch(batch.batch)
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                if batch.split == 'train':
                    unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                    permuted_louvain = torch.zeros(batch.LouvainCluster.shape).long().to(batch.LouvainCluster.device)
                    random_permute = torch.randperm(unique_cluster_n+1).long().to(batch.LouvainCluster.device)
                    for i in range(unique_cluster_n):
                        indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                        permuted_louvain[indices] = random_permute[i]
                    #h_ind_perm_1 = lexsort([deg[h_ind_perm], permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]])
                    #h_ind_perm_1 = lexsort([permuted_louvain[h_ind_perm], deg[h_ind_perm], batch.batch[h_ind_perm]])
                    h_ind_perm_1 = lexsort([permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]])
                    h_ind_perm = h_ind_perm[h_ind_perm_1]
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    #h_ind_perm = lexsort([batch.LouvainCluster, deg, batch.batch])
                    #h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    #h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    #h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                    mamba_arr = []
                    for i in range(5):
                        unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                        permuted_louvain = torch.zeros(batch.LouvainCluster.shape).long().to(batch.LouvainCluster.device)
                        random_permute = torch.randperm(unique_cluster_n+1).long().to(batch.LouvainCluster.device)
                        for i in range(len(torch.unique(batch.LouvainCluster))):
                            indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                            permuted_louvain[indices] = random_permute[i]
                        # potentially permute it 5 times and average
                        # on the cluster level
                        #h_ind_perm_1 = lexsort([deg[h_ind_perm], permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]])
                        #h_ind_perm_1 = lexsort([permuted_louvain[h_ind_perm], deg[h_ind_perm], batch.batch[h_ind_perm]])
                        h_ind_perm_1 = lexsort([permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]])
                        h_ind_perm = h_ind_perm[h_ind_perm_1]
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5
        
            elif self.global_model_type == 'Mamba_Hybrid_Degree_Bucket':
                if batch.split == 'train':
                    h_ind_perm = permute_within_batch(batch.batch)
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                    indices_arr, emb_arr = [],[]
                    for i in range(self.NUM_BUCKETS): 
                        ind_i = h_ind_perm[h_ind_perm%self.NUM_BUCKETS==i]
                        h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        h_ind_perm = permute_within_batch(batch.batch)
                        deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                        indices_arr, emb_arr = [],[]
                        for i in range(self.NUM_BUCKETS):
                            ind_i = h_ind_perm[h_ind_perm%self.NUM_BUCKETS==i]
                            h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5
            
            elif self.global_model_type == 'Mamba_Cluster_Bucket':
                h_ind_perm = permute_within_batch(batch.batch)
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                if batch.split == 'train':
                    indices_arr, emb_arr = [],[]
                    unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                    permuted_louvain = torch.zeros(batch.LouvainCluster.shape).long().to(batch.LouvainCluster.device)
                    random_permute = torch.randperm(unique_cluster_n+1).long().to(batch.LouvainCluster.device)
                    for i in range(len(torch.unique(batch.LouvainCluster))):
                        indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                        permuted_louvain[indices] = random_permute[i]
                    for i in range(self.NUM_BUCKETS): 
                        ind_i = h_ind_perm[h_ind_perm%self.NUM_BUCKETS==i]
                        h_ind_perm_sort = lexsort([permuted_louvain[ind_i], deg[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        indices_arr, emb_arr = [],[]
                        unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                        permuted_louvain = torch.zeros(batch.LouvainCluster.shape).long().to(batch.LouvainCluster.device)
                        random_permute = torch.randperm(unique_cluster_n+1).long().to(batch.LouvainCluster.device)
                        for i in range(len(torch.unique(batch.LouvainCluster))):
                            indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                            permuted_louvain[indices] = random_permute[i]
                        for i in range(self.NUM_BUCKETS): 
                            ind_i = h_ind_perm[h_ind_perm%self.NUM_BUCKETS==i]
                            h_ind_perm_sort = lexsort([permuted_louvain[ind_i], deg[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == 'Mamba_Augment':
                aug_idx, aug_mask = augment_seq(batch.edge_index, batch.batch, 3)
                h_dense, mask = to_dense_batch(h[aug_idx], batch.batch[aug_idx])
                aug_idx_reverse = torch.nonzero(aug_mask).squeeze()
                h_attn = self.self_attn(h_dense)[mask][aug_idx_reverse]
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s
