import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch, k_hop_subgraph, degree
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE
from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from mamba_ssm import Mamba
from typing import List
import random
import time
import math

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

def dynamic_sampling_ratio(deg, max_degree, min_bound=(0.5, 0.4), max_bound=(0.9, 0.8)):
    normalized_degree = deg / max_degree

    low_bound = min_bound[1] + (max_bound[1] - min_bound[1]) * normalized_degree
    high_bound = min_bound[0] + (max_bound[0] - min_bound[0]) * normalized_degree
    
    low_bound = max(0.1, min(low_bound, 0.9))
    high_bound = max(low_bound, min(high_bound, 0.9))
    
    sampling_ratio = torch.rand(1).item() * (high_bound - low_bound) + low_bound
    
    return sampling_ratio

class SubgraphEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SubgraphEncoder, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x, edge_attr):
        # Your existing aggregation logic here
        target_device = x.device
        # Aggregate node features
        if x.size(0) > 0:  # Check if there are nodes
            node_features_aggregated = torch.mean(x, dim=0, keepdim=True).to(target_device)
        else:
            # Handle case with no nodes
            node_features_aggregated = torch.zeros(1, x.size(1)).to(target_device)
        
        # Aggregate edge features
        if edge_attr is not None and edge_attr.size(0) > 0:  # Check if there are edges
            edge_features_aggregated = torch.mean(edge_attr, dim=0, keepdim=True).to(target_device)
        else:
            # Handle case with no edges
            edge_features_aggregated = torch.zeros(1, edge_attr.size(1)).to(target_device) if edge_attr is not None else torch.zeros(1, x.size(1)).to(target_device)
        
        subgraph_representation = torch.cat([node_features_aggregated, edge_features_aggregated], dim=1).to(target_device)
        del edge_features_aggregated
        del node_features_aggregated
        # Project back to original dimension
        return self.projection(subgraph_representation)


def calculate_max_k(degrees, min_k=2, max_k=4, scale_factor=1.0):
    normalized_degrees = (degrees - degrees.min()) / (degrees.max() - degrees.min())
    
    k_values = max_k - (normalized_degrees * (max_k - min_k) * scale_factor)
    
    k_values = torch.clamp(k_values, min=min_k, max=max_k)
    
    k_values = torch.round(k_values).long()
    
    return k_values



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
            # nn.BatchNorm1d(expand*d_model),
            nn.Linear(expand*d_model, d_model)
        )

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1)

        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        x = self.mlp(x)
        x = x.reshape(x_shape)

        x = (x*mask).sum(dim=-2, keepdim=True) / mask.sum(dim=-2, keepdim=True)
        x = x.expand(x_shape)
        return x
    
class ConvModel(nn.Module):
    def __init__(self, d_model, expand, kernel_size=11):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, expand*d_model),
            nn.ReLU(),
            nn.Linear(expand*d_model, d_model)
        )
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding='same', groups=d_model)

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1)

        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])
        x = self.mlp(x)
        x = x.reshape(x_shape)
        x = mask*x

        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)

        return x

class GraphSSM(nn.Module):
    def __init__(self,
                 d_model, # D in Mamba paper
                 d_state, # N in Mamba paper
                 expand, # E in Mamba paper
                 A_complex=True,
                 BC_complex=False,
                 epsilon=0.001,
                 ):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model

        d_inner = expand*d_model # Number of channels in the SSM
        self.d_inner = d_inner
        self.in_proj = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, 2*d_inner),
        )
        Delta_min = 0.001
        Delta_max = 0.1
        self.log_Delta = nn.Parameter(math.log(Delta_min) + (math.log(Delta_max)-math.log(Delta_min))*torch.rand(1, d_inner))
        self.A_complex = A_complex
        self.BC_complex = BC_complex
        if BC_complex:
            self.B = nn.Parameter(0.01*torch.complex(torch.randn(1, d_inner, d_state), torch.randn(1, d_inner, d_state)))
            self.C = nn.Parameter(0.01*torch.complex(torch.randn(1, d_inner, d_state), torch.randn(1, d_inner, d_state)))
        else:
            self.B = nn.Parameter(0.01*torch.randn(1, d_inner, d_state))
            self.C = nn.Parameter(0.01*torch.randn(1, d_inner, d_state))
        if A_complex:
            self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
            self.A_imag = nn.Parameter(torch.pi * torch.arange(d_model).expand(d_state, d_model).T)
        else:
            self.log_A = nn.Parameter(torch.rand(1, d_inner, d_state))
        self.out_proj = nn.Linear(d_inner, d_model)

        self.Abar_epsilon = epsilon
 
    def _change_basis(self, Q, X):
        # Q: (B, L', L)
        # X: (B, L, ED, N)
        # output: (B, L', ED, N)
        Xa = X.view(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])  # (B, L, ED*N)
        QX = torch.bmm(Q, Xa)  # (B, L', ED*N)
        QX = QX.view(X.shape[0], Q.shape[1], X.shape[2], X.shape[3])  # (B, L', ED, N)
        return QX

    def forward(self, x, attn_mask, EigVecs, EigVals):
        # x: (B, L, D)
        # attn_mask: (B, L)
        # EigVecs: (B, L, L')
        # EigVals: (B, L')
        EigVals = EigVals.view(*EigVals.shape, 1, 1) # Eigenvalues of L = 1-M
        EigVals = 1.0 - EigVals # Eigenvalues of M = 1-L
        
        EigVals = torch.nan_to_num(EigVals, nan=0.0)
        EigVecs = torch.nan_to_num(EigVecs, nan=0.0)
        EigVecs = EigVecs / EigVecs.norm(dim=-1, keepdim=True)
        EigVecs = torch.nan_to_num(EigVecs, nan=0.0)

        if self.BC_complex:
            EigVecs = torch.complex(EigVecs, torch.zeros_like(EigVecs))
        
        # print(x.norm(dim=-1).max().item(), x.norm(dim=-1).min().item())

        # Gated MLP
        B, L, D = x.shape
        x = x.reshape(B*L, D)
        x = self.in_proj(x).reshape(B, L, 2*self.d_inner) # (B, L, 2*ED)
        x, z = torch.split(x, [self.d_model, self.d_model], dim=-1) # (B, L, ED), (B, L, ED)
        x = x*F.silu(z) # (B, L, ED)

        # Selective parameters
        B = self.B # (1, ED, N)
        C = self.C # (1, ED, N)
        delta = torch.exp(self.log_Delta) # (1, ED)
        if self.A_complex:
            A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)
        else:
            A = -torch.exp(self.log_A) # (1, ED, N)
        
        # Discretisation
        deltaA = delta.unsqueeze(-1) * A # (1, ED, N)
        Abar = torch.exp(-self.Abar_epsilon + deltaA) # (1, ED, N) # epsilon so that Abar is not too close to 1 (which happens in practice)
        deltaB = delta.unsqueeze(-1) * B # (1, ED, N)
        Bbar = deltaB # (1, ED, N)

        X = x.unsqueeze(-1) # (B, L, ED, 1)

        # Propagate along graph

        # XB = (X * Bbar) # (B, L, ED, N)
        # QTXB = self._change_basis(EigVecs.permute(0,2,1), XB) # (B, L', ED, N)
        QTX = self._change_basis(EigVecs.permute(0,2,1), X) # (B, L', ED, 1)
        QTXB = (QTX * Bbar) # (B, L', ED, N)

        lambdaA = EigVals*Abar.unsqueeze(1)
        lambdaQTXBA = (lambdaA / (1.0 - lambdaA)) * QTXB # (B, L', ED, N)
        if self.A_complex and not self.BC_complex:
            lambdaQTXBA = lambdaQTXBA.real
        MXBA = self._change_basis(EigVecs, lambdaQTXBA) # (B, L, ED, N)

        # Project back to original space
        MXBAC = (MXBA * C.unsqueeze(0)).sum(dim=-1) #(B, L, ED)
        if self.BC_complex:
            MXBAC = MXBAC.real

        y = MXBAC # (B, L, ED)

        # Output projection
        y = F.silu(y) # (B, L, ED)
        y = self.out_proj(y) # (B, L, D)
        return y
    


class GraphSSSM(nn.Module):
    def __init__(self,
                 d_model, # D in Mamba paper
                 d_state, # N in Mamba paper
                 expand, # E in Mamba paper
                 B_selective=True,
                 C_selective=True,
                 Delta_selective=True):
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model

        d_inner = expand*d_model # Number of channels in the SSM
        self.d_inner = d_inner
        self.in_proj = nn.Linear(d_model, 2*d_inner)
        self.s_proj = nn.Linear(d_inner, 1+2*d_state) # s_Delta, s_B, s_C in Mamba paper
        self.parameter_delta = nn.Parameter(torch.zeros(1, 1, d_inner))
        # self.A_log = nn.Parameter(torch.complex(torch.rand(1, d_inner, d_state), 1.0-2.0*torch.rand(1, d_inner, d_state)))
        self.A_log = nn.Parameter(torch.rand(1, d_inner, d_state))
        self.out_proj = nn.Linear(d_inner, d_model)

    def _change_basis(self, Q, X):
        # Q: (B, L', L)
        # X: (B, L, ED, N)
        # output: (B, L', ED, N)
        Xa = X.view(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])  # (B, L, ED*N)
        QX = torch.bmm(Q, Xa)  # (B, L', ED*N)
        QX = QX.view(X.shape[0], Q.shape[1], X.shape[2], X.shape[3])  # (B, L', ED, N)
        return QX

    def forward(self, x, attn_mask, EigVecs, EigVals):
        # x: (B, L, D)
        # attn_mask: (B, L)
        # EigVecs: (B, L, L')
        # EigVals: (B, L')
        attn_mask = attn_mask.view(*attn_mask.shape, 1, 1)
        EigVals = EigVals.view(*EigVals.shape, 1, 1) # Eigenvalues of L = 1-M
        EigVals = 1.0 - EigVals # Eigenvalues of M = 1-L


        # Gated MLP
        x = self.in_proj(x) # (B, L, 2*ED)
        x, z = torch.split(x, [self.d_model, self.d_model], dim=-1) # (B, L, ED), (B, L, ED)
        x = x*F.silu(z) # (B, L, ED)

        # Selective parameters
        deltaBC = self.s_proj(x) # (B, L, 1+2*N)
        delta, B, C = torch.split(deltaBC, [1, self.d_state, self.d_state], dim=-1) # (B, L, 1), (B, L, N), (B, L, N)
        delta = F.softplus(self.parameter_delta + delta.expand(-1, -1, self.d_inner)) # (B, L, ED)
        A = -torch.exp(self.A_log) # (1, ED, N)
        
        # Discretisation
        deltaA = delta.unsqueeze(-1) * A # (B, L, ED, N)
        Abar = torch.exp(deltaA) # (B, L, ED, N)
        # Bbar_A_multiplier = ((Abar - 1.0) / deltaA) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)
        Bbar = deltaB # (B, L, ED, N)

        # Compute transition matrix
        XB = (x.unsqueeze(-1) * Bbar) # (B, L, ED, N)
        # XBA = XB * (1.0 / (1.0 - Abar)) # (B, L, ED, N)
        XBA = XB
        # XBA = XBA.real

        # # Propagate along graph
        QTXBA = self._change_basis(EigVecs.permute(0,2,1), XBA) # (B, L', ED, N)
        lambdaQTXBA = (1.0 / (1.0 - EigVals)) * QTXBA # (B, L', ED, N)
        MXBA = self._change_basis(EigVecs, lambdaQTXBA) # (B, L, ED, N)
        # MXBA = XBA

        # Project back to original space
        MXBAC = (MXBA @ C.unsqueeze(-1)).squeeze(-1) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1) -> (B, L, ED)
        # MXBAC = MXBAC.real

        # Residual connection
        y = x + MXBAC # (B, L, ED)

        # Output projection
        y = F.silu(y) # (B, L, ED)
        y = self.out_proj(y) # (B, L, D)
        print(y.shape)
        return y


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
        elif 'MeanL65' == global_model_type:
            self.self_attn = MeanModel(d_model=dim_h, # Model dimension d_model
                        expand=2,    # Block expansion factor
                    )
        elif 'MultiMambaL65' in global_model_type:
            self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,    # Local convolution width
                    expand=1,    # Block expansion factor
                )
        elif 'Subgraph_Mamba_L65' in global_model_type:
            self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,    # Local convolution width
                    expand=1,    # Block expansion factor
                )
            self.subgraph_encoder = SubgraphEncoder(dim_h + dim_h, dim_h) # TODO(guillem): assuming the number of edge features is the same as the number of node features
        elif 'ConvL65' in global_model_type:
            kernel_size = int(global_model_type.split('_')[-1])
            self.self_attn = ConvModel(d_model=dim_h, # Model dimension d_model
                        expand=2,    # Block expansion factor
                        kernel_size=kernel_size
                    )
        elif 'GraphSSML65' in global_model_type:
            self.self_attn = GraphSSM(d_model=dim_h, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                expand=1,    # Block expansion factor
            )
        elif 'SharedMambaL65' in global_model_type:
            self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=1,    # Block expansion factor
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
            if self.global_model_type in ['Transformer', 'Performer', 'BigBird', 'Mamba', 'MeanL65', 'ConvL65', 'GraphSSML65']:
                h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)


            elif 'MultiMambaL65' in self.global_model_type:
                repeats = int(self.global_model_type.split("_")[-1])
                num_nodes = batch.x.shape[0]
                heuristic = heuristic_fns[self.mamba_heuristics[0]][0](batch)[0]

                encodings_sequence = torch.repeat_interleave(h, repeats, dim=0)
                # Copy heuristics where the first repeat has the right sign, the second is negated, and so on
                heuristics_sequence = torch.repeat_interleave(heuristic, repeats, dim=0)
                for i in range(1, repeats, 2):
                    heuristics_sequence[i::repeats] = -heuristics_sequence[i::repeats]
                batches_sequence = torch.repeat_interleave(batch.batch, repeats, dim=0)
                subgraph_indices_sequence = torch.arange(repeats, device=encodings_sequence.device).repeat(num_nodes)
                node_indices = (repeats-1) + repeats*torch.arange(num_nodes, device=encodings_sequence.device)

                # Reorder the sequence according to the heuristics and the index of the subgraph or whether it is a node
                h_ind_perm = lexsort([heuristics_sequence, subgraph_indices_sequence, batches_sequence])
                h_dense, mask = to_dense_batch(encodings_sequence[h_ind_perm], batches_sequence[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)

                # Run mamba to compute the final embeddings for subgraphs and nodes
                final_embeddings = self.self_attn(h_dense)[mask]

                # Get the indices of the n-th node in the final embedding sequence and get the corresponding embedding
                node_indices_in_reordered_sequence_reverse = h_ind_perm_reverse[node_indices]
                h_attn = final_embeddings[node_indices_in_reordered_sequence_reverse]


            elif self.global_model_type == 'Subgraph_Mamba_L65':
                print('Lets go')
                degrees = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                num_nodes = batch.x.shape[0]
                heuristic = heuristic_fns[self.mamba_heuristics[0]][0](batch)[0]

                encodings = []
                heuristics = []
                subgraph_indices = [] # Stores the index of a subgraph for each specific node, or the node itself
                batches = []
                node_indices = []

                # Unique graphs in the batch
                unique_graphs = torch.unique(batch.batch)
                sampled_node_indices = []
                num_samples_per_graph = 3
                num_strata = 3
                device = batch.batch.device

                start_time = time.time()
                from tqdm import tqdm

                for graph_id in tqdm(unique_graphs):
                    # Mask for nodes in the current graph
                    graph_mask = (batch.batch == graph_id)
                    
                    # Extract nodes, degrees, and heuristic values for the current graph
                    graph_degrees = degrees[graph_mask]
                    graph_heuristic = heuristic[graph_mask]
                    graph_node_indices = torch.arange(len(batch.batch), device=device)[graph_mask]
                    
                    # Sort nodes within the graph by heuristic, then degree
                    sorted_indices = lexsort([graph_degrees, graph_heuristic])
                    sorted_graph_node_indices = graph_node_indices[sorted_indices]
                    
                    # Determine the number of samples per stratum
                    samples_per_stratum = max(1, num_samples_per_graph // num_strata)
                    
                    for stratum in range(num_strata):
                        # Determine start and end indices for this stratum
                        start = (len(sorted_graph_node_indices) // num_strata) * stratum
                        end = min(start + samples_per_stratum, len(sorted_graph_node_indices))
                        
                        # Sample nodes from this stratum
                        stratum_sample_indices = sorted_graph_node_indices[start:end]
                        
                        sampled_node_indices.append(stratum_sample_indices)
                
                # Combine samples from all strata and all graphs
                sampled_node_indices = torch.cat(sampled_node_indices).to(device)
                

                # NOTE(guillem): For now, we will order the sequence as (sugraph 1 for node 1), (sugraph 1 for node 2), ..., (subgraph 1 for node `num_nodes-1`), (subgraph 2 for node 1), ..., node 1, node 2, ...
                # where the nodes are ordered according to `heuristic` 

                max_k_values = calculate_max_k(degrees[sampled_node_indices])
                    
                unique_max_ks = torch.unique(max_k_values).tolist()

                # Process nodes in batches based on their unique max_k values
                for current_max_k in tqdm(range(2, max(unique_max_ks)+1)):
                    new_node_indices = (current_max_k <= max_k_values).nonzero(as_tuple=True)[0]
                    
                    # Compute k-hop subgraph for all nodes with the current max_k
                    _, sub_edges, _, edge_mask = k_hop_subgraph(
                        new_node_indices, current_max_k, batch.edge_index, num_nodes=num_nodes)

                    for node_idx in new_node_indices:
                        # Extract the subgraph for the current node

                        # Find edges connected to the node
                        connected_edges = (sub_edges == node_idx).any(dim=0)
                        
                        # Find all unique nodes connected through these edges
                        connected_nodes = sub_edges[:, connected_edges].unique()

                        current_x = batch.x[connected_nodes]
                        current_edge_attr = batch.edge_attr[edge_mask][connected_edges]
                        
                        # Encode the subgraph
                        embedding = self.subgraph_encoder(current_x, current_edge_attr)
                        encodings.append(embedding)
                        heuristics.append(heuristic[node_idx])
                        batches.append(batch.batch[node_idx])
                        subgraph_indices.append(current_max_k)
                    

                encodings_sequence = torch.cat(encodings, dim=0)
                heuristics_sequence = torch.stack(heuristics)
                batches_sequence = torch.stack(batches)
                subgraph_indices_sequence = torch.tensor(subgraph_indices, device=encodings_sequence.device)


                encodings_sequence = torch.cat([encodings_sequence, h], dim=0)
                heuristics_sequence = torch.cat([heuristics_sequence, heuristic], dim=0)
                batches_sequence = torch.cat([batches_sequence, batch.batch], dim=0)
                subgraph_indices_sequence = torch.cat([subgraph_indices_sequence, torch.ones(num_nodes, device=encodings_sequence.device)*999], dim=0)
                node_indices = len(encodings) + torch.arange(num_nodes, device=encodings_sequence.device)
                
                

                # Reorder the sequence according to the heuristics and the index of the subgraph or whether it is a node
                h_ind_perm = lexsort([heuristics_sequence, subgraph_indices_sequence, batches_sequence])
                h_dense, mask = to_dense_batch(encodings_sequence[h_ind_perm], batches_sequence[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                del heuristics_sequence, batches_sequence
                torch.cuda.empty_cache()

                # Run mamba to compute the final embeddings for subgraphs and nodes
                final_embeddings = self.self_attn(h_dense)[mask]

                # Get the indices of the n-th node in the final embedding sequence and get the corresponding embedding
                node_indices_in_reordered_sequence_reverse = h_ind_perm_reverse[node_indices]
                h_attn = final_embeddings[node_indices_in_reordered_sequence_reverse]
                end_time = time.time()
                print(f"Execution time: {end_time - start_time} seconds")
            
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
            
            elif self.global_model_type == 'SharedMambaL65':

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
                                h_dense = self.self_attn(h_dense)[mask]
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
                            h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

                        else:
                            h_ind_perm = lexsort([heuristic, batch.batch])
                            h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                            h_ind_perm_reverse = torch.argsort(h_ind_perm)
                            h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

                        mamba_arr.append(h_attn)

                h_attn = sum(mamba_arr) / len(mamba_arr)
            
            elif self.global_model_type == 'MeanL65':
                h_attn = self.self_attn(h_dense, mask)[mask]  
            elif 'ConvL65' in self.global_model_type:

                permute_iterations = self.mamba_permute_iterations
                if batch.split == 'train':
                    permute_iterations = 1

                mamba_arr = []
                for _ in range(max(permute_iterations, 1)):
                    h_ind_perm = permute_within_batch(batch.batch)
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense, mask)[mask][h_ind_perm_reverse]

                    mamba_arr.append(h_attn)

                h_attn = sum(mamba_arr) / len(mamba_arr)  
            
            elif self.global_model_type == 'GraphSSML65':
                EigVecs_dense, _ = to_dense_batch(batch.EigVecs, batch.batch)
                EigVals_dense, _ = to_dense_batch(batch.EigVals.squeeze(-1), batch.batch)
                h_attn = self.self_attn(h_dense, mask, EigVecs_dense, EigVals_dense[:,0,:])[mask]  
            
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
