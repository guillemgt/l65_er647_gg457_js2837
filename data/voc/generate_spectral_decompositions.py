import pickle
import numpy as np
from tqdm import tqdm
import torch

for split in ['train', 'val']:

    data = pickle.load(open(f'{split}.pickle', 'rb'))

    output = []

    print("Processing", split, "split...")

    for sample in tqdm(data):
        node_features = sample[0] # (N, 12+2): (node features)+(2D coordinates)
        edge_features = sample[1] # (E, 1+1): (weight)+(length of boundary)
        edges = sample[2]
        N = node_features.shape[0]
        E = edge_features.shape[0]
        adj = np.zeros((N, N))
        for i in range(E):
            adj[edges[0,i], edges[1,i]] = 1
            adj[edges[1,i], edges[0,i]] = 1

        # Construct Laplacian
        D = np.sum(adj, axis=1)
        Dm12 = np.diag(np.power(D, -0.5))
        L = np.eye(N) - np.dot(np.dot(Dm12, adj), Dm12)

        L_eigenvalues, L_eigenbasis = np.linalg.eigh(L)

        output.append((torch.tensor(L_eigenvalues, dtype=torch.float32), torch.tensor(L_eigenbasis, dtype=torch.float32)))

    pickle.dump(output, open(f'{split}_spectral_decomposition.pickle', 'wb'))