import numpy as np
import scipy.misc as sci
import torch
import networkx as nx
from scipy.linalg import expm

def torch_kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2
'''
t1 = torch.randn(2, 2)
t2 = torch.randn(2, 2)
'''

def compute_similarity(adj_A, adj_B):
	# compute numpy kronecker product
	adj_C = np.kron(adj_A.todense(), adj_B.todense())
	# obtain degrees matrix
	#D = np.diagflat(np.sum(adj_C,0))
	D = np.eye(adj_C.shape[0])
	# obtain normalized adjacency of C
	norm_adj_C = adj_C * np.linalg.inv(D)
	# using equation 19 as the kernel
	_lambda = 0.3
	# computing the kernel
	K = expm(_lambda*norm_adj_C)
	# ones vector
	e = np.ones((1,K.shape[0]))
	# similarities
	similarity = np.matmul(np.matmul(e, K), e.T)
	
	return similarity

def ray_similarity(adj_A, adj_B):
	pass

# generate random graphs
A = nx.star_graph(3)
B = nx.cycle_graph(4)
# generate adjacency matrix 
adj_A = nx.adjacency_matrix(A)
adj_B = nx.adjacency_matrix(B)
# compute similarities
ab = compute_similarity(adj_A, adj_B)
aa = compute_similarity(adj_A, adj_A)
bb = compute_similarity(adj_B, adj_B)


















































