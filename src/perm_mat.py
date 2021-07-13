import torch


def make_p_mat(K):
    """Make a matrix of permuations."""
    mat = [[int(k1 == k2) for k2 in range(len(K))] for k1 in K]
    return torch.tensor(mat)


def transform(A, P):
    """Transform A under P"""
    return torch.matmul(P.T, torch.matmul(A, P))
