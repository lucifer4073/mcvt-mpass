import torch
from torch.nn import functional as F
def compute_distance_matrix(q_feats, g_feats, metric='cosine'):
    """A wrapper function for computing distance matrix.

    Args:
        q_feats (torch.Tensor): 2-D feature matrix.
        g_feats (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        torch.Tensor: distance matrix.

    Examples::
       >>> from torchreid import metrics
       >>> q_feats = torch.rand(10, 2048)
       >>> g_feats = torch.rand(100, 2048)
       >>> distmat = metrics.compute_distance_matrix(q_feats, g_feats)
       >>> distmat.size()  # (10, 100)
    """

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(q_feats, g_feats)
    elif metric == 'cosine':
        distmat = cosine_distance(q_feats, g_feats)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat

def euclidean_squared_distance(q_feats,g_feats):
    
    euclidean_distance = torch.cdist(q_feats, g_feats, p=2)

    return euclidean_distance
def cosine_distance(q_feats, g_feats):
    """Computes cosine distance.

    Args:
        q_feats (torch.Tensor): 2-D feature matrix normalized
        g_feats (torch.Tensor): 2-D feature matrix normalized

    Returns:
        torch.Tensor: distance matrix.
    """
    q_feats=F.normalize(q_feats, p=2, dim=1)
    g_feats=F.normalize(g_feats,p=2,dim=1)
    distmat = 1 - torch.mm(q_feats,g_feats.t())
    return distmat
