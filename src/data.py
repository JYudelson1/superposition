from pickletools import float8
import torch

def generate_data(n: int, num_features: int, S: float) -> torch.Tensor:
    """Generates dummy data for the toy model

    Arguments:
        n -- number of datapoints
        num_features -- number of total features (what the paper calls n)
        S -- sparsity of the model. This means that each feature of each
            datapoint has a (1-S) probability of being zeroed

    Returns:
        A (n x num_features) matrix
    """
    assert S >= 0 and S < 1
    
    # Random data
    data = torch.rand(n, num_features, dtype=torch.float32, requires_grad=True)
    
    # If S==1, the matrix is fully dense, so return as is.
    if S == 0: return data
    
    # Else: we use torch.where to zero out features at random, with (1-S) chance
    rand_mask = torch.rand(n, num_features)
    return torch.where(
        rand_mask <= (1-S),
        data,
        torch.zeros(n, num_features, dtype=torch.float32, requires_grad=True))