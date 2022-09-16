import torch
from torch import nn

class ToyModel(nn.Module):
    """
    This is the toy model used in the paper. 
    It's a very simple NN, one pass + a ReLU.
    Needs to be variable on sparsity.
    """
    
    def __init__(self, num_features: int, num_dimensions: int, S: float) -> None:
        super().__init__()
        self.S = S
        self.W: torch.Tensor = torch.rand(num_features, num_dimensions)
        self.bias = torch.zeros(num_features)
        
    def forward(self, x):
        # First, the input vector of size (num_features)
        # gets multiplied by W to convert it into the smaller
        # dimensionality of (num_dimensions)
        # which the paper calls conversion from x ∈ R^n -> h ∈ R^m
        # where here n = num_features, m = num_dimensions

        h = torch.mm(x, self.W)
        
        # Then, we use W^T as a linear layer to return h to R^n space
        out = torch.mm(h, self.W.transpose(0,1)) + self.bias
        
        return nn.functional.relu(out)
    
