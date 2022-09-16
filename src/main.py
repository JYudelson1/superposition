from model import ToyModel
from data import generate_data
import torch

NUM_FEATURES = 3
NUM_DIMENSIONS = 2
BATCH_SIZE = 2

m = ToyModel(num_features=NUM_FEATURES, num_dimensions=NUM_DIMENSIONS, S=0.5)
data: torch.Tensor = generate_data(n=BATCH_SIZE, num_features=NUM_FEATURES, S=0.5)
print(m(data))