import torch
import numpy as np
import scipy

print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)

x = torch.tensor([1.0,2.0,3.0])
print(x)