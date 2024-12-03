import torch

if torch.cuda.is_available():
    print("GPU.")
else:
    print("PyTorch is available but using the CPU.")
