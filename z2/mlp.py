from torch import nn
import torch
import torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(3, 20)
        self.hidden1 = nn.Linear(20, 40)
        self.hidden2 = nn.Linear(40, 20)
        self.output = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.output(x))
        return x
