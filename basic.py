import torch

t = torch.tensor([1, 2])
t = t.unsqueeze(0)
print(t)
t = t.squeeze()
print(t)
