import torch
from collections import OrderedDict

# Compute how close to ETF is a set of features w.r.t. labels
def dist_from_etf(x: torch.Tensor, y: torch.Tensor, lamb=0.1):
  device = x.device
  num_classes = len(torch.unique(y))
  total_var = torch.tensor(0).to(device)
  means = []
  for i in torch.unique(y):
    x_i = x[y == i]
    means.append(x_i.mean(axis=0))
    x_central = x_i - x_i.mean(axis=0)
    total_var = total_var + (x_central ** 2).sum() / x_i.shape[0]
  total_var = total_var / num_classes
  means = torch.stack(means)
  means = means / torch.linalg.norm(means, axis=1).view(-1, 1)
  cos_sim = means @ means.T
  cos_sim_flat = cos_sim.masked_select(~torch.eye(num_classes, dtype=bool).to(device))
  cos_diff = ((cos_sim_flat + 1 / (num_classes - 1)) ** 2).mean()
  return total_var + lamb * cos_diff

# Compute approximate rank of matrix
def matrix_rank(m, eps=0.01):
  u, s, vh = torch.svd(m)
  total = 0
  s /= s.sum()
  for i in range(len(s)):
    total += s[i].item()
    if total > 1 - eps:
      return i + 1
  raise Exception("SVD Invalid")

# Hooking Related
class Features:
    def __init__(self):
      self.values = {}

features = Features()

def hook(self, input, output):
    features.value = input[0].clone()

def hook_helper(module, i):
    def hook_temp(self, input, output):
      features.values[i] = input[0].clone()
    module.register_forward_hook(hook_temp)

def hook_group(modules):
  for i, module in enumerate(modules):
    hook_helper(module, i)

def remove_all_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks = OrderedDict()
            elif hasattr(child, "_forward_pre_hooks"):
                child._forward_pre_hooks = OrderedDict()
            elif hasattr(child, "_backward_hooks"):
                child._backward_hooks = OrderedDict()
            remove_all_hooks(child)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)