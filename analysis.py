from torch_utils import dist_from_etf, hook_group, remove_all_hooks, matrix_rank, features
import torch.nn as nn
import torch
from tqdm import tqdm

# custom analysis function
def cos_analysis(model: torch.nn.Module, modules: list[torch.nn.Module], loader: torch.utils.data.DataLoader, num_classes: int, output_layer=True, delta=0.05, criterion_summed=torch.nn.CrossEntropyLoss(reduction='sum'), device=None):
    model.eval()
    if device is None:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_modules = len(modules)
    num_modules_o = num_modules + 1 if output_layer else num_modules
    remove_all_hooks(model)
    hook_group(modules)

    # Output variables, features are all centered
    loss = 0
    N = 0
    weight_norms = []
    bn_norms = []
    intra_cos = []
    inter_cos = []
    avg_intra = []
    avg_inter = []
    delta_intra = []
    delta_inter = []
    qmean_norms = []
    # Weight Matrix Rank
    ranks = []
    feature_dims = [0 for _ in range(num_modules_o)]

    # Since no update here anyway
    with torch.no_grad():
      # Data-independent measures
      for m in modules:
        if isinstance(m, nn.Linear):
          weight_norms.append(torch.norm(m.weight).cpu().item())
          ranks.append(matrix_rank(m.weight))
        if isinstance(m, nn.BatchNorm1d):
          bn_norms.append(torch.norm(m.weight).cpu().item())

      # First interation to get global means, class means, loss and accuracy

      # Global means for each layer
      means = [0 for _ in range(num_modules_o)]

      # Class Means for each layer for each class
      class_means = [[0 for _ in range(num_classes)] for _ in range(num_modules_o)]

      # Total number of data
      cnt = 0

      # Number of data for each class
      class_cnt = [0 for _ in range(num_classes)]
      pbar = tqdm(total=len(loader), position=0, leave=True)
      for batch_idx, (data, target) in enumerate(loader, start=1):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss += criterion_summed(output, target)
        N += target.shape[0]
        for i in range(num_modules_o):
          if i == num_modules:
            class_features_i = output.view(output.shape[0], -1)
            feature_dims[i] = class_features_i.shape[1]
          else:
            class_features_i = features.values[i].view(output.shape[0], -1)
          for c in range(num_classes):
            c_indices = (target == c)
            if i == 0:
              class_cnt[c] += c_indices.int().sum().item()
            class_means[i][c] =  class_means[i][c] + class_features_i[c_indices].sum(axis=0)
          means[i] += class_features_i.sum(axis=0)
          if i == 0:
            cnt += class_features_i.shape[0]
        pbar.update(1)
      # change sums to means
      for i in range(num_modules_o):
        means[i] /= cnt
        for c in range(num_classes):
          class_means[i][c] /= class_cnt[c]
        class_means[i] = torch.stack(class_means[i])
      loss /= N

      # Second iteration computes cos similarities

      # Num of vecs for each class
      cnts = [0 for _ in range(num_classes)]

      # Unit Norm Feature Vectors
      normed_vecs = [[0 for _ in range(num_classes)] for _ in range(num_modules_o)]

      # Feature Vector Norms
      norms = [0 for _ in range(num_modules_o)]

      # Nearest Class Center Classifcation Accuracy
      nccs = [0 for _ in range(num_modules_o)]

      pbar = tqdm(total=len(loader), position=0, leave=True)
      for batch_idx, (data, target) in enumerate(loader, start=1):
        num_samples = 0
        data, target = data.to(device), target.to(device)
        output = model(data)

        for i in range(num_modules_o):
          if i == num_modules:
            class_features_i = output.view(output.shape[0], -1)
          else:
            class_features_i = features.values[i].view(output.shape[0], -1)

          # Compute Class Mean in the First Interation

          # Center feature relative to global mean
          centered_features =  class_features_i - means[i]

          # Quadratic Average of Vector Norms
          norms[i] += torch.norm(centered_features) ** 2
          # pairwise_dists = torch.norm(class_features_i[:, None, :] - class_means[i][None, :, :], dim=-1)

          # Number of correct NCC predictions
          # ncc_pred = torch.argmin(pairwise_dists, dim=1)
          # nccs[i] += (ncc_pred == target).int().sum().item()
          for c in range(num_classes):
            # Features for class c
            centered_features_c = centered_features[target == c]
            # Normalize each feature to norm 1
            centered_features_c_normed = centered_features_c / torch.norm(centered_features_c, dim=1).reshape(-1, 1)
            normed_vecs[i][c] += centered_features_c_normed.sum(dim=0)
            if i == 0:
              cnts[c] += centered_features_c.shape[0]

        pbar.update(1)
      cnts = torch.tensor(cnts).to(device)
      # Compute Inter and Intra cos using mean normalized vectors
      for i in range(num_modules_o):
        nccs[i] /= cnt
        qmean_norms.append(torch.sqrt(norms[i] / cnt).item())
        for c in range(num_classes):
          if cnts[c] == 0:
             normed_vecs[i][c] = torch.ones_like(normed_vecs[i][c])
             normed_vecs[i][c] /= torch.norm(normed_vecs[i][c])
             print(f"Warning: No data for class {c}, setting mean to all ones vector.")
          else:
            normed_vecs[i][c] /= cnts[c]
        full_cos = torch.stack(normed_vecs[i]) @ torch.stack(normed_vecs[i]).T
        intra_cos_vals = torch.diag(full_cos)
        inter_cos_vals = full_cos[torch.eye(full_cos.shape[0], dtype=int) == 0]
        cnts[cnts < 2] = 2
        intra_cos_vals *= (cnts / (cnts - 1))
        intra_cos_vals -= (1 / (cnts - 1))
        intra_cos.append(intra_cos_vals.min().item())
        inter_cos.append(inter_cos_vals.max().item())
        avg_intra.append(intra_cos_vals.mean().item())
        avg_inter.append(inter_cos_vals.mean().item())
        delta_intra.append(intra_cos_vals.quantile(delta).item())
        delta_inter.append(inter_cos_vals.quantile(1 - delta).item())


    return loss, intra_cos, inter_cos, avg_intra, avg_inter, delta_intra, delta_inter, qmean_norms, bn_norms, weight_norms, nccs, ranks

def cos_analysis_str(loss, intra_cos, inter_cos, avg_intra, avg_inter, delta_intra, delta_inter, qmean_norms, bn_norms, weight_norms, nccs, ranks, hooked_modules, delta):
  output_str = ""
  bn_id = 0
  linear_id = 0
  num_modules = len(hooked_modules)
  output_str += f"Loss: {loss}\n"
  for i in range(num_modules + 1):
    if i == num_modules:
      output_str += f"Output Layer:\n"
    else:
      output_str += f"Layer {i}: {hooked_modules[i]}\n"
      if isinstance(hooked_modules[i], nn.Linear):
        output_str += f"Linear Weight Norm: {weight_norms[linear_id]}\n"
        output_str += f"Linear Weight Rank: {ranks[linear_id]}\n"
        linear_id += 1
      elif isinstance(hooked_modules[i], nn.BatchNorm1d):
        output_str += f"Batch Normalization Weight Norm: {bn_norms[bn_id]}\n"
        bn_id += 1
    output_str += f"Intra Cos: {intra_cos[i]}\n"
    output_str += f"Inter Cos: {inter_cos[i]}\n"
    output_str += f"Intra Avg: {avg_intra[i]}\n"
    output_str += f"Inter Avg: {avg_inter[i]}\n"
    output_str += f"Intra Delta (delta={delta}): {delta_intra[i]}\n"
    output_str += f"Inter Delta (delta={delta}): {delta_inter[i]}\n"
    output_str += f"Norm Quadratic Average: {qmean_norms[i]}\n"
    output_str += f"Nearest Class Center Accuracy: {nccs[i]}\n"
    output_str += "\n"
  return output_str