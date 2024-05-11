import numpy as np

def generate_data_plane(num_classes, num_points, dim):
    r"""
    Generate conic hull data through plane separation
    """
    num_planes = int(np.ceil(np.log2(num_classes)))
    planes = np.random.randn(num_planes, dim)
    datapoints = np.random.randn(num_points, dim)
    class_bin = ((datapoints @ planes.T) > 0).astype(int)
    class_orig = (class_bin * (2 ** np.arange(class_bin.shape[1])[::-1])).sum(axis=1).astype('float')
    diff = 2 ** num_planes - num_classes
    class_orig[class_orig < 2 * diff] //= 2.0
    class_orig[class_orig >= 2 * diff] -= diff
    print('Data count:', np.unique(class_orig.astype(int), return_counts=True))
    return datapoints, class_orig.astype(int)

# %%
def generate_MLP_data(num_points, dim):
  import time
  set_all_seeds(989996)
  data_model = MLP(layer_width).to(device)
  datapoints = np.random.randn(num_points, dim)
  data_tensor = torch.tensor(datapoints).to(device).float()
  outputs = model(data_tensor)
  _, labels = torch.max(outputs, 1)
  labels_numpy = labels.cpu().detach().numpy()
  set_all_seeds(rand_seed)
  return datapoints, labels_numpy, data_model