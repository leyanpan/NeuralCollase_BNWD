from torchvision import datasets, transforms
import torch
import numpy as np
from imagenet32_dataset import ImageNet32

custom_data_path = {
    'conic': 'data/conic.npz',
    'mlp6': 'data/mlp.npz',
    'mlp3': 'data/mlp3.npz',
    'mlp9': 'data/mlp9.npz',
    'CIFAR10': 'data/CIFAR10',
    'CIFAR100': 'data/CIFAR100',
    'MNIST': 'data/MNIST',
    'ImageNet32': 'data/ImageNet32'
}


def get_dataset(dataset_name, train_samples, test_samples, batch_size, random_labels=False):
  im_size, padded_im_size, num_classes, in_channels = dataset_stats(dataset_name)
  target_trans = (lambda y: torch.randint(0, num_classes, (1,)).item()) if random_labels else None
  if dataset_name == 'MNIST':
    transform = transforms.Compose([transforms.Pad((padded_im_size - im_size)//2),
                                  transforms.ToTensor(),
                                  transforms.Normalize(0.1307,0.3081)])

    train_set = datasets.MNIST(custom_data_path['MNIST'], train=True, download=True, transform=transform, target_transform=target_trans)
    if train_samples:
      train_set = torch.utils.data.Subset(train_set, range(train_samples))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = datasets.MNIST(custom_data_path['MNIST'], train=False, download=True, transform=transform, target_transform=target_trans)
    if test_samples:
      test_set = torch.utils.data.Subset(test_set, range(test_samples))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    in_channels = 1
    num_classes = 10

  if dataset_name == 'CIFAR10':
    transform = transforms.Compose(
    [transforms.Pad((padded_im_size - im_size)//2),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = datasets.CIFAR10(root=custom_data_path['CIFAR10'], train=True, download=True, transform=transform, target_transform=target_trans)
    if train_samples:
      train_set = torch.utils.data.Subset(train_set, range(train_samples))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = datasets.CIFAR10(root=custom_data_path['CIFAR10'], train=False, download=True, transform=transform, target_transform=target_trans)
    if test_samples:
      test_set = torch.utils.data.Subset(test_set, range(test_samples))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    in_channels = 3
    num_classes = 10


  if dataset_name == 'CIFAR100':
    transform = transforms.Compose(
    [transforms.Pad((padded_im_size - im_size)//2),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = datasets.CIFAR100(root=custom_data_path['CIFAR100'], train=True, download=True, transform=transform, target_transform=target_trans)
    if train_samples:
      train_set = torch.utils.data.Subset(train_set, range(train_samples))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)

    test_set = datasets.CIFAR100(root=custom_data_path['CIFAR100'], train=False, download=True, transform=transform, target_transform=target_trans)
    if test_samples:
      test_set = torch.utils.data.Subset(test_set, range(test_samples))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8)
    in_channels = 3
    num_classes = 100

  if dataset_name == 'ImageNet32':
    transform = transforms.Compose(
      [transforms.Pad((padded_im_size - im_size)//2),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_set = ImageNet32(root=custom_data_path['ImageNet32'], train=True, transform=transform, target_transform=target_trans)
    test_set = ImageNet32(root=custom_data_path['ImageNet32'], train=False, transform=transform, target_transform=target_trans)
    if train_samples:
      train_set = torch.utils.data.Subset(train_set, range(train_samples))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    if test_samples:
      test_set = torch.utils.data.Subset(test_set, range(test_samples))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8)
    in_channels = 3
    num_classes = 1000

  if dataset_name == 'conic' or dataset_name[:3] == 'mlp':
    print(f'Using custom dataset at {custom_data_path[dataset_name]}')
    npz = np.load(custom_data_path[dataset_name])
    X, y = npz['X'], npz['y']
    if random_labels:
      y = torch.randint(0, num_classes, y.shape)

    tensor_X, tensor_y = torch.tensor(X).float(), torch.tensor(y).to(torch.int64)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tensor_X[:train_samples, :], tensor_y[:train_samples]),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tensor_X[train_samples: train_samples + test_samples, :], tensor_y[train_samples: train_samples + test_samples]),
        batch_size=batch_size, shuffle=True)

    in_channels = None

  return train_loader, test_loader, num_classes, in_channels

def dataset_stats(dataset):
    if dataset == 'MNIST':
      # dataset parameters
      im_size             = 28
      padded_im_size      = 32
      num_classes         = 10
      input_ch            = 1

    elif dataset == 'CIFAR10':
      # dataset parameters
      im_size             = 32
      padded_im_size      = 32
      num_classes         = 10
      input_ch            = 3

    elif dataset == 'CIFAR100':
      # dataset parameters
      im_size             = 32
      padded_im_size      = 32
      num_classes         = 100
      input_ch            = 3

    elif dataset == 'ImageNet32':
      # dataset parameters
      im_size             = 32
      padded_im_size      = 32
      num_classes         = 1000
      input_ch            = 3

    elif dataset == 'conic' or dataset.startswith('mlp'):
      # dataset parameters
      im_size             = None
      padded_im_size      = None
      num_classes         = 4
      input_ch            = None

    else:
        raise ValueError('Dataset not recognized.')

    return im_size, padded_im_size, num_classes, input_ch
