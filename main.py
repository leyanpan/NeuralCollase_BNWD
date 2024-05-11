import argparse
import json
import os

import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
# import torchvision.models as models
from tqdm import tqdm
from collections import OrderedDict
# from scipy.sparse.linalg import svds
# from torchvision import datasets, transforms
from imagenet32_dataset import ImageNet32

from utils import combine_dicts, dict_to_file_string
from torch_utils import hook, hook_group, features
from train import run_expt
from dataloader import get_dataset
from models import get_model
from train import train
from analysis import cos_analysis, cos_analysis_str


def setup_parser():
    parser = argparse.ArgumentParser(description="Setup for training a neural network model.")

    # Options file
    parser.add_argument('options_file', type=str, help='Path to options file in JSON format to specify experiment configurations.')

    # Path options
    parser.add_argument('--model_save_path', type=str, default='models', help='Path to save the model.')
    parser.add_argument('--data_save_path', type=str, default='data', help='Path to save the data.')
    parser.add_argument('--figure_save_path', type=str, default='figures', help='Path to save figures.')

    # Model hyperparameters
    parser.add_argument('--model_type', type=str, default='MLP', choices=['MLP', 'ResNet', 'CNN'], help='Type of model to use.')
    parser.add_argument('--dataset', type=str, default='mlp6', help='Dataset to use.')
    parser.add_argument('--train_samples', type=int, default=None, help='Number of training samples.')
    parser.add_argument('--run_test', action='store_true', help='Run tests after training.')
    parser.add_argument('--test_samples', type=int, default=2000, help='Number of test samples.')

    # Network structure (MLP specific)
    parser.add_argument('--model_depth_MLP', type=int, default=6, help='Depth of the MLP model.')
    parser.add_argument('--hidden_layer_width', type=int, default=300, help='Width of hidden layers in MLP.')
    parser.add_argument('--bn', action='store_true', help='Use Batch Normalization.')
    parser.add_argument('--bn_affine', action='store_true', help='Batch Normalization affine parameter.')
    parser.add_argument('--bn_eps', type=float, default=1e-5, help='Batch Normalization epsilon value.')
    parser.add_argument('--linear_bias', action='store_true', help='Use bias in linear layers.')
    parser.add_argument('--weight_norm', action='store_true', help='Normalize weights after each iteration.')
    parser.add_argument('--inst_norm', action='store_true', help='Use Instance Normalization.')
    parser.add_argument('--layer_norm', action='store_true', help='Use Layer Normalization.')
    parser.add_argument('--nc_train', action='store_true', help='Train with NC measures optimization.')
    parser.add_argument('--nc_coeff', type=float, default=0.1, help='Coefficient for NC measure optimization.')
    parser.add_argument('--no_batch', action='store_true', help='Use full dataset per iteration instead of batching.')
    parser.add_argument('--activation_MLP', type=str, default='ReLU', choices=['ReLU', 'Sigmoid', 'Tanh'], help='Activation function for MLP.')
    parser.add_argument('--lamb', type=float, default=1, help='Coefficient of cosine similarity for NC training.')

    # Synthetic Data Specific
    parser.add_argument('--input_dim', type=int, default=16, help='Input dimension for conic data.')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes in the dataset.')

    # Optimization hyperparameters
    parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay.')
    parser.add_argument('--lr', type=float, default=0.0679, help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for optimizer.')
    parser.add_argument('--rand_seed', type=int, default=12138, help='Random seed.')
    parser.add_argument('--loss_name', type=str, default='CrossEntropyLoss', choices=['CrossEntropyLoss', 'MSELoss'], help='Loss function to use.')

    # Data augmentation
    parser.add_argument('--random_labels', action='store_true', help='Use random labels for the dataset.')

    # Others
    parser.add_argument('--save_model', action='store_true', help='Whether to save the trained models.')

    return parser


def run_expt(args, options_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, num_classes, in_channels = get_dataset(args.dataset, args.train_samples, args.test_samples, args.random_labels)

    # Save files to param dictionary
    dict_str = dict_to_file_string(options_dict)
    model_fn = os.path.join(args.model_save_path, f"{dict_str}.pth.tar")
    data_fn = os.path.join(args.data_save_path, f"{dict_str}.txt")
    figure_fn = os.path.join(args.figure_save_path, f"{dict_str}.png")

    model, classifier, hooked_modules = get_model(args.model_type, num_classes, in_channels)
    model.register_forward_hook(hook)

    optimizer = optim.Adam(model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay)

    epochs_lr_decay = [int(args.epochs * 0.25), int(args.epochs * 0.5), int(args.epochs * 0.75)]

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                            milestones=epochs_lr_decay,
                                            gamma=args.lr_decay)

    if args.loss_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
        criterion_summed = nn.CrossEntropyLoss(reduction='sum')
    elif args.loss_name == 'MSELoss':
        criterion = nn.MSELoss()
        riterion_summed = nn.MSELoss(reduction='sum')

    for epoch in range(1, args.epochs + 1):
        train(model, criterion, device, num_classes, train_loader, optimizer, epoch)
        lr_scheduler.step()

    loss, intra_cos, inter_cos, avg_intra, avg_inter, qmean_norms, bn_norms, weight_norms, nccs, ranks = cos_analysis(model, hooked_modules, train_loader, num_classes, criterion_summed=criterion_summed)
    analysis_str = cos_analysis_str(loss, intra_cos, inter_cos, avg_intra, avg_inter, qmean_norms, bn_norms, weight_norms, nccs, ranks, hooked_modules)

    with open(data_fn, 'w') as f:
        f.write(analysis_str)

    if args.save_model:
        torch.save(model, model_fn)


def main(args):
    """
        Wrapper function for running batched experiments according to option file specifications
    """
    # Load options file
    with open(args.options_file, 'r') as f:
        options = json.load(f)

    for options_dict in combine_dicts(options[0], options[1]):
        args_copy = vars(args).copy()
        args_copy.update(options_dict)
        run_expt(args_copy, options_dict)

if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    main(args)
