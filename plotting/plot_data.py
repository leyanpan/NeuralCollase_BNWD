import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

from plot_utils import process_file, filename_to_dict, plot_data

matplotlib.use('Agg')

dir_name = '../New_Results'
debug = False

fig, axes = plt.subplots(3, 5, figsize=(15, 9))

# Define fixed_params, axis_param, and random_avg_param
fixed_params = {"bn": False, "dataset": "CIFAR10", "lr": 1e-2, "epochs": 200, "train_samples": None, "test_samples": None, "model_type": "vgg11"}
axis_param = 'weight_decay'
random_avg_param = 'rand_seed'

# # Call plot_data for each subplot
# plot_data(dir_name, axes[0, 0], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params['bn'] = True
# plot_data(dir_name, axes[0, 1], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params = {"bn": False, "dataset": "CIFAR10", "lr": 1e-2, "epochs": 200, "train_samples": None, "test_samples": None, "model_type": "vgg19"}
# plot_data(dir_name, axes[0, 2], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params['bn'] = True
# plot_data(dir_name, axes[0, 3], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params = {'bn': False, 'dataset': 'CIFAR10', 'epochs': 100.0, 'lr': 0.001, 'model_type': 'ResNet'}
# plot_data(dir_name, axes[0, 4], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params = {"bn": False, "dataset": "MNIST", "lr": 1e-2, "epochs": 200, "train_samples": None, "test_samples": None, "model_type": "vgg11"}
# plot_data(dir_name, axes[1, 0], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params['bn'] = True
# plot_data(dir_name, axes[1, 1], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params = {"bn": False, "dataset": "MNIST", "lr": 1e-2, "epochs": 200, "train_samples": None, "test_samples": None, "model_type": "vgg19"}
# plot_data(dir_name, axes[1, 2], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params['bn'] = True
# plot_data(dir_name, axes[1, 3], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params = {"bn": False, "dataset": "MNIST", "lr": 1e-3, "epochs": 100, "model_type": "ResNet"}
# plot_data(dir_name, axes[1, 4], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params = {"bn": False, "dataset": "CIFAR100", "lr": 1e-2, "epochs": 200, "train_samples": None, "test_samples": None, "model_type": "vgg11"}
# plot_data(dir_name, axes[2, 0], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params['bn'] = True
# plot_data(dir_name, axes[2, 1], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params = {"bn": False, "dataset": "CIFAR100", "lr": 1e-2, "epochs": 200, "train_samples": None, "test_samples": None, "model_type": "vgg19"}
# plot_data(dir_name, axes[2, 2], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params['bn'] = True
# plot_data(dir_name, axes[2, 3], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params = {"bn": False, "dataset": "CIFAR100", "lr": 1e-2, "epochs": 200, "train_samples": None, "test_samples": None, "model_type": "ResNet"}
# plot_data(dir_name, axes[2, 4], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# # Set column labels
# axes[0, 0].set_title('VGG11 (no BN)')
# axes[0, 1].set_title('VGG11 (Contains BN)')
# axes[0, 2].set_title('VGG19 (no BN)')
# axes[0, 3].set_title('VGG19 (Contains BN)')
# axes[0, 4].set_title('ResNet (Contains BN)')

# # Set row labels
# axes[0, 0].set_ylabel('CIFAR10')
# axes[1, 0].set_ylabel('MNIST')
# axes[2, 0].set_ylabel('CIFAR100')

# axes[2, 0].set_xlabel('weight_decay')
# axes[2, 1].set_xlabel('weight_decay')
# axes[2, 2].set_xlabel('weight_decay')
# axes[2, 3].set_xlabel('weight_decay')
# axes[2, 4].set_xlabel('weight_decay')


# # Adjust spacing
# fig.tight_layout(rect=[0, 0, 1, 0.95])

# # Save the plot
# plt.savefig("plot_CV.png")

fig, axes = plt.subplots(2, 4, figsize=(12,6))

# Define fixed_params, axis_param, and random_avg_param
fixed_params = {"bn": False, "dataset": "CIFAR10", "lr": 1e-4, "epochs": 300, "model_type": "vgg11"}
axis_param = 'weight_decay'
random_avg_param = 'rand_seed'

# Call plot_data for each subplot
plot_data(dir_name, axes[0, 0], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True, min_axis_param=0.0002)
plot_data(dir_name, axes[0, 1], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, min_axis_param=0.0002)
fixed_params['bn'] = True
plot_data(dir_name, axes[0, 0], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True, min_axis_param=0.0002)
plot_data(dir_name, axes[0, 1], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, min_axis_param=0.0002)
fixed_params = {'bn': False, 'dataset': 'CIFAR10', 'epochs': 300, 'lr': 1e-4, 'model_type': 'vgg19'}
plot_data(dir_name, axes[0, 2], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True, min_axis_param=0.0002)
plot_data(dir_name, axes[0, 3], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, min_axis_param=0.0002)
fixed_params['bn'] = True
plot_data(dir_name, axes[0, 2], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True, min_axis_param=0.0002)
plot_data(dir_name, axes[0, 3], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, min_axis_param=0.0002)
fixed_params = {"bn": False, "dataset": "CIFAR100", 'lr': 1e-4, "epochs": 300, "model_type": "vgg11"}
plot_data(dir_name, axes[1, 0], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True, min_axis_param=0.0002)
plot_data(dir_name, axes[1, 1], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, min_axis_param=0.0002)
fixed_params['bn'] = True
plot_data(dir_name, axes[1, 0], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True, min_axis_param=0.0002)
plot_data(dir_name, axes[1, 1], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, min_axis_param=0.0002)
fixed_params = {"bn": False, "dataset": "CIFAR100", "lr": 1e-4, "epochs": 300, "model_type": "vgg19"}
plot_data(dir_name, axes[1, 2], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True, min_axis_param=0.0002)
plot_data(dir_name, axes[1, 3], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, min_axis_param=0.0002)
fixed_params['bn'] = True
plot_data(dir_name, axes[1, 2], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True, min_axis_param=0.0002)
plot_data(dir_name, axes[1, 3], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, min_axis_param=0.0002)
# Set column labels
axes[0, 0].set_title('VGG11 Avg class')
axes[0, 1].set_title('VGG11 Worst class')
axes[0, 2].set_title('VGG19 Avg class')
axes[0, 3].set_title('VGG19 Worst class')

# Set row labels
axes[0, 0].set_ylabel('CIFAR10')
axes[1, 0].set_ylabel('CIFAR100')

axes[1, 0].set_xlabel('weight_decay')
axes[1, 1].set_xlabel('weight_decay')
axes[1, 2].set_xlabel('weight_decay')
axes[1, 3].set_xlabel('weight_decay')


# Adjust spacing
fig.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot
plt.savefig("plot_CV_2x3.png")

fig, axes = plt.subplots(3, 6, figsize=(16, 8))
# Set column labels
axes[0, 0].set_title('3-layer MLP without BN')
axes[0, 1].set_title('3-layer MLP with BN')
axes[0, 2].set_title('6-layer MLP without BN')
axes[0, 3].set_title('6-layer MLP with BN')
axes[0, 4].set_title('9-layer MLP without BN')
axes[0, 5].set_title('9-layer MLP with BN')

# Set row labels
axes[0, 0].set_ylabel('conic hull dataset')
axes[1, 0].set_ylabel('MLP3 dataset')
axes[2, 0].set_ylabel('MLP6 dataset')

axes[2, 0].set_xlabel('weight_decay')
axes[2, 1].set_xlabel('weight_decay')
axes[2, 2].set_xlabel('weight_decay')
axes[2, 3].set_xlabel('weight_decay')
axes[2, 4].set_xlabel('weight_decay')
axes[2, 5].set_xlabel('weight_decay')


# # Define fixed_params, axis_param, and random_avg_param

# axis_param = 'weight_decay'
# random_avg_param = 'rand_seed'

# # Call plot_data for each subplot
# fixed_params = {"dataset": "conic", "model_type" : "MLP", "lr": 1e-3, "epochs": 300, 'model_depth_MLP': 4, 'bn': False}
# plot_data(dir_name, axes[0, 0], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params['bn'] = True
# plot_data(dir_name, axes[0, 1], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params = {"dataset": "conic", "model_type" : "MLP", "lr": 1e-3, "epochs": 300, 'model_depth_MLP': 6, 'bn': False}
# plot_data(dir_name, axes[0, 2], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params['bn'] = True
# plot_data(dir_name, axes[0, 3], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)

# fixed_params = {"dataset": "conic", "model_type" : "MLP", "lr": 1e-3, "epochs": 300, 'model_depth_MLP': 9, 'bn': False}
# plot_data(dir_name, axes[0, 4], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# # There's some problems with naming here
# fixed_params['bn'] = True
# fixed_params['epochs'] = 100
# del fixed_params['model_type']
# print(fixed_params)
# plot_data(dir_name, axes[0, 5], fixed_params, axis_param, random_avg_param)

# # Call plot_data for each subplot
# fixed_params = {"dataset": "mlp3", "model_type" : "MLP", "lr": 1e-3, "epochs": 100, 'model_depth_MLP': 3, 'bn': False}
# plot_data(dir_name, axes[1, 0], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params['bn'] = True
# plot_data(dir_name, axes[1, 1], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params = {"dataset": "mlp3", "model_type" : "MLP", "lr": 1e-3, "epochs": 100, 'model_depth_MLP': 6, 'bn': False}
# plot_data(dir_name, axes[1, 2], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params['bn'] = True
# plot_data(dir_name, axes[1, 3], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params = {"dataset": "mlp3", "model_type" : "MLP", "lr": 1e-3, "epochs": 100, 'model_depth_MLP': 9, 'bn': False}
# plot_data(dir_name, axes[1, 4], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params['bn'] = True
# plot_data(dir_name, axes[1, 5], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)

# # Call plot_data for each subplot
# fixed_params = {"dataset": "mlp6", "model_type" : "MLP", "lr": 1e-3, "epochs": 100, 'model_depth_MLP': 3, 'bn': False}
# plot_data(dir_name, axes[2, 0], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params['bn'] = True
# plot_data(dir_name, axes[2, 1], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params = {"dataset": "mlp6", "model_type" : "MLP", "lr": 1e-3, "epochs": 100, 'model_depth_MLP': 6, 'bn': False}
# plot_data(dir_name, axes[2, 2], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params['bn'] = True
# plot_data(dir_name, axes[2, 3], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params = {"dataset": "mlp6", "model_type" : "MLP", "lr": 1e-3, "epochs": 100, 'model_depth_MLP': 9, 'bn': False}
# plot_data(dir_name, axes[2, 4], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
# fixed_params['bn'] = True
# plot_data(dir_name, axes[2, 5], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)

# plt.savefig("plot_MLP.png")

# Code for small main content plots

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Set column labels

axes[0, 0].set_title('4-layer MLP, Average')
axes[0, 1].set_title('4-layer MLP, Worst Class')
axes[0, 2].set_title('6-layer MLP, Average')
axes[0, 3].set_title('6-layer MLP, Worst Class')

# Set row labels
axes[0, 0].set_ylabel('conic hull dataset')
axes[1, 0].set_ylabel('MLP3 dataset')

# Set x-axis labels
for i in range(4):
    axes[1, i].set_xlabel('weight_decay')

# Define fixed_params, axis_param, and random_avg_param
axis_param = 'weight_decay'
random_avg_param = 'rand_seed'

# Call plot_data for each subplot for the conic hull dataset
fixed_params = {"dataset": "conic", "model_type": "MLP", "lr": 1e-3, "epochs": 300, 'model_depth_MLP': 4, 'bn': False}
plot_data(dir_name, axes[0, 0], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True)
plot_data(dir_name, axes[0, 1], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
fixed_params['bn'] = True
plot_data(dir_name, axes[0, 0], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True)
plot_data(dir_name, axes[0, 1], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
fixed_params['model_depth_MLP'] = 6
fixed_params['bn'] = False
plot_data(dir_name, axes[0, 2], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True)
plot_data(dir_name, axes[0, 3], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
fixed_params['bn'] = True
plot_data(dir_name, axes[0, 2], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True)
plot_data(dir_name, axes[0, 3], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)

# Call plot_data for each subplot for the MLP3 dataset
fixed_params = {"dataset": "mlp3", "model_type": "MLP", "lr": 1e-3, "epochs": 300, 'model_depth_MLP': 4, 'bn': False}
plot_data(dir_name, axes[1, 0], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True)
plot_data(dir_name, axes[1, 1], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
fixed_params['bn'] = True
plot_data(dir_name, axes[1, 0], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True)
plot_data(dir_name, axes[1, 1], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
fixed_params['model_depth_MLP'] = 6
fixed_params['bn'] = False
plot_data(dir_name, axes[1, 2], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True)
plot_data(dir_name, axes[1, 3], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)
fixed_params['bn'] = True
plot_data(dir_name, axes[1, 2], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True)
plot_data(dir_name, axes[1, 3], fixed_params, axis_param, random_avg_param, max_axis_param=0.01)

plt.savefig("plot_MLP_2x4.png")

fig, axes = plt.subplots(2, 4, figsize=(16, 5))
axes[0,0].set_title('NC vs $|\gamma|$ value')
axes[1, 0].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')
axes[1, 1].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')
axes[1, 2].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')
axes[1, 3].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')

axes[0, 0].set_ylabel('cos similarity')
axes[1, 0].set_ylabel('cos similarity')
axis_param = 'fix_const'
random_avg_param = 'rand_seed'

dir_name = '../fix_const'
fixed_params = {"bn": True, "weight_decay": 5e-3, "dataset": "mlp3", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 4}
plot_data(dir_name, axes[0, 0], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 0], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 0].set_title('MLP3 dataset, 4-layer')
fixed_params = {"bn": True, "weight_decay": 5e-3, "dataset": "mlp3", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 6}
plot_data(dir_name, axes[0, 1], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 1], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 1].set_title('MLP3 dataset, 6-layer')
fixed_params = {"bn": True, "weight_decay": 5e-3, "dataset": "conic", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 4}
plot_data(dir_name, axes[0, 2], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 2], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 2].set_title('conic dataset, 4-layer')
fixed_params = {"bn": True, "weight_decay": 5e-3, "dataset": "conic", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 6}
plot_data(dir_name, axes[0, 3], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 3], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 3].set_title('conic dataset, 6-layer')
plt.savefig("plot_fix_const_wd_5e-3.png")

fig, axes = plt.subplots(2, 4, figsize=(16, 5))
fig.suptitle('NC vs $|\gamma|$ value (wd=0.0001)')
axes[1, 0].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')
axes[1, 1].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')
axes[1, 2].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')
axes[1, 3].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')

axes[0, 0].set_ylabel('cos similarity')
axes[1, 0].set_ylabel('cos similarity')
axis_param = 'fix_const'
random_avg_param = 'rand_seed'

dir_name = '../fix_const'
fixed_params = {"bn": True, "weight_decay": 1e-4, "dataset": "mlp3", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 4}
plot_data(dir_name, axes[0, 0], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 0], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 0].set_title('MLP3 dataset, 4-layer')
fixed_params = {"bn": True, "weight_decay": 1e-4, "dataset": "mlp3", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 6}
plot_data(dir_name, axes[0, 1], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 1], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 1].set_title('MLP3 dataset, 6-layer')
fixed_params = {"bn": True, "weight_decay": 1e-4, "dataset": "conic", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 4}
plot_data(dir_name, axes[0, 2], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 2], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 2].set_title('conic dataset, 4-layer')
fixed_params = {"bn": True, "weight_decay": 1e-4, "dataset": "conic", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 6}
plot_data(dir_name, axes[0, 3], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 3], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 3].set_title('conic dataset, 6-layer')
plt.savefig("plot_fix_const_wd_1e-4.png")

fig, axes = plt.subplots(2, 4, figsize=(16, 5))
fig.suptitle('NC vs $|\gamma|$ value (wd=0.0005)')
axes[1, 0].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')
axes[1, 1].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')
axes[1, 2].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')
axes[1, 3].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')

axes[0, 0].set_ylabel('cos similarity')
axes[1, 0].set_ylabel('cos similarity')
axis_param = 'fix_const'
random_avg_param = 'rand_seed'

dir_name = '../fix_const'
fixed_params = {"bn": True, "weight_decay": 5e-4, "dataset": "mlp3", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 4}
plot_data(dir_name, axes[0, 0], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 0], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 0].set_title('MLP3 dataset, 4-layer')
fixed_params = {"bn": True, "weight_decay": 5e-4, "dataset": "mlp3", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 6}
plot_data(dir_name, axes[0, 1], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 1], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 1].set_title('MLP3 dataset, 6-layer')
fixed_params = {"bn": True, "weight_decay": 5e-4, "dataset": "conic", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 4}
plot_data(dir_name, axes[0, 2], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 2], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 2].set_title('conic dataset, 4-layer')
fixed_params = {"bn": True, "weight_decay": 5e-4, "dataset": "conic", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 6}
plot_data(dir_name, axes[0, 3], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 3], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 3].set_title('conic dataset, 6-layer')
plt.savefig("plot_fix_const_wd_5e-4.png")

fig, axes = plt.subplots(2, 4, figsize=(16, 5))
fig.suptitle('NC vs $|\gamma|$ value (wd=0.001)')
axes[1, 0].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')
axes[1, 1].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')
axes[1, 2].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')
axes[1, 3].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')

axes[0, 0].set_ylabel('cos similarity')
axes[1, 0].set_ylabel('cos similarity')
axis_param = 'fix_const'
random_avg_param = 'rand_seed'
fixed_params = {"bn": True, "weight_decay": 1e-3, "dataset": "mlp3", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 4}
plot_data(dir_name, axes[0, 0], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 0], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 0].set_title('MLP3 dataset, 4-layer')
fixed_params = {"bn": True, "weight_decay": 1e-3, "dataset": "mlp3", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 6}
plot_data(dir_name, axes[0, 1], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 1], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 1].set_title('MLP3 dataset, 6-layer')
fixed_params = {"bn": True, "weight_decay": 1e-3, "dataset": "conic", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 4}
plot_data(dir_name, axes[0, 2], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 2], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 2].set_title('conic dataset, 4-layer')
fixed_params = {"bn": True, "weight_decay": 1e-3, "dataset": "conic", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 6}
plot_data(dir_name, axes[0, 3], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 3], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 3].set_title('conic dataset, 6-layer')
plt.savefig("plot_fix_const_wd_1e-3.png")

fig, axes = plt.subplots(2, 4, figsize=(16, 5))
fig.suptitle('NC vs $|\gamma|$ value (wd=0.007)')
axes[1, 0].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')
axes[1, 1].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')
axes[1, 2].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')
axes[1, 3].set_xlabel('Quadratic Avg of Last Layer Feature Norm $|\gamma|$')

axes[0, 0].set_ylabel('cos similarity')
axes[1, 0].set_ylabel('cos similarity')
axis_param = 'fix_const'
random_avg_param = 'rand_seed'
fixed_params = {"bn": True, "weight_decay": 7e-3, "dataset": "mlp3", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 4}
plot_data(dir_name, axes[0, 0], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 0], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 0].set_title('MLP3 dataset, 4-layer')
fixed_params = {"bn": True, "weight_decay": 7e-3, "dataset": "mlp3", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 6}
plot_data(dir_name, axes[0, 1], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 1], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 1].set_title('MLP3 dataset, 6-layer')
fixed_params = {"bn": True, "weight_decay": 7e-3, "dataset": "conic", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 4}
plot_data(dir_name, axes[0, 2], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 2], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 2].set_title('conic dataset, 4-layer')
fixed_params = {"bn": True, "weight_decay": 7e-3, "dataset": "conic", "lr": 1e-3, "epochs": 300, "model_type": "MLP", "fix_bn": True, "model_depth_MLP": 6}
plot_data(dir_name, axes[0, 3], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 3], fixed_params, axis_param, random_avg_param, avg=True, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=-0.2)
axes[0, 3].set_title('conic dataset, 6-layer')
plt.savefig("plot_fix_const_wd_7e-3.png")

fig, axes = plt.subplots(2, 4, figsize=(16, 5))
fig.suptitle('NC vs $|\gamma|$ value (wd=5e-4)')
axes[1, 0].set_xlabel('Norm of BN weight vector $|\gamma|$')
axes[1, 1].set_xlabel('Norm of BN weight vector $|\gamma|$')
axes[1, 2].set_xlabel('Norm of BN weight vector $|\gamma|$')
axes[1, 3].set_xlabel('Norm of BN weight vector $|\gamma|$')

axes[0, 0].set_ylabel('cos similarity')
axes[1, 0].set_ylabel('cos similarity')
axis_param = 'fix_const'
random_avg_param = 'rand_seed'
fixed_params = {"bn": True, "weight_decay": 5e-4, "dataset": "CIFAR10", "lr": 1e-4, "epochs": 300, "model_type": "vgg11", "fix_bn": True}
plot_data(dir_name, axes[0, 0], fixed_params, axis_param, random_avg_param, avg=False, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 0], fixed_params, axis_param, random_avg_param, avg=False, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=0.5)
axes[0, 0].set_title('CIFAR10, VGG11')
fixed_params = {"bn": True, "weight_decay": 5e-4, "dataset": "CIFAR10", "lr": 1e-4, "epochs": 300, "model_type": "vgg19", "fix_bn": True}
plot_data(dir_name, axes[0, 1], fixed_params, axis_param, random_avg_param, avg=False, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 1], fixed_params, axis_param, random_avg_param, avg=False, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=0.5)
axes[0, 1].set_title('CIFAR10, VGG19')
fixed_params = {"bn": True, "weight_decay": 5e-4, "dataset": "CIFAR100", "lr": 1e-4, "epochs": 300, "model_type": "vgg11", "fix_bn": True}
plot_data(dir_name, axes[0, 2], fixed_params, axis_param, random_avg_param, avg=False, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 2], fixed_params, axis_param, random_avg_param, avg=False, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=0.5)
axes[0, 2].set_title('CIFAR100, VGG11')
fixed_params = {"bn": True, "weight_decay": 5e-4, "dataset": "CIFAR100", "lr": 1e-4, "epochs": 300, "model_type": "vgg19", "fix_bn": True}
plot_data(dir_name, axes[0, 3], fixed_params, axis_param, random_avg_param, avg=False, axis_param_factor=math.sqrt(200), min_axis_param=0.02, inter=False, y_min=0.7, y_max=1)
plot_data(dir_name, axes[1, 3], fixed_params, axis_param, random_avg_param, avg=False, axis_param_factor=math.sqrt(200), min_axis_param=0.02, intra=False, y_min=-0.35, y_max=0.5)
axes[0, 3].set_title('CIFAR100, VGG19')
plt.savefig("plot_fix_const_wd_vgg_5e-4.png")