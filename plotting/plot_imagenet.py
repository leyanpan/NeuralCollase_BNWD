from plot_utils import plot_data
import matplotlib.pyplot as plt
import os

fig, axes = plt.subplots(1, 6, figsize=(18,4))


dir_name = 'results'

# Define fixed_params, axis_param, and random_avg_param
fixed_params = {"bn": True, "dataset": "ImageNet32", "model_type": "vgg11"}
axis_param = 'weight_decay'
random_avg_param = 'rand_seed'

# Call plot_data for each subplot
plot_data(dir_name, axes[0], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True, min_axis_param=0.0002)
plot_data(dir_name, axes[1], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg='delta', min_axis_param=0.0002)

fixed_params = {"bn": True, "dataset": "ImageNet32", "model_type": "vgg19"}
plot_data(dir_name, axes[2], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True, min_axis_param=0.0002)
plot_data(dir_name, axes[3], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg='delta', min_axis_param=0.0002)

fixed_params = {"bn": True, "dataset": "ImageNet32", "model_type": "ResNet"}
plot_data(dir_name, axes[4], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg=True, min_axis_param=0.0002)
plot_data(dir_name, axes[5], fixed_params, axis_param, random_avg_param, max_axis_param=0.01, avg='delta', min_axis_param=0.0002)

# Set column labels
axes[0].set_title('VGG11 Avg class')
axes[1].set_title('VGG11 Worst class')
axes[2].set_title('VGG19 Avg class')
axes[3].set_title('VGG19 Worst class')
axes[4].set_title('ResNet Avg class')
axes[5].set_title('ResNet Worst class')

# Set row labels
axes[0].set_ylabel('ImageNet32')

axes[0].set_xlabel('weight_decay')
axes[1].set_xlabel('weight_decay')
axes[2].set_xlabel('weight_decay')
axes[3].set_xlabel('weight_decay')


# Adjust spacing
fig.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot
plt.savefig("plot_ImageNet.png")
