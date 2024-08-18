import os
import numpy as np


def process_file(file_name, avg=False):
    intra_cos = []
    inter_cos = []
    loss = None
    nc_intra = None
    nc_inter = None

    with open(file_name, 'r') as f:
        for line in f:
            if line.startswith('Test Set:'):
                break
            elif line.startswith('Intra Cos: ') and not avg:
                intra_cos.append(float(line.split()[2]))
            elif line.startswith('Inter Cos: ') and not avg:
                inter_cos.append(float(line.split()[2]))
            elif line.startswith('Intra Avg: ') and avg == True:
                intra_cos.append(float(line.split()[2]))
            elif line.startswith('Inter Avg: ') and avg == True:
                inter_cos.append(float(line.split()[2]))
            elif line.startswith('Intra Delta ') and avg == 'delta':
                intra_cos.append(float(line.split()[3]))
            elif line.startswith('Inter Delta ') and avg == 'delta':
                inter_cos.append(float(line.split()[3]))
            elif line.startswith('Average Loss:'):
                loss = float(line.split()[2]) 
            elif line.startswith('Loss'):
                loss = float(line.split()[1])

    if len(intra_cos) >= 2:
        nc_intra = intra_cos[-2]
    if len(inter_cos) >= 2:
        nc_inter = inter_cos[-2]
    return intra_cos, inter_cos, nc_intra, nc_inter, loss


def filename_to_dict(file_name):
    # Initialize an empty dictionary
    result = {}
    raw_file_name = '.'.join(file_name.split('.')[:-1])
    # Split the file name by underscores to extract keys and values
    parts = raw_file_name.split('_')
    # Loop through the parts to determine which are keys and which are values
    i = 0
    key = ""
    while i < len(parts):
        part = parts[i]
        # Check if the part is a key
        if part == 'dataset':
            # The next part is a value
            i += 1
            value = parts[i]
            # Add the key-value pair to the result dictionary
            result[part] = value
            i += 1
            continue
        else:
            if key == "model_type":
                is_value = True
                value = part
            else:
                # The part is either a key or a value
                is_value = False
                # Check if the part is a number
                try:
                    float_part = float(part)
                    # If the part is a number, it must be a value
                    is_value = True
                    value = float_part
                except ValueError:
                    # The part is not a number, so it could be a key or a string value
                    if part.lower() == 'true':
                        # If the part is "True", it must be a value
                        is_value = True
                        value = True
                    elif part.lower() == 'false':
                        # If the part is "False", it must be a value
                        is_value = True
                        value = False
                    elif part.lower() == 'none':
                        # If the part is "False", it must be a value
                        is_value = True
                        value = None
                    elif any(c.isdigit() for c in part):
                        is_value = True
                        value = part
                    else:
                        # Otherwise, assume the part is a key and move on to the next part to check if it's a value
                        is_value = False
                        value = parts[i]
            # Add the key-value pair to the result dictionary
            if is_value:
                result[key] = value
                key = ""
            else:
                if key == "":
                    key = part
                else:
                    key += "_" + part
            i += 1
    return result



def plot_data(dir_name, ax, fixed_params, axis_param, random_avg_param, max_axis_param=None, avg=False, min_axis_param=None, axis_param_factor=None, y_min=-0.35, y_max=1, intra=True, inter=True, loss=True):
    # Initialize lists and dictionaries for storing values
    axis_values = []
    grouped_values = {}

    # Loop through all files in the folder
    for file_name in os.listdir(dir_name):
        # Only process files with .txt extension
        if not file_name.endswith('.txt'):
            continue
        # Get the dictionary from the file name
        full_name = os.path.join(dir_name, file_name)
        file_dict = filename_to_dict(file_name)

        # Only continue if the dictionary contains all same corresponding key-value entries as fixed_params
        if not all(file_dict.get(key) == value for key, value in fixed_params.items()):
            continue
        # Get the values for grouping
        axis_value = float(file_dict.get(axis_param))
        if max_axis_param and axis_value > max_axis_param or min_axis_param and axis_value < min_axis_param:
            continue
        random_avg_value = file_dict.get(random_avg_param)
        if axis_param_factor:
            axis_value *= axis_param_factor
        # Create a unique key based on axis_param and random_avg_param
        group_key = axis_value

        # Call the process_file function to get nc_intra and nc_inter
        intra_cos, inter_cos, nc_intra, nc_inter, loss = process_file(full_name, avg)

        # Add the values to the grouped_values dictionary
        if group_key not in grouped_values:
            grouped_values[group_key] = {'nc_intra': [], 'nc_inter': [], 'loss': []}

        grouped_values[group_key]['nc_intra'].append(nc_intra)
        grouped_values[group_key]['nc_inter'].append(nc_inter)
        grouped_values[group_key]['loss'].append(loss)

        # Keep note of the axis value
        if axis_value not in axis_values:
            axis_values.append(axis_value)
    # Compute averages and standard errors for each group
    averages = []
    std_errors = []
    print(grouped_values)
    for group_key, group_data in grouped_values.items():
        if group_data['nc_intra'][0] == None or group_data['nc_inter'][0] == None:
            continue
        nc_intra_avg = np.mean(group_data['nc_intra'])
        nc_inter_avg = np.mean(group_data['nc_inter'])
        loss_avg = np.mean(group_data['loss'])
        nc_intra_std_err = np.std(group_data['nc_intra']) / np.sqrt(len(group_data['nc_intra']))
        nc_inter_std_err = np.std(group_data['nc_inter']) / np.sqrt(len(group_data['nc_inter']))
        loss_std_err = np.std(group_data['loss']) / np.sqrt(len(group_data['loss']))

        averages.append((group_key, nc_intra_avg, nc_inter_avg, loss_avg))
        std_errors.append((group_key, nc_intra_std_err, nc_inter_std_err, loss_std_err))

    # Sort averages and std_errors based on axis values
    averages.sort(key=lambda x: x[0])
    std_errors.sort(key=lambda x: x[0])

    # Extract axis values, nc_intra_avg, and nc_inter_avg from averages
    axis_values = [x[0] for x in averages]
    nc_intra_avg_values = [x[1] for x in averages]
    nc_inter_avg_values = [x[2] for x in averages]
    loss_avg_values = [x[3] for x in averages]

    # Extract nc_intra_std_err and nc_inter_std_err from std_errors
    nc_intra_std_err_values = [x[1] for x in std_errors]
    nc_inter_std_err_values = [x[2] for x in std_errors]
    loss_std_err_values = [x[3] for x in std_errors]

    intra_label = 'intra_BN' if fixed_params['bn'] else 'intra_no_BN'
    inter_label = 'inter_BN' if fixed_params['bn'] else 'inter_no_BN'

    # Create a line plot with error bars
    if intra:
        ax.errorbar(axis_values, nc_intra_avg_values, yerr=nc_intra_std_err_values, label=intra_label, fmt='o-', capsize=5, markersize=3, color='green' if fixed_params['bn'] else 'blue')
    if inter:
        ax.errorbar(axis_values, nc_inter_avg_values, yerr=nc_inter_std_err_values, label=inter_label, fmt='-.', capsize=5, markersize=3, color='orange' if fixed_params['bn'] else 'red')
    ax.set_xscale('log')
    ax.set_ylim(y_min, y_max)
    ax.legend()