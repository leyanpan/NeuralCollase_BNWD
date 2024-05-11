import itertools

def combine_dicts(base_dict, enum_dict):
    # Initialize an empty list to store the result
    result = []
    # Get the keys and values from the enum_dict
    enum_keys = list(enum_dict.keys())
    enum_values = list(enum_dict.values())
    # Create a list of all combinations of enum_dict values
    enum_combinations = list(itertools.product(*enum_values))
    # Loop through the enum_dict combinations
    for enum_combination in enum_combinations:
        # Create a new dictionary for this combination
        new_dict = base_dict.copy()
        # Add the enum_dict values to the new dictionary
        for i, key in enumerate(enum_keys):
            new_dict[key] = enum_combination[i]
        # Add the new dictionary to the result list
        result.append(new_dict)
    return result


def dict_to_file_string(d):
  dict_str = ''
  first = True
  for key in sorted(d.keys()):
    if first:
      first = False
    else:
      dict_str += '_'
    dict_str += f'{key}_{d[key]}'
  return dict_str
