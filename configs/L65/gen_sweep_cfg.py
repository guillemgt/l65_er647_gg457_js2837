import argparse
import yaml
import sys
import os

def update_nested_dict(original_dict, updates):
    """
    Recursively update nested dictionaries.
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in original_dict:
            update_nested_dict(original_dict[key], value)
        else:
            original_dict[key] = value

# Parse command line arguments
parser = argparse.ArgumentParser(description='Update YAML configuration with command line arguments.')
parser.add_argument('sweep_id', help='The ID of the sweep for naming the output YAML folder.')
parser.add_argument('run_id', help='The ID of the run for naming the output YAML file.')

# Use `parse_known_args` to accept arbitrary command line arguments
args, unknown = parser.parse_known_args()

# Convert arbitrary arguments into a nested dictionary structure
additional_args = {}
for i in range(0, len(unknown), 2):
    keys = unknown[i].lstrip('-').split('.')
    value = unknown[i + 1]
    current_level = additional_args
    for key in keys[:-1]:
        if key not in current_level:
            current_level[key] = {}
        current_level = current_level[key]
    current_level[keys[-1]] = value

# Exclude sweep_id and run_id from additional_args to prevent them from being added to the YAML
if 'sweep_id' in additional_args:
    del additional_args['sweep_id']
if 'run_id' in additional_args:
    del additional_args['run_id']

base_dir = os.path.join(os.getcwd(), 'configs', 'L65')

# Load existing YAML configuration
try:
    with open(f'{base_dir}/default_cfg.yaml', 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print("Error: default_cfg.yaml not found.")
    sys.exit(1)

# Update configuration with additional arguments
update_nested_dict(config, additional_args)

# Save updated configuration to a new file
output_dir = os.path.join(base_dir, 'wandb_sweep_cfgs', args.sweep_id)
output_filename = os.path.join(output_dir, f"{args.run_id}.yaml")

# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)

with open(output_filename, 'w') as file:
    yaml.dump(config, file, sort_keys=False)

print(f"Configuration saved to {output_filename}.")
