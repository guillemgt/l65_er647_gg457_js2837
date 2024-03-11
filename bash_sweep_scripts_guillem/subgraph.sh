#!/bin/bash

conda activate L65

# Define configs and seeds
configs=("configs/Experiments/peptides-func.yaml" "configs/Experiments/peptides-struct.yaml" "configs/Experiments/pascal-voc.yaml")
seeds=(0)

# Full path to application executable
application="python"

# Work directory
workdir=$(pwd) # Sets workdir to the directory from which the script is run

# Move to work directory
cd $workdir
echo -e "Changed directory to $(pwd).\n"


# Loop through configs and seeds
for config in "${configs[@]}"; do
    for seed in "${seeds[@]}"; do
        # Define run options for the application
        options="main.py --cfg $config seed $seed  wandb.project Experiments_round_1 wandb.entity l65 device cuda gt.layer_type CustomGatedGCN+Subgraph_Mamba_L65 gt.mamba_heuristics global_pe_1"

        # Build the command to execute
        CMD="$application $options"
                
        # Execute the command in the background
        eval $CMD
    done
done

echo "All jobs completed."