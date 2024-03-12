#!/bin/bash

# Define configs and seeds
configs=("configs/Experiments/peptides-func.yaml" "configs/Experiments/peptides-struct.yaml" "configs/Experiments/pascal-voc.yaml")
seeds=(3 4)

# Full path to application executable
application="python"

# Work directory
workdir=$(pwd) # Sets workdir to the directory from which the script is run

# Move to work directory
cd $workdir
echo -e "Changed directory to $(pwd).\n"

# Export the number of threads for OpenMP
export OMP_NUM_THREADS=1

# Loop through configs and seeds
for config in "${configs[@]}"; do
    for seed in "${seeds[@]}"; do
        # Define run options for the application
        options="main.py --cfg $config seed $seed wandb.project mamba_multi_pe1u_devgpu wandb.entity l65 device cuda gt.layer_type CustomGatedGCN+MultiMambaL65_10 gt.mamba_heuristics global_pe_1"


        # Build the command to execute
        CMD="$application $options"
        
        echo -e "Executing command in background:\n==================\n$CMD\n"
        
        # Execute the command in the background
        eval $CMD &
    done
    # Wait for all background jobs to finish
    wait
done

echo "All jobs completed."