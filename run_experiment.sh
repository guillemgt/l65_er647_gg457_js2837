#!/bin/bash

# Example Python script name: train_model.py

# Initialize an empty string for additional arguments
ARGS=""

# Extract the sweep_id and other parameters
for arg in "$@"
do
    if [[ "$arg" == "--sweep_id" ]]; then
        NEXT_IS_SWEEP_ID=true
    elif [[ $NEXT_IS_SWEEP_ID ]]; then
        SWEEP_ID="$arg"
        NEXT_IS_SWEEP_ID=false
    elif [[ "$arg" == *=* ]]; then  # Handle args like --key=value
        ARGS="$ARGS ${arg%%=*} ${arg#*=}"
    else
        ARGS="$ARGS $arg"
    fi
done

# Now, $SWEEP_ID holds the sweep_id, and $ARGS holds the rest of the arguments

RUN_ID=$(uuidgen)

# Example of using the sweep_id if needed, adjust based on your use case
echo "Running experiment $RUN_ID for sweep_id: $SWEEP_ID"

# Call the Python script with all arguments
python configs/L65/gen_sweep_cfg.py --sweep_id "$SWEEP_ID"  --run_id "$RUN_ID" $ARGS 
python main.py --cfg "configs/L65/wandb_sweep_cfgs/${SWEEP_ID}/${RUN_ID}.yaml"
