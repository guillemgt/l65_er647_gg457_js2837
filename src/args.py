import argparse
from datetime import datetime
parser = argparse.ArgumentParser(description='State Space Graph Neural networks')


# Dataset
parser.add_argument('--dataset', type=str, default='ZINC', choices=['CSL', 'ZINC'])

# Graph Model 
parser.add_argument('--graph_model_type', type=str, default='mamba', choices=['mamba'])
parser.add_argument('--graph_model_channels', type=int, default=64)
parser.add_argument('--graph_model_pe_dim', type=int, default=8)
parser.add_argument('--graph_model_num_layers', type=int, default=10)
parser.add_argument('--graph_model_shuffle_ind', type=int, default=0)
parser.add_argument('--graph_model_d_conv', type=int, default=4)
parser.add_argument('--graph_model_d_state', type=int, default=16)
parser.add_argument('--graph_model_order_by_degree', action='store_true')

# Training Configuration
parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs for training.')
parser.add_argument('--max_steps', type=int, default=1000000, help='Maximum number of training steps (batches) for training.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
parser.add_argument('--optimizer', type=str, choices=['adam', 'adamw'], default='adamw')
parser.add_argument('--train_batch_size', type=int, default=128, help='Batch size for training.')
parser.add_argument('--validation_batch_size', type=int, default=128, help='Batch size for validation.')
parser.add_argument('--checkpoint_dir', type=str, default='./model_checkpoints/', help='Directory to save model checkpoints.')
parser.add_argument('--force_full_epoch_training', action='store_true', help='If True, then training will continue for the specified amount of epochs regardless.')
parser.add_argument('--lr_scheduler', type=str, choices=['plateau', 'cosine_warm_restart', 'linear', 'lambda'], default=None)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--class_weight', type=str, choices=['standard', 'balanced'], default='balanced', 
                    help="If `standard`, all classes use a weight of 1.\
                    If `balanced`, classes are weighted inverse proportionally to their size (see https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html)")
parser.add_argument('--gradient_clip_val', type=float, default=0.5, help='Gradient clipping value to prevent exploding gradients.')
parser.add_argument('--deterministic', action='store_true', dest='deterministic')

# General Configuration
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
parser.add_argument('--logging_interval', type=int, default=100, help='Interval for logging training metrics.')
parser.add_argument('--save_top_k', type=int, default=1, help='Select k-best model checkpoints to save for each run.')
parser.add_argument('--test_only', action='store_true', help='If True, will only run testing, no training or validation will be performed.')

parser.add_argument('--subset_argument', type=str, help='Gouped arg configuration.')

# Validation
parser.add_argument('--metric_model_selection', type=str, default='val_loss',
                    choices=['cross_entropy_loss', 'total_loss', 'balanced_accuracy', 'accuracy', 'lr-Adam', 'train_loss', 'train_loss_step', 'train_acc', 'train_acc_step', 'val_loss', 'val_acc'], help='Metric used for model selection.')
parser.add_argument('--patience_early_stopping', type=int, default=3,
                    help='Set number of checks (set by *val_check_interval*) to do early stopping. Minimum training duration: args.val_check_interval * args.patience_early_stopping epochs')
parser.add_argument('--val_check_interval', type=float, default=1.0, 
                    help='Number of steps at which to check the validation. If set to 1.0, will simply perform the default behaviour of an entire batch before validation.')
parser.add_argument('--train_on_full_data', action='store_true', dest='train_on_full_data', \
                    help='Train on the full data (train + validation), leaving only `--test_split` for testing.')
parser.add_argument('--overfit_batches', type=int, default=0, help='PyTorch Lightning trick to pick only N batches and iteratively overfit on them. Useful for debugging. Default set to 0, i.e. normal behaviour.')

# Weights & Biases (wandb) Integration
parser.add_argument('--wandb_entity_name', type=str, default='evangeorgerex')
parser.add_argument('--wandb_project_name', type=str, default='L65') #  
parser.add_argument('--wandb_run_name', type=str, default=f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
parser.add_argument('--wandb_log_freq', type=int, default=10)
parser.add_argument('--group', type=str, help="Group runs in wand")
parser.add_argument('--job_type', type=str, help="Job type for wand")
parser.add_argument('--notes', type=str, help="Notes for wandb logging.")
parser.add_argument('--tags', nargs='+', type=str, default=[], help='Tags for wandb')
parser.add_argument('--suffix_wand_run_name', type=str, default="", help="Suffix for run name in wand")
parser.add_argument('--wandb_log_model', action='store_true', dest='wandb_log_model', help='True for storing the model checkpoints in wandb')
parser.set_defaults(wandb_log_model=False)
parser.add_argument('--disable_wandb', action='store_true', dest='disable_wandb', help='True if you dont want to create wandb logs.')
parser.set_defaults(disable_wandb=False)

# Experiment set up
parser.add_argument('--hpc_run', action='store_true', dest='hpc_run', help='True for when running on HPC')

# Seeds

parser.add_argument('--seed_model_init', type=int, default=42, help='Seed for initializing the model (to have the same weights)')
parser.add_argument('--seed_training', type=int, default=42, help='Seed for training (e.g., batch ordering)')

parser.add_argument('--seed_kfold', type=int, help='Seed used for doing the kfold in train/test split')
parser.add_argument('--seed_validation', type=int, help='Seed used for selecting the validation split.')

def apply_subset_arguments(subset_args_str, args):

    # Proceed only if the string is not empty
    if subset_args_str and subset_args_str is not None:
        # Split the subset argument string into individual arguments
        # Trim the string to remove any leading/trailing whitespace
        subset_args_str = subset_args_str.strip()
        subset_args = subset_args_str.split()
        
        # Iterate over the subset arguments and update the args Namespace
        i = 0
        while i < len(subset_args):
            arg = subset_args[i]
            # Ensure that it starts with '--'
            if arg.startswith("--"):
                key = arg[2:]  # Remove '--' prefix to match the args keys
                value = subset_args[i + 1]
                # Update the args Namespace if the attribute exists
                if hasattr(args, key):
                    # Convert value to the right type based on the existing attribute
                    attr_type = type(getattr(args, key))
                    setattr(args, key, attr_type(value))
                i += 2  # Move to the next argument
            else:
                raise ValueError(f"Expected an argument starting with '--', found: {arg}")


