import logging
import os
import wandb
import pytorch_lightning as pl

from args import parser, apply_subset_arguments
from dataset_utils import get_datamodule
from trainer_utils import train_model
from wandb_utils import create_wandb_logger

import traceback
import sys

from torch.cuda import OutOfMemoryError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Run Experiment Pipeline for GraphMamba!')


def process_results(args):
    # If anything needs to be processed after training/testing, it should go here
    return

def main():
    # ================================
    # SET UP ARGS
    # ================================
    args = parser.parse_args()
    apply_subset_arguments(args.subset_argument, args)
    # ================================
    # CONFIGURE WANDB
    # ================================
    if args.disable_wandb:
        os.environ['WANDB_MODE'] = 'disabled'
    wandb.init(project=args.wandb_project_name, config=args)

    wandb_logger = create_wandb_logger(args)
    # wandb.run.name = f"{get_run_name(args)}_{args.suffix_wand_run_name}_{wandb.run.id}"
    wandb.run.name = args.wandb_run_name

    # ================================
    # FETCH DATASET
    # ================================
    
    data_module = get_datamodule(args)
    
    # ================================
    # UNDERGO TRAINING
    # ================================
    train_model(
        args, data_module, wandb_logger
    )
    
    process_results(args)
    
    wandb.finish()

if __name__ == '__main__':
    main()
