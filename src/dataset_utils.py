import glob
import logging
import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('utils/dataset_utils.py')

class CustomDataModule(LightningDataModule):
    def __init__(self, train_loader, val_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

def get_train_val_test_sets(dataset_name, val_split_seed, prev_run_name_for_dynamics, keep_full=False):
    logger.info(f'Fetching train, val, and test sets according to args. dataset_name: {dataset_name} | val_split_seed: {val_split_seed} | prev_run_name_for_dynamics: {prev_run_name_for_dynamics}')
    
    # Define transformations for image datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
    elif dataset_name == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        
    elif dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Unknown dataset")
    
    if keep_full:
        return train_dataset, test_dataset
    # Split the training dataset into train and validation
    local_generator = torch.Generator()
    if val_split_seed:
        local_generator.manual_seed(val_split_seed)
    else:
        if prev_run_name_for_dynamics:
            raise Exception("if using datamapped subset for training, val split seed for prev run must be provided.")
        val_split_seed = local_generator.initial_seed()
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=local_generator)

    return train_dataset, val_dataset, test_dataset

    
def get_dataloaders(args):
    train_dataset, val_dataset, test_dataset = get_train_val_test_sets(args.dataset, args.val_split_seed, args.prev_run_name_for_dynamics)

    logger.info(f'Fetching dataloaders.')
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    # Unshuffled required for datamap_callback later on
    train_unshuffled_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False) 
    val_loader = DataLoader(val_dataset, batch_size=args.validation_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.validation_batch_size, shuffle=False)

    return train_loader, train_unshuffled_loader, val_loader, test_loader

def preprocess_cifar10(dataset):
    return dataset

def preprocess_cifar100(dataset):
    return dataset

def preprocess_mnist(dataset):
    return dataset

def preprocess_speechcommands(dataset):
    return dataset

def preprocess_urbansounds8k(dataset):
    return dataset
