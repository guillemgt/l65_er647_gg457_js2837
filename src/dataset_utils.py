import glob
import logging
import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
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

def get_train_val_test_sets(dataset_name, keep_full=False):
    logger.info(f'Fetching train, val, and test sets according to args. dataset_name: {dataset_name}')
    
    # Define transformations for image datasets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]) 

    if dataset_name == 'ZINC':
        transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        train_dataset = ZINC(root='./data', subset=True, split='train', pre_transform=transform)
        val_dataset = ZINC(root='./data', subset=True, split='val', pre_transform=transform)
        test_dataset = ZINC(root='./data', subset=True, split='test', pre_transform=transform)
    elif dataset_name == 'CSL':
        transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        train_dataset = GNNBenchmarkDataset(root='./data', name='CSL', split='train', pre_transform=transform)
        val_dataset = GNNBenchmarkDataset(root='./data', name='CSL', split='val', pre_transform=transform)
        test_dataset = GNNBenchmarkDataset(root='./data', name='CSL', split='test', pre_transform=transform)
    else:
        raise ValueError("Unknown dataset")
    
    if keep_full:
        return train_dataset, test_dataset
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    return train_dataset, val_dataset, test_dataset

    
def get_dataloaders(args):
    train_dataset, val_dataset, test_dataset = get_train_val_test_sets(args.dataset)

    logger.info(f'Fetching dataloaders.')
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    # Unshuffled required for datamap_callback later on
    train_unshuffled_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False) 
    val_loader = DataLoader(val_dataset, batch_size=args.validation_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.validation_batch_size, shuffle=False)

    return train_loader, train_unshuffled_loader, val_loader, test_loader

def get_datamodule(args):
    train_loader, train_unshuffled_loader, val_loader, test_loader = get_dataloaders(args)
    data_module = CustomDataModule(train_loader, val_loader, test_loader)
    return data_module

