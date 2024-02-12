import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('models.py')

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
import wandb
import scipy
import pandas as pd

from graphmamba_utils import *


def get_labels_lists(outputs):
	all_y_true, all_y_pred = [], []
	for output in outputs:
		all_y_true.extend(output['y_true'].detach().cpu().numpy().tolist())
		all_y_pred.extend(output['y_pred'].detach().cpu().numpy().tolist())

	return all_y_true, all_y_pred


def compute_all_metrics(args, y_true, y_pred):
	metrics = {}
	metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
	metrics['F1_weighted'] = f1_score(y_true, y_pred, average='weighted')
	metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
	metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
	if args.num_classes==2:
		metrics['AUROC_weighted'] = roc_auc_score(y_true, y_pred, average='weighted')
	
	return metrics


def detach_tensors(tensors):
	"""
	Detach losses 
	"""
	if type(tensors)==list:
		detached_tensors = list()
		for tensor in tensors:
			detach_tensors.append(tensor.detach())
	elif type(tensors)==dict:
		detached_tensors = dict()
		for key, tensor in tensors.items():
			detached_tensors[key] = tensor.detach()
	else:
		raise Exception("tensors must be a list or a dict")
	
	return detached_tensors

def reshape_batch(batch):
	"""
	When the dataloaders create multiple samples from one original sample, the input has size (batch_size, no_samples, D)
	
	This function reshapes the input from (batch_size, no_samples, D) to (batch_size * no_samples, D)
	"""
	x, edge_index, edge_attr, y, pe, batch, ptr = batch
	x, y = batch
	x = x.reshape(-1, x.shape[-1])
	y = y.reshape(-1)

	return x, y


class TrainingLightningModule(pl.LightningModule):
	"""
	General class to be inherited by all implemented models (e.g., MLP, CAE, FsNet etc.)

	It implements general training and evaluation functions (e.g., computing losses, logging, training etc.)
	"""
	def __init__(self, args):
		super().__init__()
		self.training_step_outputs = []
		self.validation_step_outputs = []
		self.test_step_outputs = []
		self.args = args
		self.learning_rate = args.learning_rate

	def compute_loss(self, y_true, y_hat):
		losses = {}
		losses['cross_entropy'] = F.cross_entropy(input=y_hat, target=y_true, weight=torch.tensor(self.args.class_weights, device=self.device))

		losses['total'] = losses['cross_entropy'] 
		
		return losses
	
	def log_losses(self, losses, key, dataloader_name=""):
		self.log(f"{key}/total_loss{dataloader_name}", losses['total'].item(), sync_dist=self.args.hpc_run)
		self.log(f"{key}/cross_entropy_loss{dataloader_name}", losses['cross_entropy'].item(), sync_dist=self.args.hpc_run)

	def log_epoch_metrics(self, outputs, key, dataloader_name=""):
		y_true, y_pred = get_labels_lists(outputs)
		if self.args.pretrain:
			self.log(f'{key}/balanced_accuracy{dataloader_name}', balanced_accuracy_score(y_true, y_pred), sync_dist=self.args.hpc_run)
		self.log(f'{key}/F1_weighted{dataloader_name}', f1_score(y_true, y_pred, average='weighted'), sync_dist=self.args.hpc_run)
		self.log(f'{key}/precision_weighted{dataloader_name}', precision_score(y_true, y_pred, average='weighted'), sync_dist=self.args.hpc_run)
		self.log(f'{key}/recall_weighted{dataloader_name}', recall_score(y_true, y_pred, average='weighted'), sync_dist=self.args.hpc_run)
		if self.args.num_classes==2:
			self.log(f'{key}/AUROC_weighted{dataloader_name}', roc_auc_score(y_true, y_pred, average='weighted'), sync_dist=self.args.hpc_run)

	def training_step(self, batch, batch_idx):
		x, y_true = batch


		y_hat = self.forward(x)

		losses = self.compute_loss(y_true, y_hat)

		self.log_losses(losses, key='train')
		self.log("train/lr", self.learning_rate)
		
		outputs = {
			'loss': losses['total'],
			'losses': detach_tensors(losses),
			'y_true': y_true,
			'y_pred': torch.argmax(y_hat, dim=1)
		}
		self.training_step_outputs.append(outputs)
		return outputs

	def on_train_epoch_end(self):
		self.log_zepoch_metrics(self.training_step_outputs, 'train')
		self.training_step_outputs.clear()  # free memory

	def validation_step(self, batch, batch_idx, dataloader_idx=0):
		"""
		- dataloader_idx (int) tells which dataloader is the `batch` coming from
		"""

		x, y_true = reshape_batch(batch)

		y_hat = self.forward(x)

		losses = self.compute_loss(y_true, y_hat)

		output = {
			'losses': detach_tensors(losses),
			'y_true': y_true,
			'y_pred': torch.argmax(y_hat, dim=1)
		}
		while len(self.validation_step_outputs) <= dataloader_idx:
			self.validation_step_outputs.append([])
    
		self.validation_step_outputs[dataloader_idx].append(output)
		return output

	def on_validation_epoch_end(self):
		"""
		- outputs: when no_dataloaders==1 --> A list of dictionaries corresponding to a validation step. 
            when no_dataloaders>1  --> List with length equal to the number of validation dataloaders. Each element is a list with the dictionaries corresponding to a validation step.
		"""
		### Log losses and metrics
		# `outputs_all_dataloaders` is expected to a list of dataloaders.
		# However, when there's only one dataloader, outputs_all_dataloaders is NOT a list.
		# Thus, we transform it in a list to preserve compatibility
		outputs_all_dataloaders = self.validation_step_outputs

		for dataloader_id, outputs in enumerate(outputs_all_dataloaders):
			losses = {
				'total': np.mean([output['losses']['total'].item() for output in outputs]),
				'cross_entropy': np.mean([output['losses']['cross_entropy'].item() for output in outputs])
			}
			if dataloader_id==0: # original validation dataset
				dataloader_name=""
			else:
				dataloader_name=f"__{self.args.val_dataloaders_name[dataloader_id]}"
			self.log_losses(losses, key='valid', dataloader_name=dataloader_name)
			self.log_epoch_metrics(outputs, key='valid', dataloader_name=dataloader_name)
		self.validation_step_outputs.clear()

	def test_step(self, batch, batch_idx, dataloader_idx=0):
		'''accommodates multiple dataloaders'''
		x, y_true = reshape_batch(batch)
		y_hat, x_hat, sparsity_weights = self.forward(x, test_time=True)
		losses = self.compute_loss(y_true, y_hat, x, x_hat, sparsity_weights)

		output =  {
			'losses': detach_tensors(losses),
			'y_true': y_true,
			'y_pred': torch.argmax(y_hat, dim=1),
			'y_hat': y_hat.detach().cpu().numpy()
		}
		while len(self.test_step_outputs) <= dataloader_idx:
			self.test_step_outputs.append([])
		self.test_step_outputs[dataloader_idx].append(output)
		return output

	def on_test_epoch_end(self):
		'''accommodates multiple dataloaders but only uses first'''

		outputs = self.test_step_outputs[0]

		### Save losses
		losses = {
			'total': np.mean([output['losses']['total'].item() for output in outputs]),
			'cross_entropy': np.mean([output['losses']['cross_entropy'].item() for output in outputs])
		}
		self.log_losses(losses, key=self.log_test_key)
		self.log_epoch_metrics(outputs, self.log_test_key)

		#### Save prediction probabilities
		y_hat_list = [output['y_hat'] for output in outputs]
		y_hat_all = np.concatenate(y_hat_list, axis=0)
		y_hat_all = scipy.special.softmax(y_hat_all, axis=1)

		y_hat_all = wandb.Table(dataframe=pd.DataFrame(y_hat_all))
		wandb.log({f'{self.log_test_key}_y_hat': y_hat_all})


	
	def configure_optimizers(self):
		params = self.parameters()

		if self.args.optimizer=='adam':
			optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.args.weight_decay)
		if self.args.optimizer=='adamw':
			optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.args.weight_decay, betas=[0.9, 0.98])
		
		if self.args.lr_scheduler == None:
			return optimizer
		else:
			if self.args.lr_scheduler == 'plateau':
				lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, verbose=True)
			elif self.args.lr_scheduler == 'cosine_warm_restart':
				# Usually the model trains in 1000 epochs. The paper "Snapshot ensembles: train 1, get M for free"
				# 	splits the scheduler for 6 periods. We split into 6 periods as well.
				lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
					T_0 = self.args.cosine_warm_restart_t_0,
					eta_min = self.args.cosine_warm_restart_eta_min,
					verbose=True)
			elif self.args.lr_scheduler == 'linear':
				lr_scheduler = torch.optim.lr_scheduler.LinearLR(
					optimizer, 
					start_factor = self.args.learning_rate,
					end_factor = 3e-5,
					total_iters = self.args.max_steps / self.args.val_check_interval)
			elif self.args.lr_scheduler == 'lambda':
				def scheduler(epoch):
					if epoch < 500:
						return 0.995 ** epoch
					else:
						return 0.1

				lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
					optimizer,
					scheduler)
			else:
				raise Exception()

			return {
				'optimizer': optimizer,
				'lr_scheduler': {
					'scheduler': lr_scheduler,
					'monitor': 'valid/cross_entropy_loss',
					'interval': 'step',
					'frequency': self.args.val_check_interval,
					'name': 'lr_scheduler'
				}
			}



class GraphModel(TrainingLightningModule):
    def __init__(self, args):
        super().__init__(args)
        
        channels = args.graph_model_channels
        pe_dim = args.graph_model_pe_dim
        num_layers = args.graph_model_num_layers
        model_type = args.graph_model_type
        shuffle_ind = args.graph_model_shuffle_ind
        d_state = args.graph_model_d_state
        d_conv = args.graph_model_d_conv
        order_by_degree = args.graph_model_order_by_degree

        self.node_emb = Embedding(28, channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(4, channels)
        self.model_type = model_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree
        
        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            if self.model_type == 'gine':
                conv = GINEConv(nn)
                
            if self.model_type == 'mamba':
                conv = GPSConv(channels, GINEConv(nn), heads=4, attn_dropout=0.5,
                               att_type='mamba',
                               shuffle_ind=self.shuffle_ind,
                               order_by_degree=self.order_by_degree,
                               d_state=d_state, d_conv=d_conv)
                
            if self.model_type == 'transformer':
                conv = GPSConv(channels, GINEConv(nn), heads=4, attn_dropout=0.5, att_type='transformer')
                
            # conv = GINEConv(nn)
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            if self.model_type == 'gine':
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index, batch, edge_attr=edge_attr)
                
        x = global_add_pool(x, batch)
        return self.mlp(x)
    

def get_model(args):
    logger.info(f'Fetching model. args.graph_model_type {args.graph_model_type}')
    
    model = GraphModel(args)
    
    return model
