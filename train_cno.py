import torch
import sys; sys.path.append('../neuralseismic/'); sys.path.append('..')
from neuralseismic_xiao import load_helmholtz_small, seed_torch, LitModel,CNO_net,CNO_net_v2
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import datetime as dt
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, TensorDataset
from neuralop.models import FNO
import blobfile as bf
import wandb

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('--batch_size', nargs='?', type=int, default=128,
                        help='Batch Size')
parser.add_argument('--lr', nargs='?', type=float, default=0.0001,
                        help='Learning Rate')
parser.add_argument('--weight_decay', nargs='?', type=float, default=1e-6,
                        help='weight_decay')
parser.add_argument('--results_path', nargs='?', type=str, default='/home/max0b/neuroseismic/results/',
                        help='Results path')
parser.add_argument('--train_num_steps', nargs='?', type=int, default=200000,
                        help='number of training steps')
parser.add_argument('--device', nargs='?', type=str, default='cuda:0',
						help='device')
parser.add_argument('--epochs', nargs='?', type=int, default=1000,
                        help='number of epochs')
parser.add_argument('--exp_name', nargs='?', type=str, default='CurveAtest',
                        help='Experiment name')
parser.add_argument('--N_layers', nargs='?', type=int, default=5,
                        help='Number of (D) and (U) Blocks in the network')
parser.add_argument('--N_res', nargs='?', type=int, default=4,
                        help='Number of (R) blocks per level (except the neck)')
parser.add_argument('--N_res_neck', nargs='?', type=int, default=6,
                        help='Number of (R) blocks in the neck)')
parser.add_argument('--in_size', nargs='?', type=int, default=128)
parser.add_argument('--freq', nargs='?', type=int, default=8,
                      help='the frequency of the input data')
parser.add_argument('--v0', nargs='?', type=float, default=1.50,
                        help='the value of background field velocity')
parser.add_argument('--fixed_data_loader', nargs='?', type=bool, default=False,
                        help='whether to fix the dataloader for all training, if yes, also need to specify the path')
parser.add_argument('--data_loader_path', nargs='?', type=str, default='/home/max0b/neuroseismic/data_loader_curveA.pth',
                        help='path to save the dataloader')
parser.add_argument('--nn_type', nargs='?', type=str, default='fno',
                        help='the type of neural networks')
parser.add_argument('--io_type', nargs='?', type=str, default='back_in_residual_out',
                        help='the type of input and output data')
parser.add_argument('--fast_dev_run', action='store_true',
                    help='whether to use fast_dev_run in PyTorch Lightning (default: False)')

parser.add_argument('--precision', type=str, choices=["16-mixed", "bf16-mixed", "32","bf16"], default="32",
                    help='whether to use mixed precision (16 for mixed precision, 32 for full precision, default: 32)')
parser.add_argument('--gpus', type=int, default='0',
                    help='comma separated list of GPU ids to use for training (e.g., "0,1,2")')

parser.add_argument('--train_data_path', type=str, default='0',
                    help='data_path')

parser.add_argument('--valid_data_path', type=str, default='0',
                    help='valid_data_path')

parser.add_argument('--weight', type=float, nargs=3, default=[1.0, 1.0, 1.0],
                    help='weights between diff loss functions')
parser.add_argument('--seed', type=int, nargs="?", default=42,
                    help='set the seed')


def find_files(folder_path, file_pattern_key='*.npy'):
    file_pattern = os.path.join(folder_path, file_pattern_key)
    npy_files = glob.glob(file_pattern)

    return npy_files

args = parser.parse_args()
for arg in vars(args):
	print(format(arg, '<20'), format(str(getattr(args, arg)), '<')) 


wandb_logger = WandbLogger(name=args.exp_name, project="neuroseismic",config=vars(args))


# Load data
print("load xiao's dataset")
def _list_image_files_recursively(dir):

    data_dir=dir+'data/'

    label_dir=dir+'label/'

    data_path = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            data_path.append(full_path)
        elif bf.isdir(full_path):
            data_path.extend(_list_image_files_recursively(full_path))

    label_path = []
    for entry in sorted(bf.listdir(label_dir)):
        full_path = bf.join(label_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy"]:
            label_path.append(full_path)
        elif bf.isdir(full_path):
            label_path.extend(_list_image_files_recursively(full_path))

    return data_path,label_path

class BasicDataset(Dataset):
    def __init__(self, data_path,label_path):
        super().__init__()

        self.data_path =data_path
        self.label_path =label_path

    def __len__(self):
        
        return len(self.data_path)

    def __getitem__(self, idx):
        
        path_data = self.data_path[idx]#

        path_label=self.label_path[idx]

        data = np.load(path_data)

        label = np.load(path_label)

        data = torch.tensor(data,dtype=torch.float32)

        label = torch.tensor(label,dtype=torch.float32)
        #print(data.shape,label.shape)
        return  data , label

data_path,label_path = _list_image_files_recursively(args.train_data_path)

data_path_label,label_path_label = _list_image_files_recursively(args.valid_data_path)

train_dataset = BasicDataset(data_path,label_path)

valid_dataset =  BasicDataset(data_path_label,label_path_label)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True, pin_memory=True)
    
val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1 ,drop_last=False, pin_memory=True)
print("done!")

        
if args.io_type == 'back_in_residual_out':
    #model = LitModel(input_channel=3, output_channel=2, nn_type=args.nn_type)
    model = CNO_net(input_channel=3,N_layers=args.N_layers,N_res=args.N_res,N_res_neck=args.N_res_neck,in_size=args.in_size, weight=args.weight,freq=args.freq,v0=args.v0)
  
    # Model

model.configure_optimizers(lr=args.lr, step_size=100, gamma=0.97)

pl.seed_everything(args.seed)

root_path = f'{args.results_path}/{args.exp_name}-{dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'

checkpoint_callback = ModelCheckpoint(
    dirpath=f'{root_path}/checkpoints/',
    save_top_k=3,
    monitor='val_relative_err_real',
    verbose=True,
    mode='min',
    save_last=True
)

trainer = Trainer(logger=wandb_logger,
                  max_epochs=args.epochs,
                  accelerator='gpu',
                  devices=[args.gpus],
                  log_every_n_steps=1,
                  check_val_every_n_epoch=1,
                  callbacks=[checkpoint_callback],
                  default_root_dir=root_path,
                  fast_dev_run=args.fast_dev_run,  
                  precision=args.precision,        
                  )

trainer.fit(model, train_loader, val_loader) 

trainer.save_checkpoint(f"{root_path}/{args.exp_name}_sup_constantv.ckpt")