from pytorch_lightning import LightningModule
import torch
from torch.utils.data import DataLoader
from .fourier_2d import FNO2d
from .unet import UNet
from .resnet import Resnet18
from .utilis import calculate_relative_loss
from .utilis import count_params
from neuralseismic_xiao import  equation_fd,equation_fd_4th,equation_fft,equation_fft_pad,equation_fd_8th
from .CNOModule import CNO
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
# Updated version with different networks and mode


class LitModel(LightningModule):
    def __init__(self, input_channel=3, output_channel=2, mode_x=16, mode_y=16, width=64,nn_type='fno', weight=[1,1,1],freq=8):
        super().__init__()
        if nn_type == 'fno':
            self.model = FNO2d(mode_x, mode_y, width, input_channel, output_channel)
            #self.model = FNO(n_modes=(mode_x, mode_y),in_channels=input_channel, out_channels= output_channel, n_layers=layers, hidden_channels=width,norm= 'group_norm')
            
        self.loss_fn = torch.nn.MSELoss()
        #self.pde_loss = equation_fd(0.025, preds , 1.0/ (inputs[:,2:3] ** 2), 1.0 / (inputs[:,3:4] **2), inputs[:,0:1] , inputs[:,1:2] , v_0.to(device)) * 1e-5
        self.input_channel = input_channel
        self.weight_data,self.weight_pde,self.weight_IC = weight
        self.freq=freq
        print(self.model)
        n_params = count_params(self.model)
        print(f'\nOur model has {n_params} parameters.')
        self.save_hyperparameters()
        
    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
 
        # Main forward, loss computation, and metrics goes here
        x, y = batch
        device = x.device
        # x, y = batch['x'], batch['y']
        y_hat = self.model(x[:,:self.input_channel])
        #loss_mse = self.loss_fn(y, y_hat)
        # Assuming min_value and freq is already defined
        freq = torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.freq, dtype=torch.float32)#8hz
        frequency = torch.tensor(freq,dtype=torch.float32).to(device)
        
        v_0=torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), 1.50, dtype=torch.float32)
        v_0=torch.tensor(v_0,dtype=torch.float32).to(device)

        #=========combine of the PDE and MSE loss===========#
        data_loss = self.loss_fn(y_hat,y)


        #pde_loss = equation_fd(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 #xinquan's
        #pde_loss = equation_fd_4th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4
        #pde_loss=equation_fft(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        pde_loss=  equation_fd_8th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
          
        pde_loss=torch.sum(pde_loss ** 2)
        
        f = torch.zeros(pde_loss.shape, device=x.device)
        
        pde_loss = F.mse_loss(pde_loss, f)*1e-5        
        
        IC_loss= self.loss_fn(y_hat[:,:,:2,:],y[:,:,:2,:]) \
            + self.loss_fn(y_hat[:,:,:,:2],y[:,:,:,:2]) \
            + self.loss_fn(y_hat[:,:,:,-2:],y[:,:,:,-2:])\
            + self.loss_fn(y_hat[:,:,-2:,:],y[:,:,-2:,:])
          
    


        loss = data_loss *self.weight_data+ pde_loss *self.weight_pde+ IC_loss *self.weight_IC # 使用组合损失
        
        #print("data_loss,pde_loss,IC_loss",data_loss,pde_loss,IC_loss)
        
        
        #loss=pde_loss
        relative_err_real = calculate_relative_loss((y_hat-y)[:,0:1], y[:,0:1], reduction='mean')
        
        relative_err_imag = calculate_relative_loss((y_hat-y)[:,1:2], y[:,1:2], reduction='mean')
        # pde_loss = equation_fd(0.025, y_hat[:,:,...] , 1.0/ (x[:,2:3,...] ** 2), 1.0 / (x[:,3:4,...] **2), x[:,0:1,...] , x[:,1:2,...] , x[:,4:5,...]) * 1e-5
        # pde_loss = 0.2 * torch.sum(pde_loss ** 2)
        #loss = loss_mse + 0.2 * torch.sum(pde_loss ** 2)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_MSE_loss", data_loss, on_step=True, on_epoch=True)
        self.log("train_pde_loss", pde_loss, on_step=True, on_epoch=True)
        self.log("train_relative_err_real", relative_err_real, on_step=True, on_epoch=True)
        self.log("train_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        device = x.device
        # x, y = batch['x'], batch['y']
        y_hat = self.model(x[:,:self.input_channel])
        #loss_mse = self.loss_fn(y, y_hat)
        # Assuming min_value and freq is already defined
        freq = torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.freq, dtype=torch.float32)#8hz
        frequency = torch.tensor(freq,dtype=torch.float32).to(device)
        
        v_0=torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), 1.50, dtype=torch.float32)
        v_0=torch.tensor(v_0,dtype=torch.float32).to(device)
        
        #=========combine of the PDE and MSE loss===========#
        data_loss = self.loss_fn(y_hat,y)
        
        #pde_loss = equation_fd(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4

        #pde_loss = equation_fd_4th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        pde_loss=equation_fft(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        pde_loss=torch.sum(pde_loss ** 2)
        f = torch.zeros(pde_loss.shape, device=x.device)
        pde_loss = F.mse_loss(pde_loss, f) *1e-5 

        #pde_loss = equation_fd_loss(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)
        
        IC_loss = self.loss_fn(y_hat[:,:,:2,:],y[:,:,:2,:]) \
            + self.loss_fn(y_hat[:,:,:,:2],y[:,:,:,:2]) + self.loss_fn(y_hat[:,:,-2:,:],y[:,:,-2:,:]) + self.loss_fn(y_hat[:,:,:,-2:],y[:,:,:,-2:])
            
        loss = data_loss *self.weight_data+ pde_loss *self.weight_pde+ IC_loss *self.weight_IC # 使用组合损失        

        
        relative_err_real = calculate_relative_loss((y_hat-y)[:,0:1], y[:,0:1], reduction='mean')
        
        relative_err_imag = calculate_relative_loss((y_hat-y)[:,1:2], y[:,1:2], reduction='mean')
        
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_MSE_loss", data_loss, on_step=True, on_epoch=True)
        self.log("val_pde_loss", pde_loss, on_step=True, on_epoch=True)
        self.log("val_relative_err_real", relative_err_real, on_step=True, on_epoch=True)
        self.log("val_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True)




    def configure_optimizers(self,lr=1e-3, step_size=100, gamma=0.9):
        # Return one or several optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
        #              'interval': 'epoch',  # 或 'step' 对于每步更新
        #              'frequency': 1}
        return [optimizer]#, [scheduler]

#==================自定义CNO===================#

def equation_fd_loss(inn_var, out_var, m_train, m0_train, u0_real_train, u0_imag_train, f):
    
    # inn_var: Spatial grid spacing  
    # out_var: The perturbed part of the wavefield, referred to as "scattered wavefield" in the paper
    # m_train: Updated medium parameter model, which is the square of the slowness
    # m0_train: Background medium model
    # u0_real_train, u0_imag_train: Real and imaginary parts of the background wavefield
    # f: Frequency
    # omega: 2 * pi * f
    # d2udx2 and d2udy2: Second-order spatial derivatives of the wavefield in the x and y directions
    
    omega = (2.0 * f * torch.pi)[...,1:-1,1:-1]
    d2udx2 = torch.zeros(out_var.shape).to(out_var.device)
    d2udy2 = torch.zeros(out_var.shape).to(out_var.device)
    d2udx2[:,0:1,1:-1,1:-1] = (out_var[:,0:1,0:-2, 1:-1] + out_var[:,0:1,2:, 1:-1] - 2*out_var[:,0:1,1:-1, 1:-1]) / (inn_var)**2
    d2udx2[:,1:2,1:-1,1:-1] = (out_var[:,1:2,0:-2, 1:-1] + out_var[:,1:2,2:, 1:-1] - 2*out_var[:,1:2,1:-1, 1:-1]) / (inn_var)**2
    d2udy2[:,0:1,1:-1,1:-1] = (out_var[:,0:1,1:-1, 0:-2] + out_var[:,0:1,1:-1, 2:] - 2*out_var[:,0:1,1:-1, 1:-1]) / (inn_var)**2
    d2udy2[:,1:2,1:-1,1:-1] = (out_var[:,1:2,1:-1, 0:-2] + out_var[:,1:2,1:-1, 2:] - 2*out_var[:,1:2,1:-1, 1:-1]) / (inn_var)**2
    res_x = omega**2*(m_train[:,:,1:-1,1:-1]) * out_var[:,0:1,1:-1,1:-1] + d2udx2[:,0:1,1:-1,1:-1] + d2udy2[:,0:1,1:-1,1:-1] + omega**2 * (m_train[:,:,1:-1,1:-1]- m0_train[:,:,1:-1, 1:-1])* u0_real_train[:,:,1:-1,1:-1] #equation 4 in the paper
    res_y = omega**2*(m_train[:,:,1:-1,1:-1]) * out_var[:,1:2,1:-1,1:-1] + d2udx2[:,1:2,1:-1,1:-1] + d2udy2[:,1:2,1:-1,1:-1] + omega**2 * (m_train[:,:,1:-1,1:-1]- m0_train[:,:,1:-1,1:-1])* u0_imag_train[:,:,1:-1,1:-1] 
    # return torch.cat((res_x, res_y), dim=1), d2udx2, d2udy2
    aa=torch.cat((res_x, res_y), dim=1)*1e-2
    f = torch.zeros(aa.shape, device=aa.device)
    loss_f = F.mse_loss(aa, f)
    return loss_f
    
    

class CNO_net(LightningModule):
    #==================自定义CNO===================#
    
    def __init__(self, input_channel=3, N_layers=5, N_res=4, N_res_neck=6,in_size=128,pde_weight=0, weight=[1,1,1],freq=8,v0=1.50):
        super().__init__()

        self.model = CNO(in_dim  = input_channel,      # Number of input channels.
                                    in_size = in_size,                # Input spatial size
                                    N_layers = N_layers,                   # Number of (D) and (U) Blocks in the network
                                    N_res =N_res,                         # Number of (R) Blocks per level
                                    N_res_neck =N_res_neck,
                                    channel_multiplier =32,
                                    conv_kernel=3,
                                    cutoff_den = 2.0001,
                                    filter_size=6,  
                                    lrelu_upsampling = 2,
                                    half_width_mult  = 0.8,
                                    activation = 'lrelu')
        
            
        self.loss_fn = torch.nn.MSELoss()
        #self.pde_loss = equation_fd(0.025, preds , 1.0/ (inputs[:,2:3] ** 2), 1.0 / (inputs[:,3:4] **2), inputs[:,0:1] , inputs[:,1:2] , v_0.to(device)) * 1e-5
        self.input_channel = input_channel
        self.weight_data,self.weight_pde,self.weight_IC = weight
        self.pde_weight = pde_weight
        self.freq=freq
        self.v0=v0
        n_params = count_params(self.model)
        print(f'\nOur model has {n_params} parameters.')
        self.save_hyperparameters()
        
    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
 
        # Main forward, loss computation, and metrics goes here
        x, y = batch
        device = x.device
        # x, y = batch['x'], batch['y']
        y_hat = self.model(x[:,:self.input_channel])
        #loss_mse = self.loss_fn(y, y_hat)
        # Assuming min_value and freq is already defined
        freq = torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.freq, dtype=torch.float32)#8hz
        frequency = torch.tensor(freq,dtype=torch.float32).to(device)
        
        v_0=torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.v0, dtype=torch.float32)
        v_0=torch.tensor(v_0,dtype=torch.float32).to(device)

        #=========combine of the PDE and MSE loss===========#
        data_loss = self.loss_fn(y_hat,y)


        #pde_loss = equation_fd(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 #xinquan's
        #pde_loss = equation_fd_4th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4
        #pde_loss=equation_fft(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        pde_loss = equation_fd_8th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
          
        pde_loss=torch.sum(pde_loss ** 2)
        
        f = torch.zeros(pde_loss.shape, device=x.device)
        
        pde_loss = F.mse_loss(pde_loss, f)*1e-5        
        
        IC_loss= self.loss_fn(y_hat[:,:,:4,:],y[:,:,:4,:]) \
            + self.loss_fn(y_hat[:,:,:,:4],y[:,:,:,:4]) \
            + self.loss_fn(y_hat[:,:,:,-4:],y[:,:,:,-4:])\
            + self.loss_fn(y_hat[:,:,-4:,:],y[:,:,-4:,:])
          
    


        loss = data_loss *self.weight_data+ pde_loss *self.weight_pde+ IC_loss *self.weight_IC # 使用组合损失
        
        
        
        #loss=pde_loss
        relative_err_real = calculate_relative_loss((y_hat-y)[:,0:1], y[:,0:1], reduction='mean')
        
        relative_err_imag = calculate_relative_loss((y_hat-y)[:,1:2], y[:,1:2], reduction='mean')
        # pde_loss = equation_fd(0.025, y_hat[:,:,...] , 1.0/ (x[:,2:3,...] ** 2), 1.0 / (x[:,3:4,...] **2), x[:,0:1,...] , x[:,1:2,...] , x[:,4:5,...]) * 1e-5
        # pde_loss = 0.2 * torch.sum(pde_loss ** 2)
        #loss = loss_mse + 0.2 * torch.sum(pde_loss ** 2)
        self.log("train_loss", loss, on_step=True, on_epoch=True,sync_dist=True)
        self.log("train_MSE_loss", data_loss, on_step=True, on_epoch=True,sync_dist=True)
        self.log("train_pde_loss", pde_loss, on_step=True, on_epoch=True,sync_dist=True)
        self.log("train_relative_err_real", relative_err_real, on_step=True, on_epoch=True,sync_dist=True)
        self.log("train_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True,sync_dist=True)
        
        # Log learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']

        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        device = x.device
        # x, y = batch['x'], batch['y']
        y_hat = self.model(x[:,:self.input_channel])
        #loss_mse = self.loss_fn(y, y_hat)
        # Assuming min_value and freq is already defined
        freq = torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.freq, dtype=torch.float32)#8hz
        frequency = torch.tensor(freq,dtype=torch.float32).to(device)
        
        v_0=torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.v0, dtype=torch.float32)
        v_0=torch.tensor(v_0,dtype=torch.float32).to(device)
        
        #=========combine of the PDE and MSE loss===========#
        data_loss = self.loss_fn(y_hat,y)
        
        #pde_loss = equation_fd(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4

        #pde_loss = equation_fd_4th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        #pde_loss=equation_fft(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        pde_loss = equation_fd_8th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        pde_loss=torch.sum(pde_loss ** 2)
        
        f = torch.zeros(pde_loss.shape, device=x.device)
        
        pde_loss = F.mse_loss(pde_loss, f) *1e-5 

        #pde_loss = equation_fd_loss(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)
        
        IC_loss = self.loss_fn(y_hat[:,:,:4,:],y[:,:,:4,:]) \
            + self.loss_fn(y_hat[:,:,:,:4],y[:,:,:,:4]) + self.loss_fn(y_hat[:,:,-4:,:],y[:,:,-4:,:]) + self.loss_fn(y_hat[:,:,:,-4:],y[:,:,:,-4:])
            
        loss = data_loss *self.weight_data+ pde_loss *self.weight_pde+ IC_loss *self.weight_IC # 使用组合损失        
        #print("data_loss,pde_loss,IC_loss",data_loss,pde_loss,IC_loss)
        
        relative_err_real = calculate_relative_loss((y_hat-y)[:,0:1], y[:,0:1], reduction='mean')
        
        relative_err_imag = calculate_relative_loss((y_hat-y)[:,1:2], y[:,1:2], reduction='mean')
        
        self.log("val_loss", loss, on_step=True, on_epoch=True,sync_dist=True)
        self.log("val_MSE_loss", data_loss, on_step=True, on_epoch=True,sync_dist=True)
        self.log("val_pde_loss", pde_loss, on_step=True, on_epoch=True,sync_dist=True)
        self.log("val_relative_err_real", relative_err_real, on_step=True, on_epoch=True,sync_dist=True)
        self.log("val_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True,sync_dist=True)
        # Log learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']



    def configure_optimizers(self,lr=1e-3, step_size=100, gamma=0.9):
        # Return one or several optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
                       'interval': 'epoch',  # 或 'step' 对于每步更新
                       'frequency': 1}
        return [optimizer], [scheduler]
    
    
    
class CNO_net_mult_freq(LightningModule):
    #==================自定义CNO===================#
    
    def __init__(self, input_channel=3, N_layers=5, N_res=4, N_res_neck=6,in_size=128,pde_weight=0, weight=[1,1,1],freq=8,v0=1.50):
        super().__init__()

        self.model = CNO(in_dim  = input_channel,      # Number of input channels.
                                    in_size = in_size,                # Input spatial size
                                    N_layers = N_layers,                   # Number of (D) and (U) Blocks in the network
                                    N_res =N_res,                         # Number of (R) Blocks per level
                                    N_res_neck =N_res_neck,
                                    channel_multiplier =32,
                                    conv_kernel=3,
                                    cutoff_den = 2.0001,
                                    filter_size=6,  
                                    lrelu_upsampling = 2,
                                    half_width_mult  = 0.8,
                                    activation = 'lrelu')
        
            
        self.loss_fn = torch.nn.MSELoss()
        #self.pde_loss = equation_fd(0.025, preds , 1.0/ (inputs[:,2:3] ** 2), 1.0 / (inputs[:,3:4] **2), inputs[:,0:1] , inputs[:,1:2] , v_0.to(device)) * 1e-5
        self.input_channel = input_channel
        self.weight_data,self.weight_pde,self.weight_IC = weight
        self.pde_weight = pde_weight
        self.freq=freq
        self.v0=v0
        n_params = count_params(self.model)
        print(f'\nOur model has {n_params} parameters.')
        self.save_hyperparameters()
        
    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)
    
        
    def training_step(self, batch, batch_idx):
        
        x, y = batch
        device = x.device

        # Forward pass
        y_hat = self.model(x[:, :self.input_channel])

        # Data loss (MSE)
        data_loss = self.loss_fn(y_hat, y)

        # IC loss (Boundary constraints)
        IC_loss = (
            self.loss_fn(y_hat[:, :, :4, :], y[:, :, :4, :])
            + self.loss_fn(y_hat[:, :, :, :4], y[:, :, :, :4])
            + self.loss_fn(y_hat[:, :, -4:, :], y[:, :, -4:, :])
            + self.loss_fn(y_hat[:, :, :, -4:], y[:, :, :, -4:])
        )

        # PDE loss (Batch-wise frequency processing)
        unique_freqs = torch.unique(x[:, 3, 0, 0])
        total_pde_loss = 0.0

        for freq in unique_freqs:
            
            indices = (x[:, 3, 0, 0] == freq).nonzero(as_tuple=True)[0]

            x_freq_batch = x[indices]
            y_hat_freq_batch = y_hat[indices]

            frequency_batch = torch.full(
                (len(indices), 1, x.shape[2], x.shape[3]),
                freq.item(), device=device
            )
            v_0_batch = torch.full_like(frequency_batch, self.v0)

            pde_residual = equation_fd_8th(
                0.025,
                y_hat_freq_batch,
                1.0 / (x_freq_batch[:, 2:3] ** 2),
                1.0 / (v_0_batch ** 2),
                x_freq_batch[:, 0:1],
                x_freq_batch[:, 1:2],
                frequency_batch
            ) * 1e-4

            total_pde_loss += torch.sum(pde_residual ** 2)

        # Final PDE loss
        pde_loss = total_pde_loss / x.shape[0]  # 正则化到样本数

        f = torch.zeros(pde_loss.shape, device=x.device)
        
        pde_loss = F.mse_loss(pde_loss, f) *1e-5 
        # Total loss (weighted sum)
        loss = (
            self.weight_data * data_loss
            
            + self.weight_pde * pde_loss
            
            + self.weight_IC * IC_loss
        )
        #print("data_loss,pde_loss,IC_loss",data_loss,pde_loss,IC_loss)
        # Relative errors
        relative_err_real = calculate_relative_loss(
            (y_hat - y)[:, 0:1], y[:, 0:1], reduction='mean'
        )
        relative_err_imag = calculate_relative_loss(
            (y_hat - y)[:, 1:2], y[:, 1:2], reduction='mean'
        )

        # Logging losses
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_MSE_loss", data_loss, on_step=True, on_epoch=True)
        self.log("train_pde_loss", pde_loss, on_step=True, on_epoch=True)
        self.log("train_relative_err_real", relative_err_real, on_step=True, on_epoch=True)
        self.log("train_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True)

        # Learning rate logging
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log("learning_rate", lr, on_step=True, on_epoch=True)

        return loss

    
    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        device = x.device

        # Forward pass
        y_hat = self.model(x[:, :self.input_channel])

        # Data loss (MSE)
        data_loss = self.loss_fn(y_hat, y)

        # IC loss (Boundary constraints)
        IC_loss = (
            self.loss_fn(y_hat[:, :, :4, :], y[:, :, :4, :])
            + self.loss_fn(y_hat[:, :, :, :4], y[:, :, :, :4])
            + self.loss_fn(y_hat[:, :, -4:, :], y[:, :, -4:, :])
            + self.loss_fn(y_hat[:, :, :, -4:], y[:, :, :, -4:])
        )

        # PDE loss (Handling different frequencies in validation batch)
        unique_freqs = torch.unique(x[:, 3, 0, 0])
        total_pde_loss = 0.0

        for freq in unique_freqs:
            indices = (x[:, 3, 0, 0] == freq).nonzero(as_tuple=True)[0]

            x_freq_batch = x[indices]
            y_hat_freq_batch = y_hat[indices]

            frequency_batch = torch.full(
                (len(indices), 1, x.shape[2], x.shape[3]),
                freq.item(), device=device
            )
            v_0_batch = torch.full_like(frequency_batch, self.v0)

            pde_residual = equation_fd_8th(
                0.025,
                y_hat_freq_batch,
                1.0 / (x_freq_batch[:, 2:3] ** 2),
                1.0 / (v_0_batch ** 2),
                x_freq_batch[:, 0:1],
                x_freq_batch[:, 1:2],
                frequency_batch
            ) * 1e-4

            total_pde_loss += torch.sum(pde_residual ** 2)

        # Final PDE loss
        pde_loss = total_pde_loss / x.shape[0]  # 正则化到样本数

        f = torch.zeros(pde_loss.shape, device=x.device)
        
        pde_loss = F.mse_loss(pde_loss, f) *1e-5

        # Total loss (weighted combination)
        loss = (
            self.weight_data * data_loss
            + self.weight_pde * pde_loss
            + self.weight_IC * IC_loss
        )

        # Relative errors calculation
        relative_err_real = calculate_relative_loss(
            (y_hat - y)[:, 0:1], y[:, 0:1], reduction='mean'
        )
        relative_err_imag = calculate_relative_loss(
            (y_hat - y)[:, 1:2], y[:, 1:2], reduction='mean'
        )

        # Logging validation metrics
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_MSE_loss", data_loss, on_step=True, on_epoch=True)
        self.log("val_pde_loss", pde_loss, on_step=True, on_epoch=True)
        self.log("val_relative_err_real", relative_err_real, on_step=True, on_epoch=True)
        self.log("val_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True)

        # Log current learning rate for monitoring (optional for validation)
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log("val_learning_rate", lr, on_step=True, on_epoch=True)



    def configure_optimizers(self,lr=1e-3, step_size=100, gamma=0.9):
        # Return one or several optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
                       'interval': 'epoch',  # 或 'step' 对于每步更新
                       'frequency': 1}
        return [optimizer], [scheduler]



class CNO_net_v2(LightningModule):
    #==================自定义CNO===================#
    #first only use data loss for 250 epochs, then  pde loss and BC loss for the rest epochs
    def __init__(self, input_channel=3, N_layers=5, N_res=4, N_res_neck=6,in_size=128,pde_weight=0, weight=[1,1,1]):
        super().__init__()

        self.model = CNO(in_dim  = input_channel,      # Number of input channels.
                                    in_size = in_size,                # Input spatial size
                                    N_layers = N_layers,                   # Number of (D) and (U) Blocks in the network
                                    N_res =N_res,                         # Number of (R) Blocks per level
                                    N_res_neck =N_res_neck,
                                    channel_multiplier =32,
                                    conv_kernel=3,
                                    cutoff_den = 2.0001,
                                    filter_size=6,  
                                    lrelu_upsampling = 2,
                                    half_width_mult  = 0.8,
                                    activation = 'lrelu')
            
        self.loss_fn = torch.nn.MSELoss()
        self.input_channel = input_channel
        self.weight_data,self.weight_pde,self.weight_IC = weight
        self.pde_weight = pde_weight
        print(self.model)
        n_params = count_params(self.model)
        print(f'\nOur model has {n_params} parameters.')
        self.save_hyperparameters()
        
    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
 
        # Main forward, loss computation, and metrics goes here
        x, y = batch
        device = x.device
        # x, y = batch['x'], batch['y']
        y_hat = self.model(x[:,:self.input_channel])
        #loss_mse = self.loss_fn(y, y_hat)
        # Assuming min_value and freq is already defined
        freq = torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), 8, dtype=torch.float32)#8hz
        frequency = torch.tensor(freq,dtype=torch.float32).to(device)
        
        v_0=torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), 1.50, dtype=torch.float32)
        v_0=torch.tensor(v_0,dtype=torch.float32).to(device)

        #=========combine of the PDE and MSE loss===========#
        data_loss = self.loss_fn(y_hat,y)


        #pde_loss = equation_fd(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 #xinquan's
        #pde_loss = equation_fd_4th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4
        pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
          
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
          
        pde_loss=torch.sum(pde_loss ** 2)
        
        f = torch.zeros(pde_loss.shape, device=x.device)
        
        pde_loss = F.mse_loss(pde_loss, f)*1e-5        
        
        IC_loss= self.loss_fn(y_hat[:,:,:2,:],y[:,:,:2,:]) \
            + self.loss_fn(y_hat[:,:,:,:2],y[:,:,:,:2])*0 \
            + self.loss_fn(y_hat[:,:,:,-2:],y[:,:,:,-2:])*0\
            + self.loss_fn(y_hat[:,:,-2:,:],y[:,:,-2:,:])*0
          
    


        loss = data_loss *self.weight_data+ pde_loss *self.weight_pde+ IC_loss *self.weight_IC # 使用组合损失
        
        #print("data_loss,pde_loss,IC_loss",data_loss,pde_loss,IC_loss)
        
        
        #loss=pde_loss
        relative_err_real = calculate_relative_loss((y_hat-y)[:,0:1], y[:,0:1], reduction='mean')
        
        relative_err_imag = calculate_relative_loss((y_hat-y)[:,1:2], y[:,1:2], reduction='mean')
        # pde_loss = equation_fd(0.025, y_hat[:,:,...] , 1.0/ (x[:,2:3,...] ** 2), 1.0 / (x[:,3:4,...] **2), x[:,0:1,...] , x[:,1:2,...] , x[:,4:5,...]) * 1e-5
        # pde_loss = 0.2 * torch.sum(pde_loss ** 2)
        #loss = loss_mse + 0.2 * torch.sum(pde_loss ** 2)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_MSE_loss", data_loss, on_step=True, on_epoch=True)
        self.log("train_pde_loss", pde_loss, on_step=True, on_epoch=True)
        self.log("train_relative_err_real", relative_err_real, on_step=True, on_epoch=True)
        self.log("train_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True)
        
        # Log learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log("learning_rate", lr, on_step=True, on_epoch=True)

        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        device = x.device
        # x, y = batch['x'], batch['y']
        y_hat = self.model(x[:,:self.input_channel])
        #loss_mse = self.loss_fn(y, y_hat)
        # Assuming min_value and freq is already defined
        freq = torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), 8, dtype=torch.float32)#8hz
        frequency = torch.tensor(freq,dtype=torch.float32).to(device)
        
        v_0=torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), 1.50, dtype=torch.float32)
        v_0=torch.tensor(v_0,dtype=torch.float32).to(device)
        
        #=========combine of the PDE and MSE loss===========#
        data_loss = self.loss_fn(y_hat,y)
        
        #pde_loss = equation_fd(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4

        #pde_loss = equation_fd_4th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        pde_loss=torch.sum(pde_loss ** 2)
        f = torch.zeros(pde_loss.shape, device=x.device)
        pde_loss = F.mse_loss(pde_loss, f) *1e-5 

        #pde_loss = equation_fd_loss(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)
        
        IC_loss = self.loss_fn(y_hat[:,:,:2,:],y[:,:,:2,:]) \
            + self.loss_fn(y_hat[:,:,:,:2],y[:,:,:,:2])*0 + self.loss_fn(y_hat[:,:,-2:,:],y[:,:,-2:,:])*0 + self.loss_fn(y_hat[:,:,:,-2:],y[:,:,:,-2:])*0
            
        loss = data_loss *self.weight_data+ pde_loss *self.weight_pde+ IC_loss *self.weight_IC # 使用组合损失        

        
        relative_err_real = calculate_relative_loss((y_hat-y)[:,0:1], y[:,0:1], reduction='mean')
        
        relative_err_imag = calculate_relative_loss((y_hat-y)[:,1:2], y[:,1:2], reduction='mean')
        
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_MSE_loss", data_loss, on_step=True, on_epoch=True)
        self.log("val_pde_loss", pde_loss, on_step=True, on_epoch=True)
        self.log("val_relative_err_real", relative_err_real, on_step=True, on_epoch=True)
        self.log("val_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True)
        # Log learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log("val_learning_rate", lr, on_step=True, on_epoch=True)



    def configure_optimizers(self,lr=1e-3, step_size=100, gamma=0.9):
        # Return one or several optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
                      'interval': 'epoch',  # 或 'step' 对于每步更新
                      'frequency': 1}
        return [optimizer], [scheduler]
    
class CNO_net_v3(LightningModule):
    #==================自定义CNO===================#
    #first only use data loss for 250 epochs, then  pde loss and BC loss for the rest epochs
    def __init__(self, input_channel=3, N_layers=5, N_res=4, N_res_neck=6,in_size=128,pde_weight=0, weight=[1,1,1]):
        super().__init__()

        self.model = CNO(in_dim  = input_channel,      # Number of input channels.
                                    in_size = in_size,                # Input spatial size
                                    N_layers = N_layers,                   # Number of (D) and (U) Blocks in the network
                                    N_res =N_res,                         # Number of (R) Blocks per level
                                    N_res_neck =N_res_neck,
                                    channel_multiplier =32,
                                    conv_kernel=3,
                                    cutoff_den = 2.0001,
                                    filter_size=6,  
                                    lrelu_upsampling = 2,
                                    half_width_mult  = 0.8,
                                    activation = 'lrelu')
            
        self.loss_fn = torch.nn.MSELoss()
        self.input_channel = input_channel
        self.weight_data,self.weight_pde,self.weight_IC = weight
        self.pde_weight = pde_weight
        print(self.model)
        n_params = count_params(self.model)
        print(f'\nOur model has {n_params} parameters.')
        self.save_hyperparameters()
        
    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
 
        # Main forward, loss computation, and metrics goes here
        x, y = batch
        device = x.device
        # x, y = batch['x'], batch['y']
        y_hat = self.model(x[:,:self.input_channel])
        #loss_mse = self.loss_fn(y, y_hat)
        # Assuming min_value and freq is already defined
        freq = torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), 8, dtype=torch.float32)#8hz
        frequency = torch.tensor(freq,dtype=torch.float32).to(device)
        
        v_0=torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), 1.50, dtype=torch.float32)
        v_0=torch.tensor(v_0,dtype=torch.float32).to(device)

        #=========combine of the PDE and MSE loss===========#
        data_loss = self.loss_fn(y_hat,y)


        #pde_loss = equation_fd(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 #xinquan's
        #pde_loss = equation_fd_4th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        pde_loss=  equation_fd_8th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
          
        pde_loss=torch.sum(pde_loss ** 2)
        
        f = torch.zeros(pde_loss.shape, device=x.device)
        
        pde_loss = F.mse_loss(pde_loss, f)*1e-5        
        
        IC_loss= self.loss_fn(y_hat[:,:,:2,:],y[:,:,:2,:]) \
            + self.loss_fn(y_hat[:,:,:,:2],y[:,:,:,:2])*1 \
            + self.loss_fn(y_hat[:,:,:,-2:],y[:,:,:,-2:])*1\
            + self.loss_fn(y_hat[:,:,-2:,:],y[:,:,-2:,:])*1
          
        print("data_loss,pde_loss,IC_loss",data_loss,pde_loss,IC_loss)
        
        #print("data_loss,pde_loss,IC_loss",data_loss,pde_loss,IC_loss)
        if self.current_epoch < 500:
            loss = data_loss *self.weight_data*(1-self.current_epoch*0.0)+ (pde_loss *self.weight_pde+ IC_loss *self.weight_IC)*1 # 使用组合损失        
        else:
            loss = pde_loss *self.weight_pde+ IC_loss *self.weight_IC        
        
        #loss=pde_loss
        relative_err_real = calculate_relative_loss((y_hat-y)[:,0:1], y[:,0:1], reduction='mean')
        
        relative_err_imag = calculate_relative_loss((y_hat-y)[:,1:2], y[:,1:2], reduction='mean')
        # pde_loss = equation_fd(0.025, y_hat[:,:,...] , 1.0/ (x[:,2:3,...] ** 2), 1.0 / (x[:,3:4,...] **2), x[:,0:1,...] , x[:,1:2,...] , x[:,4:5,...]) * 1e-5
        # pde_loss = 0.2 * torch.sum(pde_loss ** 2)
        #loss = loss_mse + 0.2 * torch.sum(pde_loss ** 2)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_MSE_loss", data_loss, on_step=True, on_epoch=True)
        self.log("train_pde_loss", pde_loss, on_step=True, on_epoch=True)
        self.log("train_relative_err_real", relative_err_real, on_step=True, on_epoch=True)
        self.log("train_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True)
        
        # Log learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log("learning_rate", lr, on_step=True, on_epoch=True)

        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        device = x.device
        # x, y = batch['x'], batch['y']
        y_hat = self.model(x[:,:self.input_channel])
        #loss_mse = self.loss_fn(y, y_hat)
        # Assuming min_value and freq is already defined
        freq = torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), 8, dtype=torch.float32)#8hz
        frequency = torch.tensor(freq,dtype=torch.float32).to(device)
        
        v_0=torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), 1.50, dtype=torch.float32)
        v_0=torch.tensor(v_0,dtype=torch.float32).to(device)
        
        #=========combine of the PDE and MSE loss===========#
        data_loss = self.loss_fn(y_hat,y)
        
        #pde_loss = equation_fd(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4

        #pde_loss = equation_fd_4th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        pde_loss=  equation_fd_8th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        pde_loss=torch.sum(pde_loss ** 2)
        f = torch.zeros(pde_loss.shape, device=x.device)
        pde_loss = F.mse_loss(pde_loss, f) *1e-5 

        #pde_loss = equation_fd_loss(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)
        
        IC_loss = self.loss_fn(y_hat[:,:,:2,:],y[:,:,:2,:]) \
            + self.loss_fn(y_hat[:,:,:,:2],y[:,:,:,:2])*1 + self.loss_fn(y_hat[:,:,-2:,:],y[:,:,-2:,:])*1 + self.loss_fn(y_hat[:,:,:,-2:],y[:,:,:,-2:])*1
            
        if self.current_epoch < 500:
            loss = data_loss *self.weight_data*(1)+ (pde_loss *self.weight_pde+ IC_loss *self.weight_IC)*1 # 使用组合损失        
        else:
            loss = pde_loss *self.weight_pde+ IC_loss *self.weight_IC
        
        relative_err_real = calculate_relative_loss((y_hat-y)[:,0:1], y[:,0:1], reduction='mean')
        
        relative_err_imag = calculate_relative_loss((y_hat-y)[:,1:2], y[:,1:2], reduction='mean')
        
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_MSE_loss", data_loss, on_step=True, on_epoch=True)
        self.log("val_pde_loss", pde_loss, on_step=True, on_epoch=True)
        self.log("val_relative_err_real", relative_err_real, on_step=True, on_epoch=True)
        self.log("val_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True)
        # Log learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log("val_learning_rate", lr, on_step=True, on_epoch=True)



    def configure_optimizers(self,lr=1e-3, step_size=100, gamma=0.9):
        # Return one or several optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
                      'interval': 'epoch',  # 或 'step' 对于每步更新
                      'frequency': 1}
        return [optimizer], [scheduler]
    

    
    
    
    
    
    
class FNO_NP(LightningModule):
    #==================自定义CNO===================#
    
    def __init__(self, input_channel=3, N_layers=5, in_size=128, pde_weight=0, weight=[1,1,1]):
        super().__init__()

        # Replace the custom CNO model with FNO2d from neuraloperator
        self.model = FNO_np(
            n_modes_height=16,           # Number of Fourier modes in the first dimension
            n_modes_width=16,           # Number of Fourier modes in the second dimension
            width=32,            # Width of the hidden layers
            input_dim=input_channel,   # Number of input channels
            output_dim=2,        # Number of output channels
            depth=N_layers       # Number of layers in the model
        )
        
        self.loss_fn = torch.nn.MSELoss()
        self.input_channel = input_channel
        self.weight_data, self.weight_pde, self.weight_IC = weight
        self.pde_weight = pde_weight
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'\nOur model has {n_params} parameters.')
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        device = x.device
        y_hat = self.model(x[:,:self.input_channel])
        data_loss = self.loss_fn(y_hat, y)
        
        # Define frequency and v_0
        freq = torch.full((x.shape[0], 1, x.shape[2], x.shape[2]), 8, dtype=torch.float32, device=device)  # 8Hz
        v_0 = torch.full((x.shape[0], 1, x.shape[2], x.shape[2]), 1.50, dtype=torch.float32, device=device)
        
        # PDE loss
        pde_loss = equation_fft(0.025, y_hat, 1.0 / (x[:, 2:3] ** 2), 1.0 / (v_0 ** 2), x[:, 0:1], x[:, 1:2], freq) * 1e-4
        pde_loss = F.mse_loss(pde_loss, torch.zeros_like(pde_loss)) * 1e-5
        
        # IC loss
        IC_loss = self.loss_fn(y_hat[:, :, :2, :], y[:, :, :2, :]) \
                + self.loss_fn(y_hat[:, :, :, :2], y[:, :, :, :2]) \
                + self.loss_fn(y_hat[:, :, :, -2:], y[:, :, :, -2:]) \
                + self.loss_fn(y_hat[:, :, -2:, :], y[:, :, -2:, :])
        
        # Combined loss
        loss = data_loss * self.weight_data + pde_loss * self.weight_pde + IC_loss * self.weight_IC
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_MSE_loss", data_loss, on_step=True, on_epoch=True)
        self.log("train_pde_loss", pde_loss, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        device = x.device
        y_hat = self.model(x[:,:self.input_channel])
        data_loss = self.loss_fn(y_hat, y)
        
        # Define frequency and v_0
        freq = torch.full((x.shape[0], 1, x.shape[2], x.shape[2]), 8, dtype=torch.float32, device=device)
        v_0 = torch.full((x.shape[0], 1, x.shape[2], x.shape[2]), 1.50, dtype=torch.float32, device=device)
        
        # PDE loss
        pde_loss = equation_fft(0.025, y_hat, 1.0 / (x[:, 2:3] ** 2), 1.0 / (v_0 ** 2), x[:, 0:1], x[:, 1:2], freq) * 1e-4
        pde_loss = F.mse_loss(pde_loss, torch.zeros_like(pde_loss)) * 1e-5

        # IC loss
        IC_loss = self.loss_fn(y_hat[:, :, :2, :], y[:, :, :2, :]) \
                + self.loss_fn(y_hat[:, :, :, :2], y[:, :, :, :2]) \
                + self.loss_fn(y_hat[:, :, :, -2:], y[:, :, :, -2:]) \
                + self.loss_fn(y_hat[:, :, -2:, :], y[:, :, -2:, :])

        # Combined loss
        loss = data_loss * self.weight_data + pde_loss * self.weight_pde + IC_loss * self.weight_IC
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_MSE_loss", data_loss, on_step=True, on_epoch=True)
        self.log("val_pde_loss", pde_loss, on_step=True, on_epoch=True)

    def configure_optimizers(self, lr=1e-3, step_size=100, gamma=0.9):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    
    

class UNO_np(LightningModule):
    #==================自定义CNO===================#
    
    def __init__(self, input_channel=3, N_layers=5, in_size=128,pde_weight=0, weight=[1,1,1],freq=8):
        super().__init__()

        # Replace the custom CNO model with CNO2d from neuraloperator
        self.model = UNO(
            in_channels=input_channel,      # Number of input channels.
            
            out_channels=2,        # Number of output channels
            
            hidden_channels=32,      
            
            n_layers=N_layers,            # Width of the hidden layers
            
            uno_out_channels=[32,64,64,64,32],
            
            uno_n_mode=[[32,32],[32,32],[32,32],[32,32],[32,32]],
            
            uno_scalings=[[1.0,1.0],[0.5,0.5],[1,1],[1,1],[2,2]]
        )
        
            
        self.loss_fn = torch.nn.MSELoss()
        #self.pde_loss = equation_fd(0.025, preds , 1.0/ (inputs[:,2:3] ** 2), 1.0 / (inputs[:,3:4] **2), inputs[:,0:1] , inputs[:,1:2] , v_0.to(device)) * 1e-5
        self.input_channel = input_channel
        self.weight_data,self.weight_pde,self.weight_IC = weight
        self.pde_weight = pde_weight
        self.freq=freq
        print(self.model)
        print("freq",self.freq)
        n_params = count_params(self.model)
        print(f'\nOur model has {n_params} parameters.')
        self.save_hyperparameters()
        
    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
 
        # Main forward, loss computation, and metrics goes here
        x, y = batch
        device = x.device
        # x, y = batch['x'], batch['y']
        y_hat = self.model(x[:,:self.input_channel])
        #loss_mse = self.loss_fn(y, y_hat)
        # Assuming min_value and freq is already defined
        freq = torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.freq, dtype=torch.float32)#8hz
        frequency = torch.tensor(freq,dtype=torch.float32).to(device)
        
        v_0=torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), 1.50, dtype=torch.float32)
        v_0=torch.tensor(v_0,dtype=torch.float32).to(device)

        #=========combine of the PDE and MSE loss===========#
        data_loss = self.loss_fn(y_hat,y)


        #pde_loss = equation_fd(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 #xinquan's
        #pde_loss = equation_fd_4th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4
        #pde_loss=equation_fft(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        pde_loss = equation_fd_8th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
          
        pde_loss=torch.sum(pde_loss ** 2)
        
        f = torch.zeros(pde_loss.shape, device=x.device)
        
        pde_loss = F.mse_loss(pde_loss, f)*1e-5        
        
        IC_loss= self.loss_fn(y_hat[:,:,:2,:],y[:,:,:2,:]) \
            + self.loss_fn(y_hat[:,:,:,:2],y[:,:,:,:2]) \
            + self.loss_fn(y_hat[:,:,:,-2:],y[:,:,:,-2:])\
            + self.loss_fn(y_hat[:,:,-2:,:],y[:,:,-2:,:])
          
    


        loss = data_loss *self.weight_data+ pde_loss *self.weight_pde+ IC_loss *self.weight_IC # 使用组合损失
        
        print("data_loss,pde_loss,IC_loss",data_loss,pde_loss,IC_loss)
        
        
        #loss=pde_loss
        relative_err_real = calculate_relative_loss((y_hat-y)[:,0:1], y[:,0:1], reduction='mean')
        
        relative_err_imag = calculate_relative_loss((y_hat-y)[:,1:2], y[:,1:2], reduction='mean')
        # pde_loss = equation_fd(0.025, y_hat[:,:,...] , 1.0/ (x[:,2:3,...] ** 2), 1.0 / (x[:,3:4,...] **2), x[:,0:1,...] , x[:,1:2,...] , x[:,4:5,...]) * 1e-5
        # pde_loss = 0.2 * torch.sum(pde_loss ** 2)
        #loss = loss_mse + 0.2 * torch.sum(pde_loss ** 2)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_MSE_loss", data_loss, on_step=True, on_epoch=True)
        self.log("train_pde_loss", pde_loss, on_step=True, on_epoch=True)
        self.log("train_relative_err_real", relative_err_real, on_step=True, on_epoch=True)
        self.log("train_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True)
        
        # Log learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log("learning_rate", lr, on_step=True, on_epoch=True)

        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        device = x.device
        # x, y = batch['x'], batch['y']
        y_hat = self.model(x[:,:self.input_channel])
        #loss_mse = self.loss_fn(y, y_hat)
        # Assuming min_value and freq is already defined
        freq = torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.freq, dtype=torch.float32)#8hz
        frequency = torch.tensor(freq,dtype=torch.float32).to(device)
        
        v_0=torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), 1.50, dtype=torch.float32)
        v_0=torch.tensor(v_0,dtype=torch.float32).to(device)
        
        #=========combine of the PDE and MSE loss===========#
        data_loss = self.loss_fn(y_hat,y)
        
        #pde_loss = equation_fd(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4

        #pde_loss = equation_fd_4th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        #pde_loss=equation_fft(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        pde_loss = equation_fd_8th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        pde_loss=torch.sum(pde_loss ** 2)
        f = torch.zeros(pde_loss.shape, device=x.device)
        pde_loss = F.mse_loss(pde_loss, f) *1e-5 

        #pde_loss = equation_fd_loss(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)
        
        IC_loss = self.loss_fn(y_hat[:,:,:2,:],y[:,:,:2,:]) \
            + self.loss_fn(y_hat[:,:,:,:2],y[:,:,:,:2]) + self.loss_fn(y_hat[:,:,-2:,:],y[:,:,-2:,:]) + self.loss_fn(y_hat[:,:,:,-2:],y[:,:,:,-2:])
            
        loss = data_loss *self.weight_data+ pde_loss *self.weight_pde+ IC_loss *self.weight_IC # 使用组合损失        
        print("data_loss,pde_loss,IC_loss",data_loss,pde_loss,IC_loss)
        
        relative_err_real = calculate_relative_loss((y_hat-y)[:,0:1], y[:,0:1], reduction='mean')
        
        relative_err_imag = calculate_relative_loss((y_hat-y)[:,1:2], y[:,1:2], reduction='mean')
        
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_MSE_loss", data_loss, on_step=True, on_epoch=True)
        self.log("val_pde_loss", pde_loss, on_step=True, on_epoch=True)
        self.log("val_relative_err_real", relative_err_real, on_step=True, on_epoch=True)
        self.log("val_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True)
        # Log learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log("val_learning_rate", lr, on_step=True, on_epoch=True)



    def configure_optimizers(self,lr=1e-3, step_size=100, gamma=0.9):
        # Return one or several optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
                       'interval': 'epoch',  # 或 'step' 对于每步更新
                       'frequency': 1}
        return [optimizer], [scheduler]
    

class SpectralConv2d(nn.Module):
    """2D Fourier layer. It does FFT, linear transform, and Inverse FFT."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
class FNO_v2(LightningModule):
    #==================自定义CNO===================#
    
    def __init__(self, input_channel=3,output_channel=2, mode1=5, mode2=4, width=6,weight=[1,1,1],freq=8,v0=1.50):
        super().__init__()
        
        self.modes1 = mode1
        self.modes2 = mode2
        self.width = width
        self.input_channel = input_channel
        self.output_channel = output_channel
        
        # self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x, y), x)
        # Add t channel
        self.fc0 = nn.Linear(self.input_channel, self.width) # input channel is 2: (a(x, y), x, t)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.output_channel)
        
            
        self.loss_fn = torch.nn.MSELoss()
        #self.pde_loss = equation_fd(0.025, preds , 1.0/ (inputs[:,2:3] ** 2), 1.0 / (inputs[:,3:4] **2), inputs[:,0:1] , inputs[:,1:2] , v_0.to(device)) * 1e-5
        self.weight_data,self.weight_pde,self.weight_IC = weight
   
        self.freq=freq
        self.v0=v0
        self.save_hyperparameters()
        
    def forward(self, x):
        
        x = x.permute(0, 2, 3, 1)
        
        batchsize = x.shape[0]
        
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
  
        return x
    
    def training_step(self, batch, batch_idx):
 
        # Main forward, loss computation, and metrics goes here
        x, y = batch
        device = x.device
        # x, y = batch['x'], batch['y']
        y_hat = self.forward(x[:,:self.input_channel])
        #loss_mse = self.loss_fn(y, y_hat)
        # Assuming min_value and freq is already defined
        freq = torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.freq, dtype=torch.float32)#8hz
        frequency = torch.tensor(freq,dtype=torch.float32).to(device)
        
        v_0=torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.v0, dtype=torch.float32)
        v_0=torch.tensor(v_0,dtype=torch.float32).to(device)

        #=========combine of the PDE and MSE loss===========#
        data_loss = self.loss_fn(y_hat,y)


        #pde_loss = equation_fd(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 #xinquan's
        #pde_loss = equation_fd_4th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4
        #pde_loss=equation_fft(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        pde_loss = equation_fd_8th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
          
        pde_loss=torch.sum(pde_loss ** 2)
        
        f = torch.zeros(pde_loss.shape, device=x.device)
        
        pde_loss = F.mse_loss(pde_loss, f)*1e-5        
        
        IC_loss= self.loss_fn(y_hat[:,:,:2,:],y[:,:,:2,:]) \
            + self.loss_fn(y_hat[:,:,:,:2],y[:,:,:,:2]) \
            + self.loss_fn(y_hat[:,:,:,-2:],y[:,:,:,-2:])\
            + self.loss_fn(y_hat[:,:,-2:,:],y[:,:,-2:,:])
          
    


        loss = data_loss *self.weight_data+ pde_loss *self.weight_pde+ IC_loss *self.weight_IC # 使用组合损失
        
        print("data_loss,pde_loss,IC_loss",data_loss,pde_loss,IC_loss)
        
        
        #loss=pde_loss
        relative_err_real = calculate_relative_loss((y_hat-y)[:,0:1], y[:,0:1], reduction='mean')
        
        relative_err_imag = calculate_relative_loss((y_hat-y)[:,1:2], y[:,1:2], reduction='mean')
        # pde_loss = equation_fd(0.025, y_hat[:,:,...] , 1.0/ (x[:,2:3,...] ** 2), 1.0 / (x[:,3:4,...] **2), x[:,0:1,...] , x[:,1:2,...] , x[:,4:5,...]) * 1e-5
        # pde_loss = 0.2 * torch.sum(pde_loss ** 2)
        #loss = loss_mse + 0.2 * torch.sum(pde_loss ** 2)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_MSE_loss", data_loss, on_step=True, on_epoch=True)
        self.log("train_pde_loss", pde_loss, on_step=True, on_epoch=True)
        self.log("train_relative_err_real", relative_err_real, on_step=True, on_epoch=True)
        self.log("train_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True)
        
        # Log learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log("learning_rate", lr, on_step=True, on_epoch=True)

        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        device = x.device
        # x, y = batch['x'], batch['y']
        y_hat = self.forward(x[:,:self.input_channel])
        #loss_mse = self.loss_fn(y, y_hat)
        # Assuming min_value and freq is already defined
        freq = torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.freq, dtype=torch.float32)#8hz
        frequency = torch.tensor(freq,dtype=torch.float32).to(device)
        
        v_0=torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.v0, dtype=torch.float32)
        v_0=torch.tensor(v_0,dtype=torch.float32).to(device)
        
        #=========combine of the PDE and MSE loss===========#
        data_loss = self.loss_fn(y_hat,y)
        
        #pde_loss = equation_fd(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4

        #pde_loss = equation_fd_4th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        #pde_loss=equation_fft(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        pde_loss = equation_fd_8th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        pde_loss=torch.sum(pde_loss ** 2)
        
        f = torch.zeros(pde_loss.shape, device=x.device)
        
        pde_loss = F.mse_loss(pde_loss, f) *1e-5 

        #pde_loss = equation_fd_loss(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)
        
        IC_loss = self.loss_fn(y_hat[:,:,:2,:],y[:,:,:2,:]) \
            + self.loss_fn(y_hat[:,:,:,:2],y[:,:,:,:2]) + self.loss_fn(y_hat[:,:,-2:,:],y[:,:,-2:,:]) + self.loss_fn(y_hat[:,:,:,-2:],y[:,:,:,-2:])
            
        loss = data_loss *self.weight_data+ pde_loss *self.weight_pde+ IC_loss *self.weight_IC # 使用组合损失        
        #print("data_loss,pde_loss,IC_loss",data_loss,pde_loss,IC_loss)
        
        relative_err_real = calculate_relative_loss((y_hat-y)[:,0:1], y[:,0:1], reduction='mean')
        
        relative_err_imag = calculate_relative_loss((y_hat-y)[:,1:2], y[:,1:2], reduction='mean')
        
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_MSE_loss", data_loss, on_step=True, on_epoch=True)
        self.log("val_pde_loss", pde_loss, on_step=True, on_epoch=True)
        self.log("val_relative_err_real", relative_err_real, on_step=True, on_epoch=True)
        self.log("val_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True)
        # Log learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']
        self.log("val_learning_rate", lr, on_step=True, on_epoch=True)



    def configure_optimizers(self,lr=1e-3, step_size=100, gamma=0.9):
        # Return one or several optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
                       'interval': 'epoch',  # 或 'step' 对于每步更新
                       'frequency': 1}
        return [optimizer], [scheduler]
    


class Unet(LightningModule):
    #==================自定义CNO===================#
    
    def __init__(self, input_channel=3, N_layers=5, N_res=4, N_res_neck=6,in_size=128,pde_weight=0, weight=[1,1,1],freq=8,v0=1.50):
        super().__init__()

        #self.model = smp.Segformer(encoder_name="resnet34",encoder_weights="imagenet",in_channels=input_channel,classes=2,activation="identity")  
        self.model = smp.UnetPlusPlus(encoder_name="resnet101",encoder_weights="imagenet",in_channels=input_channel,classes=2,activation="identity",decoder_attention_type="scse")         
            
        self.loss_fn = torch.nn.MSELoss()
        #self.pde_loss = equation_fd(0.025, preds , 1.0/ (inputs[:,2:3] ** 2), 1.0 / (inputs[:,3:4] **2), inputs[:,0:1] , inputs[:,1:2] , v_0.to(device)) * 1e-5
        self.input_channel = input_channel
        self.weight_data,self.weight_pde,self.weight_IC = weight
        self.pde_weight = pde_weight
        self.freq=freq
        self.v0=v0
        n_params = count_params(self.model)
        print(f'\nOur model has {n_params} parameters.')
        self.save_hyperparameters()
        
    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
 
        # Main forward, loss computation, and metrics goes here
        x, y = batch
        device = x.device
        # x, y = batch['x'], batch['y']
        y_hat = self.model(x[:,:self.input_channel])
        #loss_mse = self.loss_fn(y, y_hat)
        # Assuming min_value and freq is already defined
        freq = torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.freq, dtype=torch.float32)#8hz
        frequency = torch.tensor(freq,dtype=torch.float32).to(device)
        
        v_0=torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.v0, dtype=torch.float32)
        v_0=torch.tensor(v_0,dtype=torch.float32).to(device)

        #=========combine of the PDE and MSE loss===========#
        data_loss = self.loss_fn(y_hat,y)


        #pde_loss = equation_fd(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 #xinquan's
        #pde_loss = equation_fd_4th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4
        #pde_loss=equation_fft(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        pde_loss = equation_fd_8th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
          
        pde_loss=torch.sum(pde_loss ** 2)
        
        f = torch.zeros(pde_loss.shape, device=x.device)
        
        pde_loss = F.mse_loss(pde_loss, f)*1e-5        
        
        IC_loss= self.loss_fn(y_hat[:,:,:4,:],y[:,:,:4,:]) \
            + self.loss_fn(y_hat[:,:,:,:4],y[:,:,:,:4]) \
            + self.loss_fn(y_hat[:,:,:,-4:],y[:,:,:,-4:])\
            + self.loss_fn(y_hat[:,:,-4:,:],y[:,:,-4:,:])
          
    


        loss = data_loss *self.weight_data+ pde_loss *self.weight_pde+ IC_loss *self.weight_IC # 使用组合损失
        
        
        
        #loss=pde_loss
        relative_err_real = calculate_relative_loss((y_hat-y)[:,0:1], y[:,0:1], reduction='mean')
        
        relative_err_imag = calculate_relative_loss((y_hat-y)[:,1:2], y[:,1:2], reduction='mean')
        # pde_loss = equation_fd(0.025, y_hat[:,:,...] , 1.0/ (x[:,2:3,...] ** 2), 1.0 / (x[:,3:4,...] **2), x[:,0:1,...] , x[:,1:2,...] , x[:,4:5,...]) * 1e-5
        # pde_loss = 0.2 * torch.sum(pde_loss ** 2)
        #loss = loss_mse + 0.2 * torch.sum(pde_loss ** 2)
        self.log("train_loss", loss, on_step=True, on_epoch=True,sync_dist=True)
        self.log("train_MSE_loss", data_loss, on_step=True, on_epoch=True,sync_dist=True)
        self.log("train_pde_loss", pde_loss, on_step=True, on_epoch=True,sync_dist=True)
        self.log("train_relative_err_real", relative_err_real, on_step=True, on_epoch=True,sync_dist=True)
        self.log("train_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True,sync_dist=True)
        
        # Log learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']

        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        device = x.device
        # x, y = batch['x'], batch['y']
        y_hat = self.model(x[:,:self.input_channel])
        #loss_mse = self.loss_fn(y, y_hat)
        # Assuming min_value and freq is already defined
        freq = torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.freq, dtype=torch.float32)#8hz
        frequency = torch.tensor(freq,dtype=torch.float32).to(device)
        
        v_0=torch.full((x.shape[0], 1,x.shape[2], x.shape[2]), self.v0, dtype=torch.float32)
        v_0=torch.tensor(v_0,dtype=torch.float32).to(device)
        
        #=========combine of the PDE and MSE loss===========#
        data_loss = self.loss_fn(y_hat,y)
        
        #pde_loss = equation_fd(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4

        #pde_loss = equation_fd_4th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        #pde_loss=equation_fft(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        pde_loss = equation_fd_8th(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        #pde_loss=equation_fft_pad(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
        
        pde_loss=torch.sum(pde_loss ** 2)
        
        f = torch.zeros(pde_loss.shape, device=x.device)
        
        pde_loss = F.mse_loss(pde_loss, f) *1e-5 

        #pde_loss = equation_fd_loss(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)
        
        IC_loss = self.loss_fn(y_hat[:,:,:4,:],y[:,:,:4,:]) \
            + self.loss_fn(y_hat[:,:,:,:4],y[:,:,:,:4]) + self.loss_fn(y_hat[:,:,-4:,:],y[:,:,-4:,:]) + self.loss_fn(y_hat[:,:,:,-4:],y[:,:,:,-4:])
            
        loss = data_loss *self.weight_data+ pde_loss *self.weight_pde+ IC_loss *self.weight_IC # 使用组合损失        
        #print("data_loss,pde_loss,IC_loss",data_loss,pde_loss,IC_loss)
        
        relative_err_real = calculate_relative_loss((y_hat-y)[:,0:1], y[:,0:1], reduction='mean')
        
        relative_err_imag = calculate_relative_loss((y_hat-y)[:,1:2], y[:,1:2], reduction='mean')
        
        self.log("val_loss", loss, on_step=True, on_epoch=True,sync_dist=True)
        self.log("val_MSE_loss", data_loss, on_step=True, on_epoch=True,sync_dist=True)
        self.log("val_pde_loss", pde_loss, on_step=True, on_epoch=True,sync_dist=True)
        self.log("val_relative_err_real", relative_err_real, on_step=True, on_epoch=True,sync_dist=True)
        self.log("val_relative_err_imag", relative_err_imag, on_step=True, on_epoch=True,sync_dist=True)
        # Log learning rate
        optimizer = self.optimizers()
        lr = optimizer.param_groups[0]['lr']



    def configure_optimizers(self,lr=1e-3, step_size=100, gamma=0.9):
        # Return one or several optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma),
                       'interval': 'epoch',  # 或 'step' 对于每步更新
                       'frequency': 1}
        return [optimizer], [scheduler]

