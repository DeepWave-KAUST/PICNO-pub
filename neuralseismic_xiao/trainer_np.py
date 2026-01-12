from pytorch_lightning import LightningModule
import torch
from torch.utils.data import DataLoader
from .fourier_2d import FNO2d
from .unet import UNet
from .resnet import Resnet18
from .utilis import calculate_relative_loss
from .utilis import count_params
from neuralseismic_xiao import FNO2d, equation_fd,equation_fd_4th,equation_fft,equation_fft_pad
from neuralop.models import FNO,UNO
from .CNOModule import CNO
import torch.nn.functional as F
from neuralop.models import FNO2d as FNO_np
#from neuralop.models import SFNO as SFNO_np
class FNO_NP(LightningModule):
    #==================自定义CNO===================#
    
    def __init__(self, input_channel=3, N_layers=5, in_size=128, pde_weight=0, weight=[1,1,1]):
        super().__init__()

        # Replace the custom CNO model with FNO2d from neuraloperator
        # self.model = SFNO_np(
        #     n_modes_height=48,           # Number of Fourier modes in the first dimension
        #     n_modes_width=48,           # Number of Fourier modes in the second dimension
        #     hidden_channels=64,            # Width of the hidden layers
        #     in_channels=input_channel,   # Number of input channels
        #     out_channels=2,        # Number of output channels
        #     n_layers=N_layers,
        #     lifting_channels=128,
        #     projection_channels=128
        # )
        self.model = SFNO_np(
           n_modes=[48,48],           # Number of Fourier modes in the first dimension
           in_channels=input_channel,   # Number of input channels
           out_channels=2,        # Number of output channels 
           hidden_channels=128,
           n_layers=N_layers,
           positional_embedding=None
        )
        
        self.loss_fn = torch.nn.MSELoss()
        #self.pde_loss = equation_fd(0.025, preds , 1.0/ (inputs[:,2:3] ** 2), 1.0 / (inputs[:,3:4] **2), inputs[:,0:1] , inputs[:,1:2] , v_0.to(device)) * 1e-5
        self.input_channel = input_channel
        self.weight_data,self.weight_pde,self.weight_IC = weight
        self.pde_weight = pde_weight
        #print(self.model)
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
        pde_loss=equation_fft(0.025, y_hat , 1.0/ (x[:,2:3] ** 2), 1.0 / (v_0 **2), x[:,0:1] , x[:,1:2] , frequency)*1e-4 
          
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
