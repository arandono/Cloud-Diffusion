import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
from RealFFT import fft
from RealFFT import ifft
from RealFFT import rfft
from RealFFT import irfft

    
class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 activation,
                 device=torch.device("cpu"),
                 dtype=torch.float
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation=activation
        self.dtype = dtype
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dtype=dtype)
        
        if in_channels != out_channels:
            self.skip_transformation = nn.Conv2d(in_channels, 
                                                 out_channels, 
                                                 kernel_size=1, 
                                                 stride=1, 
                                                 padding=0, 
                                                 dtype=dtype)
        else:
            self.skip_transformation = nn.Identity()
            
        self.to(device)

    def forward(self, X):
        
        X0 = self.skip_transformation(X)
        delta_X = self.conv1(X)
        delta_X = self.activation(delta_X)
        delta_X = self.conv2(delta_X)
        delta_X = self.activation(delta_X)
        
        X = X0 + delta_X
        
        return X 
    

class DownBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 activation,
                 device=torch.device("cpu"),
                 dtype=torch.float): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resblock1 = ResidualBlock(in_channels=in_channels, 
                                       out_channels=out_channels, 
                                       activation=activation, 
                                       dtype=dtype)
        self.resblock2 = ResidualBlock(in_channels=out_channels, 
                                       out_channels=out_channels, 
                                       activation=activation, 
                                       dtype=dtype)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.skips = []
        
        self.to(device)
        
    def forward(self, X):
        X = self.resblock1(X)
        self.skips.append(X)
        X = self.resblock2(X)
        self.skips.append(X)
        X = self.pool(X)
        
        return X


class UpBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 downblock_pair:DownBlock, 
                 activation, 
                 device=torch.device("cpu"),
                 dtype=torch.float
                ):
        super().__init__()
        self.downblock_pair = downblock_pair
        
        def upsample(X):
            ones = torch.ones(1,1,2,2, dtype=dtype, device=device).detach()
            return torch.kron(X,ones)
        
        self.upsample = upsample
        
        self.resblock1 = ResidualBlock(in_channels=in_channels + downblock_pair.out_channels, 
                                       out_channels=out_channels, 
                                       activation=activation, 
                                       dtype=dtype)
        self.resblock2 = ResidualBlock(in_channels=out_channels + downblock_pair.out_channels, 
                                       out_channels=out_channels, 
                                       activation=activation, 
                                       dtype=dtype)
        self.to(device)
    
    def forward(self, X):
        X = self.upsample(X)
        X = torch.cat((X,self.downblock_pair.skips.pop(-1)), dim=1)
        X = self.resblock1(X)
        X = torch.cat((X,self.downblock_pair.skips.pop(-1)), dim=1)
        X = self.resblock2(X)
        
        return X


class ComplexSinusoidalEmbedding(nn.Module):
    def __init__(self,  
                 batch_size,
                 out_channels,
                 N,
                 device=torch.device("cpu"),
                 dtype=torch.float):
        super().__init__()
        self.batch_size = batch_size
        self.out_channels = out_channels
        self.N = N
        self.device = device
        self.dtype = dtype
        self.to(device)
    
    def forward(self, noise_rates):
        
        if self.dtype == torch.cfloat:
            L = self.out_channels
            n = torch.arange(L, device=self.device)
            theta = torch.einsum("i,j->ij", noise_rates.squeeze(), 2*np.pi * torch.exp(n*(np.log(1000)/(L-1))))
            y = torch.complex(torch.cos(theta), torch.sin(theta)).unsqueeze(-1).unsqueeze(-1)
            embeddings = y.repeat(1,1,self.N,self.N)
        
        else:
            L = self.out_channels//2
            n = torch.arange(L, device=self.device)
            theta = torch.einsum("i,j->ij", noise_rates.squeeze(), 2*np.pi * torch.exp(n*(np.log(1000)/(L-1))))
            s = torch.sin(theta)
            c = torch.cos(theta)
            y = torch.cat((s,c), dim=1).unsqueeze(-1).unsqueeze(-1)
            
            embeddings = y.repeat(1,1,self.N, self.N)
            
        return embeddings
    
class SinusoidalEmbedding(nn.Module):
    def __init__(self,  
                 batch_size,
                 out_channels,
                 N,
                 device=torch.device("cpu"),
                 dtype=torch.float):
        super().__init__()
        self.batch_size = batch_size
        self.out_channels = out_channels
        self.N = N
        self.device = device
        self.dtype = dtype
        self.to(device)
    
    def forward(self, noise_rates):

        L = self.out_channels//2
        n = torch.arange(L, device=self.device)
        theta = torch.einsum("i,j->ij", noise_rates.squeeze(), 2*np.pi * torch.exp(n*(np.log(1000)/(L-1))))
        s = torch.sin(theta)
        c = torch.cos(theta)
        y = torch.cat((s,c), dim=1).unsqueeze(-1).unsqueeze(-1)

        embeddings = y.repeat(1,1,self.N, self.N)
            
        return embeddings


class UNet(nn.Module):
    """
    This is the core UNet model that is used to predict the noise tensor from noisy images.
    For purposes of comparison it is designed for both traditional diffusion and cloud diffusion.
    Choosing dtype=torch.float will take a set of noisy real images and predict the real noise tensors.
    Choosing dtype=torch.cfloat will take the Fourier transform of noisy complex image pairs and predict the complex noise tensors.
    """
    
    def __init__(self, 
                 batch_size,
                 N,
                 color_channels,
                 activation, 
                 dtype, 
                 device,
                 scale_factor,
                 fft_norm, 
                ):
        super().__init__()
        self.batch_size = batch_size
        self.N = N
        self.color_channels=color_channels
        self.device = device
        self.dtype = dtype
        self.activation = activation
        self.scale_factor = scale_factor
        self.fft_norm = fft_norm
        
        self.embedding = SinusoidalEmbedding(batch_size=batch_size,
                                                        out_channels= 1*scale_factor,
                                                        N=N,
                                                        device=device,
                                                        dtype=dtype)
        
        self.first_convolution = torch.nn.Conv2d(in_channels=self.color_channels, 
                                                 out_channels= 1*scale_factor, 
                                                 kernel_size=3, 
                                                 stride=1, 
                                                 padding=1, 
                                                 device=device,
                                                 dtype=dtype)
        
        self.down_block_1 = DownBlock(in_channels= 2*scale_factor, 
                                      out_channels= 1*scale_factor, 
                                      activation=activation,
                                      device=device,
                                      dtype=dtype)
        
        self.down_block_2 = DownBlock(in_channels= 1*scale_factor, 
                                      out_channels= 2*scale_factor, 
                                      activation=activation,
                                      device=device,
                                      dtype=dtype)
        
        self.down_block_3 = DownBlock(in_channels= 2*scale_factor, 
                                      out_channels= 3*scale_factor, 
                                      activation=activation,
                                      device=device,
                                      dtype=dtype)
        
        self.residual_block_A = ResidualBlock(in_channels= 3*scale_factor, 
                                              out_channels= 4*scale_factor, 
                                              activation=activation,
                                              device=device,
                                              dtype=dtype)
        
        self.residual_block_B = ResidualBlock(in_channels= 4*scale_factor, 
                                              out_channels= 4*scale_factor, 
                                              activation=activation,
                                              device=device,
                                              dtype=dtype)
        
        self.up_block_3 = UpBlock(in_channels= 4*scale_factor, 
                                  out_channels= 3*scale_factor, 
                                  downblock_pair=self.down_block_3, 
                                  activation=activation,
                                  device=device,
                                  dtype=dtype)
                                 
        self.up_block_2 = UpBlock(in_channels= 3*scale_factor, 
                                  out_channels= 2*scale_factor, 
                                  downblock_pair=self.down_block_2, 
                                  activation=activation,
                                  device=device,
                                  dtype=dtype)
                         
        self.up_block_1 = UpBlock(in_channels= 2*scale_factor, 
                                  out_channels= 1*scale_factor, 
                                  downblock_pair=self.down_block_1, 
                                  activation=activation,
                                  device=device,
                                  dtype=dtype)
        
        self.last_convolution = torch.nn.Conv2d(in_channels= 1*scale_factor, 
                                                out_channels=self.color_channels, 
                                                kernel_size=3, 
                                                stride=1, 
                                                padding=1, 
                                                device=device,
                                                dtype=dtype)
        
        self.to(device)
        
        
    def forward(self, X, noise_rates):

        X = self.first_convolution(X)
        embeddings = self.embedding(noise_rates)
        X = torch.cat((X, embeddings), dim=1)

        X = self.down_block_1(X)
        X = self.down_block_2(X)
        X = self.down_block_3(X)

        X = self.residual_block_A(X)
        X = self.residual_block_B(X)

        X = self.up_block_3(X)
        X = self.up_block_2(X)
        X = self.up_block_1(X)

        predicted_Noise = self.last_convolution(X)
        
        return predicted_Noise
                                 
        
        
        
        
        
        
        
    
