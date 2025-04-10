import torch
import torch.nn as nn
import torch.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import linregress
import torchvision.transforms as transforms
import data
import unet
import helpers

class ModelVariables:
    """
    A container for storing and easily accessing
    all of the variables that will be needed during training
    """
    def __init__(self):

        self.dataloader=None
        self.mu=None
        self.std=None
        self.A=None
        self.Delta=None
        self.batch_size=None
        self.color_channels=None
        self.N=None
        self.scale_factor=None
        self.device=None
        self.n_epochs=None
        self.T=None
        self.dtype=None
        self.fft_norm=None
        self.activation=None
        self.learning_rate=None
        self.betas=None,
        self.weight_decay=None
        self.diffusion_schedule=None
        self.n_train_batches=None
        
def cosine_diffusion_schedule(diffusion_times):
    signal_rates = torch.cos(np.pi/2 * diffusion_times)
    noise_rates = torch.sin(np.pi/2 * diffusion_times)
    
    return signal_rates, noise_rates

def squared_time_cosine_diffusion_schedule(diffusion_times):
    signal_rates = torch.cos(np.pi/2 * diffusion_times**2)
    noise_rates = torch.sin(np.pi/2 * diffusion_times**2)
    
    return signal_rates, noise_rates

class MeanAbsoluteLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, Noise, predicted_Noise):
        z = Noise - predicted_Noise
        batch_losses = torch.einsum("bjkl,bjkl->b", z, torch.conj(z))
        mean_loss = batch_losses.real.mean()
        
        return mean_loss
    

        
        
class CloudDiffusionModel:
    def __init__(self, mv: ModelVariables):
        
        network=unet.UNet(batch_size = mv.batch_size,
                          N = mv.N,
                          device = mv.device,
                          dtype = mv.dtype,
                          activation = mv.activation,
                          scale_factor = mv.scale_factor,
                          fft_norm = mv.fft_norm)
        
        optimizer=torch.optim.Adam(network.parameters(), 
                                    lr=mv.learning_rate, 
                                    betas=mv.betas)
        
        noise_generator = helpers.NoiseGenerator(mv.N)
        
        self.mv = mv
        self.network=network
        self.optimizer=optimizer
        self.noise_generator = noise_generator
        self.loss_function = MeanAbsoluteLoss()
        
        print(f"CloudDiffusionModel Parameters = {self.count_parameters()}")
        
        
    def count_parameters(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)
    
    
    def normalize(self, z):
        if z.dtype == torch.cfloat:
            mu = torch.complex(self.mv.mu, self.mv.mu)
        else:
            mu = self.mv.mu

        std = self.mv.std

        return (z-mu)/std

    def unnormalize(self, z):
        if z.dtype == torch.cfloat:
            mu = torch.complex(self.mv.mu, self.mv.mu)
        else:
            mu = self.mv.mu

        std = self.mv.std

        return std*z + mu    
    
    
    def visualize_batch(self,
                        z, 
                        noise, 
                        predicted_noise, 
                        input_images_1, 
                        input_images_2, 
                        predicted_images_1, 
                        predicted_images_2):

        fig, axs = plt.subplots(nrows=6, ncols=5, figsize=(10,12))
        for i in range(3):
            axs[i][0].imshow(z.detach().real.squeeze()[i])
            axs[i][1].imshow(noise.detach().real.squeeze()[i], vmin=-5, vmax=5)
            axs[i][2].imshow(predicted_noise.detach().real.squeeze()[i], vmin=-5, vmax=5)
            axs[i][3].imshow(input_images_1.detach().squeeze()[i], vmin=0, vmax=1)
            axs[i][4].imshow(predicted_images_1.detach().squeeze()[i], vmin=0, vmax=1)

            axs[i+3][0].imshow(z.detach().imag.squeeze()[i])
            axs[i+3][1].imshow(noise.detach().imag.squeeze()[i], vmin=-5, vmax=5)
            axs[i+3][2].imshow(predicted_noise.detach().imag.squeeze()[i], vmin=-5, vmax=5)
            axs[i+3][3].imshow(input_images_2.detach().squeeze()[i], vmin=0, vmax=1)
            axs[i+3][4].imshow(predicted_images_2.detach().squeeze()[i], vmin=0, vmax=1)

        axs[0][0].set_title("Noisy Input")
        axs[0][1].set_title("Noise")
        axs[0][2].set_title("Pred Noise")
        axs[0][3].set_title("Input Image")
        axs[0][4].set_title("Pred Image")

        for row in range(6):
            for col in range(5):
                axs[row][col].set_xticks([])
                axs[row][col].set_yticks([])

        plt.show()
        
    
    def train_it(self):

        mv = self.mv
        
        # Record losses
        losses = []
        
        n_epochs = mv.n_epochs
        n_train_batches = mv.n_train_batches
        
        for epoch in range(n_epochs):
            # Create two data iterators from the a single dataloader
            data_iter_1 = iter(mv.dataloader)
            data_iter_2 = iter(mv.dataloader)
            
            # Get total number of batches
            if mv.n_train_batches==None:
                n_batches = len(data_iter_1)
            else:
                n_batches = mv.n_train_batches
            
            # Record start time for epoch
            t0_epoch = time.time()
            
            for batch in range(n_batches):
                
                # Record start time
                t0_batch = time.time()
                
                # Clear the gradients
                self.optimizer.zero_grad()

                # Get the image batches for the real and imaginary parts of input
                input_images_1, _ = next(data_iter_1)
                input_images_2, _ = next(data_iter_2)

                # Check for duplicate images in batch_1 and batch_2 and reshuffle if necessary
                input_images_1, input_images_2 = data.reshuffle_duplicates(input_images_1, input_images_2)

                # Normalize the images on a pixel-by-pixel basis
                normalized_images_1 = self.normalize(input_images_1)
                normalized_images_2 = self.normalize(input_images_2)

                # Create the complex input vector (in position space)
                z0 = torch.complex(normalized_images_1, normalized_images_2)

                # Create the complex input vector (in Fourier space)
                Z0 = fft.fftshift(fft.fft2(z0, norm=mv.fft_norm))

                # Generate the noise tensors (noise:position space, Noise:Fourier space)
                noise, Noise = self.noise_generator.generate_noise(Delta=mv.Delta, 
                                                                    A=mv.A, 
                                                                    batch_size=mv.batch_size, 
                                                                    color_channels=mv.color_channels, 
                                                                    fft_norm=mv.fft_norm)

                # Diffusion Times (normalized to range from 0.0 to 1.0)
                diffusion_times = torch.randint(1, mv.T, size=(mv.batch_size,))/mv.T
                diffusion_times = diffusion_times.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                # Get noise and signal rates
                signal_rates, noise_rates = mv.diffusion_schedule(diffusion_times)

                # Generate Noisy Images (z:position space, Z:Fourier space)
                z = signal_rates * z0 + noise_rates * noise
                Z = signal_rates * Z0 + noise_rates * Noise

                # Normal Approach:
                # PREDICT NOISES! (noise:position space, Noise:Fourier space)
                predicted_noise, predicted_Noise = self.network(Z, noise_rates)
                
                # Alternative approach: use unet to predict the images, not the noise
                # predicted_z0, predicted_Z0 = self.network(Z, noise_rates)

                # Normal approach:
                # Get predicted images (in Fourier space)
                predicted_z0 = (z - noise_rates * predicted_noise)/signal_rates
                predicted_Z0 = (Z - noise_rates * predicted_Noise)/signal_rates
                
                # Alternative approach: use predicted images from unet to get predicted noise
                # predicted_noise = (z - signal_rates * predicted_z0)/noise_rates
                # predicted_Noise = (Z - signal_rates * predicted_Z0)/noise_rates

                # Get unnormnalized predicted images (in position space)
                predicted_images_1 = self.unnormalize(predicted_z0.real)
                predicted_images_2 = self.unnormalize(predicted_z0.imag)

                # Calculate the loss
                loss = self.loss_function(Noise, predicted_Noise)
                losses.append(loss.item())

                # Perform a backward pass and optimizer gradient descent step
                loss.backward()
                self.optimizer.step()
                
                # Print every n batches
                print_every = 5
                if (batch+1)%print_every == 0:
                    print(f"Batch {batch+1}/{n_batches} | loss = {loss} | time elapsed = {time.time()-t0_batch}")
                    
                    self.visualize_batch(z=z, 
                                        noise=noise, 
                                        predicted_noise=predicted_noise, 
                                        input_images_1=input_images_1, 
                                        input_images_2=input_images_2, 
                                        predicted_images_1=predicted_images_1, 
                                        predicted_images_2=predicted_images_2)
            
        return losses
    