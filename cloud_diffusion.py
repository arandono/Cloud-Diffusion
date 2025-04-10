import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import linregress
import torchvision.transforms as transforms
import data
import unet
import helpers
from RealFFT import fft
from RealFFT import ifft
from RealFFT import rfft
from RealFFT import irfft
from IPython.display import clear_output

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
        self.eigs=None
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
        self.real_fourier_space=False
        self.cmap=None
        # The model type can be: "white", "cloud", or "whitening" 
        self.model_type=None
        self.whitening_epsilon=None

def cosine_diffusion_schedule(diffusion_times):
    signal_rates = torch.cos(np.pi/2 * diffusion_times)
    noise_rates = torch.sin(np.pi/2 * diffusion_times)
    
    return signal_rates, noise_rates


def offset_cosine_diffusion_schedule(diffusion_times):
    
    min_signal_rates = .02
    max_signal_rates = .95
    
    start_angle = np.arccos(max_signal_rates)
    end_angle = np.arccos(min_signal_rates)
    
    diffusion_angles = start_angle + (end_angle - start_angle)*diffusion_times
    
    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)
    
    return signal_rates, noise_rates


def squared_time_cosine_diffusion_schedule(diffusion_times):
    signal_rates = torch.cos(np.pi/2 * diffusion_times**2)
    noise_rates = torch.sin(np.pi/2 * diffusion_times**2)
    
    return signal_rates, noise_rates


class MeanAbsoluteLoss(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.to(device)
    
    def forward(self, noise, predicted_noise):
        
        x = noise - predicted_noise
        batch_losses = torch.einsum("bjkl,bjkl->b", x, x)
        mean_loss = batch_losses.mean()

        return mean_loss
    
    
class L2Loss(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        super().__init__()
        self.to(device)
    
    def forward(self, noise, predicted_noise):
        batch_size = noise.shape[0]
        x = noise - predicted_noise
        batch_losses = torch.einsum("bcij,bcij", x, x)/batch_size

        return batch_losses
    

class FlatAbsoluteLoss(nn.Module):
    """
    Normalizes the noise tensors in Fourier space using d^Delta before calculating absolute difference.
    Should input the position space tensors, NOT the Fourier space tensors. 
    """
    
    def __init__(self, Delta, dist, epsilon=.05, device=torch.device("cpu")):
        super().__init__()
        self.Delta = Delta
        self.dist = dist.to(device)
        self.epsilon = epsilon
        self.device=device
        self.to(device)
    
    def forward(self, noise, predicted_noise):
        delta_noise = noise - predicted_noise
        Delta_Noise = fft(delta_noise)
        Flat_Delta_Noise = (self.dist+self.epsilon)**self.Delta * Delta_Noise
        
        batch_losses_real = torch.einsum("bjkl,bjkl->b", Flat_Delta_Noise.real, Flat_Delta_Noise.real)
        batch_losses_imag = torch.einsum("bjkl,bjkl->b", Flat_Delta_Noise.imag, Flat_Delta_Noise.imag)
        
        mean_loss = (batch_losses_real + batch_losses_imag).mean()

        return mean_loss
    

class Mahalanobis_Loss(nn.Module):
    """
    Computes the Mahalanobis Loss of monochrome noise predictions
    """
    
    def __init__(self, Delta, dist, real_fourier_space, epsilon=.05, device=torch.device("cpu")):
        super().__init__()
        self.Delta = Delta
        self.dist = dist.to(device)
        self.epsilon = epsilon
        self.device=device
        self.real_fourier_space = real_fourier_space
        self.to(device)
        
        
    def forward(self, noise, predicted_noise):
        """
        Computes Mahalanobis Loss
        realFourierSpace=True means tensors are already transformed to Real Fourier Space
        epsilon is a regaularization parameter that combines the Maha Squared Distance with the L2 norm
        """
        epsilon = self.epsilon
        batch_size = noise.shape[0]
        
        if self.real_fourier_space:
            X = (noise - predicted_noise).squeeze()
        else:
            X = rfft((noise-predicted_noise).squeeze())
        
        k2D = (self.dist.squeeze()+epsilon)**(2*self.Delta)
        
        maha_loss = torch.einsum("bij,ij,bij", X, k2D, X)/batch_size #Maha Squared Distance
        
        #loss2 = torch.einsum("bij,bij", X, X)/batch_size #L2 Norm
        #maha_loss = (1-epsilon)*maha_loss + epsilon*loss2

        return maha_loss

    
class KL_Loss_Color(nn.Module):
    """
    Normalizes the noise tensors in Fourier space using d^Delta before calculating absolute difference.
    Should input the position space tensors, NOT the Fourier space tensors. 
    
    Uses, eigs, the output of linalg.eigs(sigma_color). This has the form (L,V)=eigs. 
    sigma_color is the 9x9 variance tensor of the image set. 
    """
    
    def __init__(self, Delta, dist, eigs, epsilon=.05, device=torch.device("cpu")):
        super().__init__()
        self.Delta = Delta
        self.dist = dist.to(device)
        self.epsilon = epsilon
        self.device=device
        self.eigs = eigs
        self.to(device)
        self.L = eigs[0].real
        self.V = eigs[1].real
        self.V_inverse = torch.inverse(eigs[1].real)
        
    
    def forward(self, noise, predicted_noise):
        epsilon = self.epsilon
        L = self.L
        V = self.V
        V_inverse = self.V_inverse
        batch_size = noise.shape[0]
        
        x = noise - predicted_noise
        
        # Invert the color variance tensor (first diagonalize it and then divide out the eigenvalues)
        x = torch.einsum("cC,bCij->bcij", V_inverse, x)
        x = torch.einsum("c,bcij->bcij", 1/torch.sqrt(L), x)
        
        # Invert the distance variance tensor in Fourier space
        Z = fft(x)
        Z = (self.dist+epsilon)**self.Delta * Z
        
        # Note: mps doesn't support some matrix operations for cfloat right now so this is a workaround
        loss = torch.sum(Z.real**2 + Z.imag**2)/batch_size

        return loss
    

class KL_Loss(nn.Module):
    """
    Normalizes the noise tensors in Fourier space using d^Delta before calculating absolute difference.
    Should input the position space tensors, NOT the Fourier space tensors. 
    """
    
    def __init__(self, flattened_sigma_noise_inverse, alpha=.1, device=torch.device("cpu")):
        super().__init__()
        self.flattened_sigma_noise_inverse = flattened_sigma_noise_inverse.to(device)
        self.device=device
        self.alpha = alpha
    
    def forward(self, noise, predicted_noise):
        batch_size = noise.shape[0]
        sigma_inverse = self.flattened_sigma_noise_inverse
        delta_noise = torch.flatten((noise - predicted_noise).squeeze(), start_dim=1)
        
        loss_1 = torch.einsum("bi,bi", delta_noise, delta_noise)/batch_size
        loss_2 = torch.einsum("bi,ij,bj", delta_noise, sigma_inverse, delta_noise)/batch_size
        
        loss = self.alpha*loss_1 + (1-self.alpha)*loss_2

        return loss
    

def movedim(x):
    color_channels = x.squeeze().shape[0]
    if color_channels == 3:
        return torch.movedim(x, (0,1,2), (2,0,1))
    else:
        return x


class CloudDiffusionModel:
    def __init__(self, mv: ModelVariables):
        
        network=unet.UNet(batch_size = mv.batch_size,
                          N = mv.N,
                          color_channels=mv.color_channels,
                          device = mv.device,
                          dtype = torch.float,
                          activation = mv.activation,
                          scale_factor = mv.scale_factor,
                          fft_norm = mv.fft_norm)
        
        optimizer=torch.optim.AdamW(network.parameters(), 
                                    lr=mv.max_learning_rate, 
                                    betas=mv.betas,
                                    weight_decay=mv.weight_decay)
        
        noise_generator = helpers.NoiseGenerator(mv.N, device=mv.device)
        
        self.mv = mv
        self.network=network
        self.optimizer=optimizer
        self.noise_generator = noise_generator
        self.loss_function = mv.loss_function
        
        print(f"Model Parameters = {self.count_parameters()}")
        
        
    def count_parameters(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)
    
    def normalize(self, x, mu=None, std=None):
        if mu==None and std==None:
            return helpers.normalize(x, self.mv.mu, self.mv.std)
        else:
            return helpers.normalize(x, mu, std)

    def denormalize(self, x, mu=None, std=None):
        if mu==None and std==None:
            return helpers.denormalize(x, self.mv.mu, self.mv.std) 
        else:
            return helpers.denormalize(x, mu, std)
    
    def visualize_batch(self,
                        x0,
                        x,
                        noise, 
                        predicted_noise,
                        predicted_x0,
                        cmap
                       ):

        with torch.no_grad():
            # First detach, squeeze, and denormalize everything for easier visualization
            color_channels = x.shape[1]
            
            input_images = x0.detach().squeeze().to("cpu")
            noisy_images = x.detach().squeeze().to("cpu")
            noise = noise.detach().squeeze().to("cpu")
            predicted_noise = predicted_noise.detach().squeeze().to("cpu")
            predicted_x0 = predicted_x0.detach().squeeze().to("cpu")
            
            if self.mv.real_fourier_space:
                noisy_images = irfft(noisy_images)
                input_images = irfft(input_images)
                noise = irfft(noise)
                predicted_noise = irfft(predicted_noise)
                predicted_x0 = irfft(predicted_x0)
                
            noisy_images = self.denormalize(noisy_images)
            input_images = self.denormalize(input_images)
            noise = self.denormalize(noise)
            predicted_noise = self.denormalize(predicted_noise)
            predicted_x0 = self.denormalize(predicted_x0)

            fig, axs = plt.subplots(nrows=6, ncols=5, figsize=(10,12))

            for i in range(6):
                axs[i][0].imshow(torch.clamp(movedim(noisy_images[i]), min=0.0, max=1.0), cmap=cmap)
                axs[i][1].imshow(torch.clamp(movedim(input_images[i]), min=0.0, max=1.0), cmap=cmap)
                axs[i][2].imshow(torch.clamp(movedim(predicted_x0[i]), min=0.0, max=1.0), cmap=cmap)
                axs[i][3].imshow(torch.clamp(movedim(noise[i]), min=0.0, max=1.0), cmap=cmap)
                axs[i][4].imshow(torch.clamp(movedim(predicted_noise[i]), min=0.0, max=1.0), cmap=cmap)
                
            axs[0][0].set_title("Noisy Images")
            axs[0][1].set_title("Input Image")
            axs[0][2].set_title("Pred Image")
            axs[0][3].set_title("Noise")
            axs[0][4].set_title("Pred Noise")
            
            plt.setp(axs, xticks=[], yticks=[])
            plt.show()
        
    
    def train_it(self, n_epochs=None, n_train_batches=None):
        
        mv = self.mv
        
        if n_epochs==None:
            n_epochs = mv.n_epochs
            
        if n_train_batches==None:
            n_train_batches = mv.n_train_batches
        
        if n_train_batches==None:
            n_steps = n_epochs * len(mv.dataloader)
        else:
            n_steps = n_train_batches
        
        # Record losses
        losses = []
        
        t0 = time.time()
        n=0
        
        for epoch in range(n_epochs):
            # Create two data iterators from the a single dataloader
            data_iter = iter(mv.dataloader)
            
            # Get total number of batches
            if mv.n_train_batches==None:
                n_batches = len(data_iter)
            else:
                n_batches = mv.n_train_batches
            
            # Record start time for epoch
            t0_epoch = time.time()
            
            # Cycle through training batches
            for batch in range(n_batches):
                
                # Record start time
                t0_batch = time.time()
                
                # Clear the gradients
                self.optimizer.zero_grad()

                # Get the image batches
                input_images = next(data_iter)[0].to(mv.device)

                # Get input images
                if mv.model_type=="cloud" or mv.model_type=="white":
                    # For white and cloud models, normalize images on a pixel basis
                    x0 = self.normalize(input_images)
                    
                elif mv.model_type=="whitening":
                    # For whitening models, first whiten the images
                    # Note that this also normalizes the output images
                    x0 = helpers.whiten(input_images, 
                                        mu=mv.mu, 
                                        std=mv.std, 
                                        Delta=mv.Delta, 
                                        eps=mv.whitening_epsilon, 
                                        noise_generator=mv.noise_generator, 
                                        mu_whitened=mv.mu_whitened, 
                                        std_whitened=mv.std_whitened)

                # Generate the noise tensors (standard normal noise if Delta=0, else cloud noise)
                if mv.model_type == "cloud":
                    # Generate Monochrome or Color Cloud Noise for cloud models
                    if mv.color_channels==1:
                        noise = self.noise_generator.generate_monochrome_noise(Delta=mv.Delta, 
                                                                               A=mv.A, 
                                                                               batch_size=mv.batch_size,  
                                                                               fft_norm=mv.fft_norm,
                                                                               device=mv.device
                                                                              )[0].real
                    else:
                        noise = self.noise_generator.generate_color_noise(Delta=mv.Delta, 
                                                                          A=mv.A,
                                                                          eigs=mv.eigs,
                                                                          batch_size=mv.batch_size,  
                                                                          fft_norm=mv.fft_norm,
                                                                          device=mv.device
                                                                         )[0].real
                elif mv.model_type == "white" or mv.model_type == "whitening":
                    # Generate white noise for white or whitening models
                    noise = torch.randn(mv.batch_size, mv.color_channels, mv.N, mv.N, device=mv.device)
                
                if mv.real_fourier_space:
                    x0 = rfft(x0)
                    noise = rfft(noise)
                
                # Diffusion Times (normalized to range from 0.0 to 1.0)
                diffusion_times = torch.randint(1, mv.T, size=(mv.batch_size,), device=mv.device)/mv.T
                diffusion_times = diffusion_times.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

                # Get noise and signal rates
                signal_rates, noise_rates = mv.diffusion_schedule(diffusion_times)

                # Generate Noisy Images
                x = signal_rates * x0 + noise_rates * noise

                # PREDICT NOISES!
                predicted_noise = self.network(x, noise_rates)
                
                # Get predicted images
                predicted_x0 = (x - noise_rates * predicted_noise)/signal_rates
                
                
                # Compute loss
                loss = self.loss_function(noise, predicted_noise)
                losses.append(loss.item())

                # Adjust learning rate (using a linearly decreasing lr schedule)
                for g in self.optimizer.param_groups:
                    learning_rate = mv.max_learning_rate + n/n_steps * (mv.min_learning_rate - mv.max_learning_rate)
                    g["lr"] = learning_rate
                
                # Perform a backward pass and optimizer gradient descent step
                loss.backward()
                self.optimizer.step()
                
                # Clear the gradients
                self.optimizer.zero_grad()
                
                # Print every n batches
                with torch.no_grad():
                    print_every = 10
                    if (batch+1)%print_every == 0:
                        print(f"Epoch: {epoch+1}/{n_epochs} | Batch: {batch+1}/{n_batches} | Loss: {loss}")
                        print(f"Total Time (minutes): {(time.time()-t0)/60} | Time per batch: {time.time()-t0_batch}")                   
                        
                        self.visualize_batch(x0=x0,
                                             x=x,
                                             noise=noise, 
                                             predicted_noise=predicted_noise,
                                             predicted_x0=predicted_x0,
                                             cmap=mv.cmap
                                            )

                        x = np.arange(0, len(losses))
                        plt.scatter(x, losses)
                        plt.show()

                        clear_output(wait=True)

                n+=1
            
        return losses
    

def show_25(model, image_batch, cmap="gray"):
    """
    Helper function that displays 25 images in a square grid
    """
    with torch.no_grad():
        
        fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(10,12))

        denormed_images = model.denormalize(image_batch.squeeze()).to("cpu")

        for i in range(5):
                axs[i][0].imshow(torch.clamp(movedim(denormed_images[i]), min=0.0, max=1.0), cmap=cmap)
                axs[i][1].imshow(torch.clamp(movedim(denormed_images[i+5]), min=0.0, max=1.0), cmap=cmap)
                axs[i][2].imshow(torch.clamp(movedim(denormed_images[i+10]), min=0.0, max=1.0), cmap=cmap)
                axs[i][3].imshow(torch.clamp(movedim(denormed_images[i+15]), min=0.0, max=1.0), cmap=cmap)
                axs[i][4].imshow(torch.clamp(movedim(denormed_images[i+20]), min=0.0, max=1.0), cmap=cmap)

        plt.setp(axs, xticks=[], yticks=[])
        plt.show()


def reverse_diffusion(model, 
                      diffusion_steps=1000, 
                      start_step=0, 
                      print_every=100,
                      stochasticity=0, 
                      color_channels=1,
                      cmap="gray",
                      num_sequences=4):
    """
    Reverses the diffusion procedure to generate images
    """
    device = model.mv.device
    
    with torch.no_grad():
        mv = model.mv
        sequences = torch.zeros(1, num_sequences, color_channels, mv.N, mv.N, device=mv.device)
        
        if mv.model_type=="cloud":
            # Generate cloud noise if Delta≠0
            
            if mv.color_channels==1:
                initial_noise = model.noise_generator.generate_monochrome_noise(Delta=mv.Delta, 
                                                                       A=mv.A, 
                                                                       batch_size=mv.batch_size,  
                                                                       fft_norm=mv.fft_norm,
                                                                       device=mv.device
                                                                      )[0].real
            else:
                initial_noise = model.noise_generator.generate_color_noise(Delta=mv.Delta, 
                                                                  A=mv.A,
                                                                  eigs=mv.eigs,
                                                                  batch_size=mv.batch_size,  
                                                                  fft_norm=mv.fft_norm,
                                                                  device=mv.device
                                                                 )[0].real
        elif mv.model_type=="white" or mv.model_type=="whitening":
            # Generate white noise if Delta=0
            initial_noise = torch.randn(mv.batch_size, mv.color_channels, mv.N, mv.N, device=mv.device)

        step_size = 1.0/diffusion_steps
        current_x = initial_noise
        t0 = time.time()

        print("Initial Noise Input")
        show_25(model, current_x)
        clear_output(wait=True)
        print_step = 0
        
        for step in range(start_step, diffusion_steps):
            
            diffusion_times = (1.0 - step*step_size) * torch.ones(mv.batch_size, device=device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            signal_rates, noise_rates = model.mv.diffusion_schedule(diffusion_times)

            predicted_noise = model.network(current_x, noise_rates)

            predicted_x0 = (current_x - noise_rates * predicted_noise)/signal_rates
            sequence_batch = predicted_x0[0:num_sequences,...].unsqueeze(0)
            if step==0:
                sequences=sequence_batch
            else:
                sequences = torch.cat((sequences,sequence_batch), dim=0)
            
            next_diffusion_times = diffusion_times - step_size
            next_signal_rates, next_noise_rates = model.mv.diffusion_schedule(next_diffusion_times)

            # Add a small noise tensor to the predicted noise if stochasticity≠0
            if stochasticity==0:
                current_noise = predicted_noise
            else:
                total_steps = diffusion_steps-start_step
                # Stochastic noise dies off with time
                #phi = (1.0-step/total_steps) * stochasticity * np.pi/2.0
                phi = stochasticity * np.pi/2.0
                if mv.model_type=="cloud":
                    if mv.color_channels==1:
                        small_noise = model.noise_generator.generate_monochrome_noise(Delta=mv.Delta, 
                                                                                       A=mv.A, 
                                                                                       batch_size=mv.batch_size,  
                                                                                       fft_norm=mv.fft_norm,
                                                                                       device=mv.device
                                                                                      )[0].real
                    else:
                        small_noise = model.noise_generator.generate_color_noise(Delta=mv.Delta, 
                                                                                  A=mv.A,
                                                                                  eigs=mv.eigs,
                                                                                  batch_size=mv.batch_size,  
                                                                                  fft_norm=mv.fft_norm,
                                                                                  device=mv.device
                                                                                 )[0].real
                elif mv.model_type=="white" or mv.model_type=="whitening":
                    small_noise = torch.randn(mv.batch_size, mv.color_channels, mv.N, mv.N, device=mv.device)
                
                current_noise = np.cos(phi)*predicted_noise + np.sin(phi)*small_noise
            
            # Define current noise tensor for use in next step
            current_x = next_signal_rates * predicted_x0 + next_noise_rates * current_noise

            if (step+1)%print_every == 0:
                fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10,12))

                denormed_initial_noise = model.denormalize(initial_noise.squeeze()).to("cpu")
                denormed_predicted_noise = model.denormalize(predicted_noise.squeeze()).to("cpu")
                denormed_predicted_x0 = model.denormalize(predicted_x0.squeeze()).to("cpu")
                denormed_current_x = model.denormalize(current_x.squeeze()).to("cpu")

                axs[0].set_title("Initial Noise")
                axs[1].set_title("Predicted Noise")
                axs[2].set_title("Noisy Image")
                axs[3].set_title("Predicted Image")
                
                axs[0].imshow(torch.clamp(movedim(denormed_initial_noise[0]), min=0.0, max=1.0), cmap=cmap)
                axs[1].imshow(torch.clamp(movedim(denormed_predicted_noise[0]), min=0.0, max=1.0), cmap=cmap)
                axs[2].imshow(torch.clamp(movedim(denormed_current_x[0]), min=0.0, max=1.0), cmap=cmap)
                axs[3].imshow(torch.clamp(movedim(denormed_predicted_x0[0]), min=0.0, max=1.0), cmap=cmap)
                plt.setp(axs, xticks=[], yticks=[])
                plt.show()
                print("****** Image Gallery ******")
                print(f"Step {step+1}/{diffusion_steps} | Time Elapsed Last {print_every} Steps = {time.time()-t0}")
                show_25(model, predicted_x0, cmap=cmap)
                clear_output(wait=True)
                t0 = time.time()

                print_step+=1
                
    return predicted_x0, sequences
