# Import packages
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import data
from scipy.stats import linregress
import torchvision.transforms as transforms
from IPython.display import clear_output
from RealFFT import fft
from RealFFT import ifft
from RealFFT import rfft
from RealFFT import irfft

def getDevice():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return device


class NoiseGenerator:
    def __init__(self, N:int=96, device="cpu"): 
        """
        Creates the class that is used to generate noise.
        Noise tensor has square dimensions NxN.
        """
        super().__init__()
        self.N = N
        self.device = device
        self.dist, self.Dist = self.build_dist()
        
    def build_dist(self):
        """
        Returns the distance tensor for use in the noise generator
        Center is fixed at N//2
        """
        N = self.N
        M = 3*self.N

        x = ((torch.arange(N, device=self.device)-N//2)/(N-1)).unsqueeze(-1).repeat((1,N))
        y = ((torch.arange(N, device=self.device)-N//2)/(N-1)).unsqueeze(0).repeat((N,1))
        
        X = ((torch.arange(M, device=self.device)-M//2)/(N-1)).unsqueeze(-1).repeat((1,M))
        Y = ((torch.arange(M, device=self.device)-M//2)/(N-1)).unsqueeze(0).repeat((M,1))
        
        # Calculate distance tensor
        dist = torch.sqrt(x**2 + y**2).unsqueeze(0).unsqueeze(0)
        Dist = torch.sqrt(X**2 + Y**2).unsqueeze(0).unsqueeze(0)
        
        return dist, Dist
        
        
#     def generate_noise(self, Delta=2, A=1, batch_size=64, color_channels=1, fft_norm="ortho"):
#         """
#         Generates square tensor of scale invariant noise of size 2^Nx2^N. 
#         The scale factor, a, determines the noise character.
#         a = 0 : White Noise
#         a = 1 : Pink Noise
#         a = 2 : Brownian Noise
#         """
#         N = self.N
            
#         # Without a center point the noise tends to be slightly skewed diagonally at large wavelengths
#         # So to compensate we add an extra row and column in Fourier space so that the exact center exists
#         # We remove a row and column in the final output
#         # Note: complex functionality is not fully implemented on mps so work on cpu first, then output real tensor to mps
#         Z_real = torch.randn(batch_size, color_channels, N, N)
#         Z_imag = torch.randn(batch_size, color_channels, N, N)

#         Z = torch.complex(Z_real, Z_imag)
        
#         dist = self.dist.repeat(batch_size, color_channels, 1, 1)
#         # 1/dist has a divide by zero error at the center point
#         # Save the value of this point and reset it after dividing by distance
#         # Note: the value at the center point controls the standard deviation of the mean of the color channels...
#         # ...since we don't need much variation in the mean we are safe by resetting it
#         Z = (A/(dist**(Delta))) * Z
#         Z[:,:,N//2, N//2] = 0.0
#         z = ifft(Z, norm=fft_norm)
        
#         return z, Z
    
    
    def generate_monochrome_noise(self, Delta=2, A=1, batch_size=64, fft_norm="ortho", device="cpu"):
        """
        Generates square tensor of scale invariant noise of size 2^Nx2^N.
        Returns two **complex** noise tensors z, and Z. Take real part if needed.
        z is in real space, Z is in Fourier space.
        
        The scale factor, Delta, determines the noise character.
        Delta = 0 : White Noise
        Delta = 1 : Pink Noise
        Delta = 2 : Brownian Noise
        
        Note: this is an improvement on the previous noise generator, which generated noise
        with periodic (toroidal) boundary conditions. This embeds the noise tensor in a larger
        array (3Nx3N) of scale invariant noise and cuts out the middle square. 
        """
        N = self.N
        M = 3*N
        
            
        Z0_real = torch.randn(batch_size, 1, M, M, device=device)
        Z0_imag = torch.randn(batch_size, 1, M, M, device=device)

        Z0 = torch.complex(Z0_real, Z0_imag)

        Dist = self.Dist.repeat(batch_size, 1, 1, 1).to(device)
        # 1/dist has a divide by zero error at the center point
        # Save the value of this point and reset it after dividing by distance
        # Note: the value at the center point controls the standard deviation of the mean of the color channels...
        # ...since we don't need much variation in the mean we are safe by resetting it
        Z0 = (A/(Dist**(Delta))) * Z0
        Z0[:,:,M//2, M//2] = 0.0

        z0 = ifft(Z0, norm=fft_norm)

        z = z0[:,:,N:2*N, N:2*N]
        Z = fft(z)

        return z, Z

        
    
    def generate_color_noise(self, Delta, A, eigs, batch_size=64,  fft_norm="ortho", device="cpu"):
        """
        Generates square tensor of scale invariant noise of size 2^Nx2^N. 
        The scale factor, a, determines the noise character.
        
        The color channel correlations are determined by the color-variance tensor, sigma_color,
        whose eigenvealues and eigenvectors, (L,V)=eigs, are computed from torch.linalg.eigs(sigma_color).

        
        Note: this uses generate_monochrome_noise, so the normalizing A value should be fixed from that.  
        """
        N = self.N
        M = 3*N
        
        z = torch.zeros(batch_size, 3, N, N, dtype=torch.cfloat, device=device)
        Z = torch.zeros(batch_size, 3, N, N, dtype=torch.cfloat, device=device)
        L,V = eigs

        for c in range(3):
            _z, _Z = self.generate_monochrome_noise(Delta=Delta, A=A, batch_size=batch_size, fft_norm=fft_norm, device=device)

            z[:,c,:,:] = torch.sqrt(L[c]) * _z.squeeze(1)
            Z[:,c,:,:] = torch.sqrt(L[c]) * _Z.squeeze(1)

        z = torch.einsum("cC,bCij->bcij", V.to("cpu"), z.to("cpu")).to(device)
        Z = torch.einsum("cC,bCij->bcij", V.to("cpu"), Z.to("cpu")).to(device)

        return z, Z
                
    
    def get_normalizing_A(self, Delta, n_batches=1000, batch_size=64, device="cpu"):
        """
        This gets mu and sigma for the real noise tensors for the chosen parameters.
        Returns to mu and sigma tensors for the real and imaginary parts separately
        """
        N=self.N
            
        sigma_diag_1 = torch.zeros(N,N, device=device)
        sigma_diag_2 = torch.zeros(N,N, device=device)
        n=0
        
        for i in range(n_batches):
            z = self.generate_monochrome_noise(Delta=Delta, A=1, batch_size=batch_size, fft_norm="ortho", device=device)[0]
            x1 = z.real.squeeze()
            x2 = z.imag.squeeze()
            
            _sigma_diag_1 = torch.sum(x1**2, dim=0)
            _sigma_diag_2 = torch.sum(x2**2, dim=0)
            
            sigma_diag_1 = (sigma_diag_1*n + _sigma_diag_1)/(n+batch_size)
            sigma_diag_2 = (sigma_diag_2*n + _sigma_diag_2)/(n+batch_size)
            
            n+=batch_size
            
            if (i+1)%(n_batches//10) == 0:
                print(f"Cycling through batches to calculate A ... {(i+1)/n_batches * 100}% complete.")
                clear_output(wait=True)
                
        A = 1/torch.sqrt((sigma_diag_1.mean()+sigma_diag_2.mean())/2.0).item()
    
        print(f"For N = {self.N}, ∆ = {Delta} to normalize position-space cloud noise to have std=1 use:")
        print(f"A = {A}")
            
        return A
    
    def get_sigma_noise(self, Delta, A, n_batches=1000, batch_size=64, device="cpu"):
        
        sigma_noise = torch.zeros(96,96,96,96, device=device)
        
        n=0
        
        for i in range(n_batches):
            x = self.generate_noise_2(Delta=Delta,
                                      A=A,
                                      batch_size=64,
                                      color_channels=1,
                                      fft_norm="ortho",
                                      device="mps"
                                     )[0].squeeze().real
            
            _sigma_noise = torch.einsum("bij,bkl->ijkl", x, x)

            sigma_noise = (sigma_noise*n + _sigma_noise)/(n+64)
            
            n+=64
            
            if i%(n_batches//10)==0:
                print(f"Batch {i}/{n_batches} ... {100*i/n_batches}% complete")
                clear_output(wait=True)
            
        return sigma_noise

     
def get_mu_and_std(dataset, device="cpu"):
    """
    Returns the means and standard deviations of the image set as a tuple (means, stds)
    Note: returns the pixel-wise values (mean and std for each pixel) not the total image mean and std. 
    """

    mu = torch.zeros_like(dataset[0][0].squeeze().to(device))
    std_squared = torch.zeros_like(dataset[0][0].squeeze().to(device))
    
    for n in range(len(dataset)):
        x = dataset[n][0].squeeze().to(device)
        mu = (mu*n + x)/(n+1)
        
    for n in range(len(dataset)):
        x = dataset[n][0].squeeze().to(device)
        _std_squared = (x-mu)**2
        std_squared = (std_squared*n + _std_squared)/(n+1)
        
    std = torch.sqrt(std_squared)

    return mu, std


def normalize(x, mu, std):
    device = x.device
    return (x-mu.to(device))/std.to(device)


def denormalize(x, mu, std):
    device = x.device
    return x*std.to(device) + mu.to(device)


def whiten(x, mu, std, Delta, eps, noise_generator, mu_whitened=None, std_whitened=None):
    d = noise_generator.dist.to(x.device)
    x = normalize(x, mu, std)
    x = irfft(rfft(x)*((d+eps)**(2*Delta)))
    
    if not (mu_whitened == None) and not (std_whitened==None):
        x = normalize(x, mu_whitened, std_whitened)

    return x


def get_whitened_mu_and_std(dataloader, 
                            mu, 
                            std, 
                            Delta,  
                            noise_generator,
                            eps=.001,
                            batch_size=64,
                            N=96,
                            device="cpu"):
    """
    Returns the means and standard deviations of the image set as a tuple (means, stds)
    Note: returns the pixel-wise values (mean and std for each pixel) not the total image mean and std. 
    """

    mu_whitened = torch.zeros(N, N , device=device)
    std_whitened_squared = torch.zeros_like(mu_whitened)
    
    data_iter = iter(dataloader)
    for n in range(len(data_iter)):
        x = next(data_iter)[0].squeeze().to(device)
        x = whiten(x, mu=mu, std=std, Delta=Delta, eps=eps, noise_generator=noise_generator)
        _mu_whitened = x.mean(dim=0)
        mu_whitened = (mu_whitened*n + _mu_whitened)/(n+1)
        if n%100==0:
            print(f"Calculating Mean: {n}/{len(data_iter)}")
            clear_output(wait=True)
        
    data_iter = iter(dataloader) # Fixed
    for n in range(len(data_iter)):
        x = next(data_iter)[0].to(device)
        x = whiten(x, mu=mu, std=std, Delta=Delta, eps=eps, noise_generator=noise_generator)
        _std_whitened_squared = torch.einsum("bij,bij->ij", x-mu_whitened, x-mu_whitened)/batch_size
        std_whitened_squared = (std_whitened_squared*n + _std_whitened_squared)/(n+1)
        if n%100==0:
            print(f"Calculating STD: {n}/{len(data_iter)}")
            clear_output(wait=True)
        
    std_whitened = torch.sqrt(std_whitened_squared)

    return mu_whitened, std_whitened


def get_sigma_color(dataset, mu, std, device="cpu"):
    """
    The RGB basis is generally not a natural color basis. 
    For a natural image set there are statistical correlations between the color channels.
    These are characterized by the variance tensor {\Sigma^{cij}}_{Ckl}.
    We will need to invert this tensor, which is numerically unstable. 
    To get around this, we average over the spatial pixels to get an average color-channel variance \Sigma^c_C.
    This is usually easily invertible, and we can easily compute its eigenvalues and eigenvectors.
    """
    
    N = dataset[0][0].shape[-1]
    sigma_diag = torch.zeros(3,3,N,N, device=device)
    mu = mu.to(device)
    std = std.to(device)
    
    for n in range(len(dataset)):
        x = normalize(dataset[n][0].to("mps"), mu, std)
        _sigma_diag = torch.einsum("cij,Cij->cCij", x, x)
        sigma_diag = (sigma_diag*n+_sigma_diag)/(n+1)
    
    sigma_color = sigma_diag.mean(dim=(2,3))
    
    return sigma_color
    

def get_sigma(dataloader, mu, std, batch_size=64, N=96, normalize=True, device="cpu", realFFT=False):
    """
    Gets the Sigma tensor for a real image set.
    """
    b = batch_size
    sigma = torch.zeros(N,N,N,N).to(device)
    _sigma = torch.zeros(N,N,N,N).to(device)

    t0=time.time()
    t_last=t0
    data_iter = iter(dataloader)
    
    n=0
    for i in range(len(data_iter)):
        image_batch, _ = next(data_iter)
        image_batch = image_batch.to(device)
        
        mu = mu.to(device)
        std = std.to(device)
        X = image_batch.squeeze()-mu
        if normalize:
            X = X/std
        
        if realFFT:
            X = rfft(X)
            
        _sigma = torch.einsum("bij,bkl->ijkl", X, X)
        sigma = (sigma*n + _sigma)/(n+b)
        
        n+=b
        
        if i%100 == 0:
            t1 = time.time()
            print(f"{i}/{len(data_iter)} || Time Elapsed = {t1-t_last}")
            clear_output(wait=True)
            t_last=time.time()
            
    print(f"Completed in {time.time()-t0} seconds.")
    
    return sigma


def get_Gamma_and_C(dataloader, mu, std, batch_size=64, N=96, device="cpu", normalize=True, fft_norm="ortho"):
    """
    Gets the Sigma tensor for the Fourier transformed image set.
    For this calculation we combine to randomly sampled images,
    x1 and x2 into one complex tensor z=x1+ix2, taking care that x1!=x2.
    Then we Fourier transform z to Z before analysis.
    """
    
    b = batch_size
    Gamma = torch.zeros(N,N,N,N,dtype=torch.cfloat, device=device)
    _Gamma = torch.zeros(N,N,N,N,dtype=torch.cfloat, device=device)
    C = torch.zeros(N,N,N,N,dtype=torch.cfloat, device=device)
    _C = torch.zeros(N,N,N,N,dtype=torch.cfloat, device=device)
    
    # Get the data iterators
    data_iter = iter(dataloader)
    
    t0=time.time()
    t_last = t0
    
    batches = len(data_iter)
    n=0
    
    for i in range(batches):
        x, _ = next(data_iter)
        x = x.squeeze().to(device)
        
        mu = mu.to(device)
        std = std.to(device)
        x0 = (x-mu)
        
        # Normalize:
        if normalize:
            x0 = x0/std
        
        # Get complex input vector
        Z = fft(x0, norm=fft_norm)
        
        _Gamma = torch.einsum("bij,bkl->ijkl", Z, torch.conj_physical(Z))
        _C = torch.einsum("bij,bkl->ijkl", Z, Z)
        
        Gamma = (Gamma*n + _Gamma)/(n+b)
        C = (C*n + _C)/(n+b)
        
        n+=b
        
        if i%100 == 0:
            t1 = time.time()
            print(f"{i}/{len(data_iter)} || Time Elapsed = {t1-t_last}")
            clear_output(wait=True)
            t_last=time.time()
            
    print(f"Completed in {time.time()-t0} seconds.")
    
    return Gamma, C



def mask_this(x, a=3, R=1, mask_value=-10000):
    """
    Applies a mask to the given (square) tensor. 
    The central cross region is masked as well as the area outside of a given region.
    Note: this is not to be confused with masked_tensors in torch. 
    
    Parameters:
    a: the width of the central cross region (default=3)
    R: the radius of the radial mask (default=1)
    mask_value: the value the tensor is set to in the masked regions (default=-10000)
    """
    N = x.shape[-1]
    center = N//2
    masked_x = x.detach()
    
    for i in range(N):
        for j in range(N):
            r = np.sqrt(((i-center)/center)**2 + ((j-center)/center)**2)
            in_central_cross = (i<(center+a) and i>(center-a)) or (j<(center+a) and j>(center-a))
            outside_R = r > R
            if in_central_cross or outside_R:
                masked_x[i,j] = mask_value
    
    return masked_x


def get_radial_dependence(Gamma, a=4, R=0.5, visualize=True):
    Gamma = Gamma.to("cpu")
    Gamma_diag = torch.einsum("ijij->ij", Gamma).real
    log_r = []
    log_y = []
    colors = []
    masked_log_r = []
    masked_log_y = []

    mask_view = torch.zeros_like(Gamma_diag)

    N = Gamma_diag.shape[0]
    r0 = N//2
    
    for i in range(N):
        for j in range(N):
            r = np.sqrt(((i-r0)/N)**2+((j-r0)/N)**2)
            if r != 0:
                log_r.append(np.log(r))
                log_y.append(np.log(Gamma_diag[i,j].item()))
                if (i<(r0+a) and i>(r0-a)) or (j<(r0+a) and j>(r0-a)):
                    colors.append([.8,.8,1])
                    mask_view[i,j] = -10000
                elif r>R:
                    colors.append([1,.8,.8])
                    mask_view[i,j] = -10000
                else:
                    colors.append([0, 0, 1])
                    masked_log_r.append(np.log(r))
                    masked_log_y.append(np.log(Gamma_diag[i,j].item()))
                    mask_view[i,j] = Gamma_diag[i,j]

    if visualize:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        axs[0].set_title('Standard Deviation with Masking')
        axs[1].set_title('Radial Dependence of STD (log-log)')
        axs[0].imshow(mask_view, vmin=0, vmax=1)
        axs[1].scatter(log_r, log_y, c=colors, s=1)

    linear_regression_data = linregress(masked_log_r, masked_log_y)
    Delta = -linear_regression_data.slope/2
    A = np.exp(linear_regression_data.intercept/2)
    print(linear_regression_data)
    print("===========================")
    print(f"From the slope of the log-log plot we determine the factors A and ∆:")
    print(f"A={A}")
    print(f"∆={Delta}")
    print("===========================")
    
    return A, Delta
        
        
def maha_batch_distance(X,Y, Delta, noise_generator):
    b = X.shape[0]
    X = X.squeeze(1)
    Y = Y.squeeze(1)
    r2D = noise_generator.dist.squeeze()**(2*Delta)
    D = torch.sqrt(torch.einsum("bij,ij,bij->b", X-Y,r2D,X-Y))
    D_ave = torch.mean(D)
    
    return D_ave
        

def maha_distances(dataloader, mu, std, A, Delta, noise_generator, device, batch_size=64):
    data_iter_1 = iter(dataloader)
    data_iter_2 = iter(dataloader)
    
    white_white = torch.ones(1, device=device)
    cloud_cloud = torch.ones(1, device=device)
    image_image = torch.ones(1, device=device)
    white_cloud = torch.ones(1, device=device)
    white_image = torch.ones(1, device=device)
    cloud_image = torch.ones(1, device=device)
    white_Mu = torch.ones(1, device=device)
    cloud_Mu = torch.ones(1, device=device)
    image_Mu = torch.ones(1, device=device)
    mu = mu.to(device)
    std = std.to(device)
    
    n=0
    for q in range(len(data_iter_1)):
        # Generate noise and image batches
        white1 = rfft(torch.randn(batch_size,1,96,96, device=device))
        white2 = rfft(torch.randn(batch_size,1,96,96, device=device))
        cloud1 = rfft(noise_generator.generate_monochrome_noise(batch_size=batch_size,A=A, Delta=Delta, device=device)[0].real)
        cloud2 = rfft(noise_generator.generate_monochrome_noise(batch_size=batch_size,A=A, Delta=Delta, device=device)[0].real)
        image1 = rfft(normalize(next(data_iter_1)[0].to(device), mu, std))
        image2 = rfft(normalize(next(data_iter_2)[0].to(device), mu, std))
        Mu = torch.zeros_like(image1) # Note that since we have normalized the images, the mean in Real Fourier Space is zero
        
        # Batch calculations of Maha Distance
        _white_white = maha_batch_distance(white1, white2, Delta, noise_generator)
        _cloud_cloud = maha_batch_distance(cloud1, cloud2, Delta, noise_generator)
        _image_image = maha_batch_distance(image1, image2, Delta, noise_generator)
        _white_cloud = maha_batch_distance(white1, cloud1, Delta, noise_generator)
        _white_image = maha_batch_distance(white1, image1, Delta, noise_generator)
        _cloud_image = maha_batch_distance(cloud1, image1, Delta, noise_generator)
        _white_Mu = maha_batch_distance(white1, Mu, Delta, noise_generator)
        _cloud_Mu = maha_batch_distance(cloud1, Mu, Delta, noise_generator)
        _image_Mu = maha_batch_distance(image1, Mu, Delta, noise_generator)
        
        # Updating the means
        white_white = (white_white*n + _white_white)/(n+1)
        cloud_cloud = (cloud_cloud*n + _cloud_cloud)/(n+1)
        image_image = (image_image*n + _image_image)/(n+1)
        white_cloud = (white_cloud*n + _white_cloud)/(n+1)
        white_image = (white_image*n + _white_image)/(n+1)
        cloud_image = (cloud_image*n + _cloud_image)/(n+1)
        white_Mu = (white_Mu*n + _white_Mu)/(n+1)
        cloud_Mu = (cloud_Mu*n + _cloud_Mu)/(n+1)
        image_Mu = (image_Mu*n + _image_Mu)/(n+1)
        
        n += 1
        
        if q%100 ==0:
            print(f"{q}/{len(data_iter_1)} complete")
            clear_output(wait=True)
        
    distances = {
        "d(cloud_noise,cloud_noise|Q)" : cloud_cloud.item(),
        "d(cloud_noise,images|Q)" : cloud_image.item(),
        "d(cloud_noise|Q)" : cloud_Mu.item(),
        "d(images,images|Q)" : image_image.item(),
        "d(images|Q)" : image_Mu.item(),
        "d(white_noise,white_noise|Q)" : white_white.item(),
        "d(white_noise,cloud_noise|Q)" : white_cloud.item(),
        "d(white_noise,images|Q)" : white_image.item(),
        "d(white_noise|Q)" : white_Mu.item()}
    
    
    for key, value in distances.items():
        print(key.rjust(28)+ " = " + str(value))
     
    return distances    
    
    
