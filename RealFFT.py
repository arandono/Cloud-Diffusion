import torch

def fft(z, norm="ortho"):
    """
    Returns the 2D fft of input. 
    Uses an fftshift to shift the origin to the center of the tensor.
    Normalization is fixed by default as "ortho", or symmetric.
    """
    return torch.fft.fftshift(torch.fft.fft2(z, norm=norm))


def ifft(Z, norm="ortho"):
    """
    Returns the 2D inverse fft of input. 
    Assumes the input is fftshifted so the origin is the center of the tensor.
    Normalization is fixed by default as "ortho", or symmetric.
    """
    return torch.fft.ifft2(torch.fft.ifftshift(Z), norm=norm)

#######################
# Below are a series of methods that define a real fft with the same information, size, and statistical properties of the complex fft

def topMask(x):
    N = x.shape[-1]
    mask = torch.ones_like(x)
    
    if N%2 == 0:
        mask[...,0,(N//2+1):] = 0
        mask[...,N//2,(N//2+1):] = 0
        mask[...,(N//2+1):,:] = 0
    else:
        mask[...,N//2, (N//2+1):]=0
        mask[...,(N//2+1):,:] = 0
        
    return mask*x

def botMask(x):
    N = x.shape[-1]
    mask = torch.zeros_like(x)
    
    if N%2 == 0:
        mask[...,0,(N//2+1):] = 1
        mask[...,N//2, (N//2+1):] = 1
        mask[...,(N//2+1):,:] = 1
    else:
        mask[...,N//2, (N//2+1):]=1
        mask[...,(N//2+1):,:] = 1
        
    return mask*x


def flipit(x):
    N = x.shape[-1]
    
    if N%2 == 0:
        A=x[...,0,1:]
        B=x[...,1:,0]
        C=x[...,1:,1:]
        
        X=torch.zeros_like(x)
        X[...,0,0]=x[...,0,0]
        X[...,0,1:]=torch.flip(A,[A.dim()-1])
        X[...,1:,0]=torch.flip(B,[B.dim()-1])
        X[...,1:,1:]= torch.flip(C, [C.dim()-2,C.dim()-1])
    else:
        X = torch.flip(x, [x.dim()-2,x.dim()-1])
        
    return X


def realForm(Z):
    A = topMask(Z.real)
    B = botMask(Z.imag)
    return A+B


def complexForm(z):
    rA = topMask(z)
    iB = botMask(z)
    iA = topMask(flipit(iB))
    rB = botMask(flipit(rA))

    return rA+rB+1.j*(iB-iA)


def rfft(x, norm="ortho"):
    """
    Returns a real form of the complex fft of a real image. 
    The Fourier t-form of a real image Z^{ij}=F(x^{ij}) is skew Hermitian
    in the following sense: Z^{IJ} = (Z^{ij})^* where IJ represents the coordinates 
    ij mirrored about the origin.
    This means that half of the pixels are not independent. 
    We use this to construct a real-form of Z that contains the same information as Z. 
    The essence is to take the real part of the top half plane and the imaginary part of the bottom half plane.
    But, there are some subtleties that must be taken into account for different size images.
    """
    
    Z = fft(x, norm=norm)
    z = realForm(Z)
    return z


def irfft(z, norm="ortho"):
    """
    Takes a real tensor with the form of the output of rfft 
    and inverts the fft to return the original image
    """
    
    Z = complexForm(z)
    x = ifft(Z, norm=norm).real
    
    return x
    