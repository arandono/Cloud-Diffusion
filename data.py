import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
import numpy as np
import cloud_diffusion as cd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import IPython.display

def get_filter_lists(dataset, short_edge=1, long_edge=32, batch_size=64):
    """
    Builds two lists of integers representing the index of images in the dataset
      clean_list: indices of images with no letterboxes
      dirt_list: indices of images with letterboxes
    Clips the clean_list so that it is divisible by the batch size
    """
    horizontal_pool = torch.nn.AvgPool2d(kernel_size=(short_edge, long_edge), stride=(1, 1))
    vertical_pool = torch.nn.AvgPool2d(kernel_size=(long_edge, short_edge), stride=(1, 1))
    
    def has_vertical_bars(x):
        y = vertical_pool(x)
        if y.min() == 0.0 or y.max() == 1.0:
            return True
        else:
            return False
        
    def has_horizontal_bars(x):
        y = horizontal_pool(x)
        if y.min() == 0.0 or y.max() == 1.0:
            return True
        else:
            return False
    
    clean_list = []
    dirty_list = []
    
    for i in range(len(dataset)):
        image, _ = dataset[i]
        
        if has_vertical_bars(image) or has_horizontal_bars(image):
            dirty_list.append(i)
        else:
            clean_list.append(i)
    
    c = len(clean_list)
    d = len(dirty_list)
    
    print(f"{d}/{c+d} images ({100 * d/(d+c)}%), moved to dirty dataset.")
    
    clean_list = clean_list[0:c//batch_size * batch_size]
    
    print(f"Clean list clipped to {len(clean_list)} images so it is divisible by batch_size={batch_size}.")
    
    return clean_list, dirty_list
    
    
def get_datasets(batch_size=64, grayscale=True):
    
    # Applies minimal transformations to STL10 data
    # Preserves 96x96 size but converts to grayscale
    if grayscale:
        minimal_transforms = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    else:
        minimal_transforms = transforms.Compose([transforms.ToTensor()])

    base_dataset = datasets.STL10(root='data', download=True, transform=minimal_transforms, split="unlabeled")
    clean_list, dirty_list = get_filter_lists(base_dataset, batch_size=batch_size)
    
    clean_dataset = torch.utils.data.Subset(base_dataset, clean_list)
    dirty_dataset = torch.utils.data.Subset(base_dataset, dirty_list)
    
    return clean_dataset, dirty_dataset
    
    
def get_dataloaders(dataset, batch_size=64):     
    
    base_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    rotated_dataloader = torch.utils.data.DataLoader(rotated_dataset, batch_size=batch_size, shuffle=True)

    return train_datasets, train_loaders


def load_data_2(batch_size):
    
    # Applies minimal transformations to STL10 data
    # Preserves 96x96 size but converts to grayscale
    transform_0 = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    
    # prepare the data set
    train_dataset = datasets.STL10(root='data', download=True, transform=transform_0, split="unlabeled")
    
    # prepare data loader
    train_loader_0 = torch.utils.data.DataLoader(train_dataset, 
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 drop_last=False
                                                )
    
    # prepare data loader
    train_loader_1 = torch.utils.data.DataLoader(train_dataset, 
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 drop_last=True
                                                )

    # prepare data loader
    train_loader_2 = torch.utils.data.DataLoader(train_dataset, 
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 drop_last=True
                                                )
    
    train_loader_2 = torch.utils.data.DataLoader(train_dataset, 
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 drop_last=True
                                                )

    train_loaders = [train_loader_0, train_loader_1, train_loader_2]

    return train_dataset, train_loaders


def reshuffle_duplicates(batch1, batch2):
    """
    Checks for duplicates in the batch items of the two tensor batches.
    If duplicates are found, reshuffles. Repeats until no duplicates are found.
    Terminates if can't eliminate duplicates in 10 iterations.
    """
    def has_duplicates(batch1, batch2):
        has_duplicates = False

        for i in range(batch1.shape[0]):
            if torch.equal(batch1[i], batch2[i]):
                has_duplicates = True
                break
        return has_duplicates
    
    B = batch1.shape[0]
    duplicates_found = has_duplicates(batch1, batch2)
    n=0
    
    while duplicates_found and n<10:
        new_indices = np.random.permutation(B)
        new_batch = torch.zeros_like(batch2)
        for i in range(B):
            new_batch[i] = batch2[new_indices[i]]
        batch2 = new_batch
        n+=1
        duplicates_found = has_duplicates(batch1, batch2)
        
    if n>=10:
        print("Couldn't reshuffle without duplicates after 10 iterations. Terminating with duplicates.")
        
    return batch1, batch2


def save_checkpoint(model, filename, folder="checkpoints/", extension=".tar"):
    """
    Saves the model at a checkpoint.
    """
    
    filepath = f"{folder}{filename}{extension}"
    torch.save({
                'model_variables': model.mv,
                'model_state_dict': model.network.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict()
                }, filepath)
    print(f"Saved model to {filepath}")
    

def load_checkpoint(filename, folder="checkpoints/", extension=".tar"):
    """
    Returns a new model loaded from a checkpoint.
    """
    checkpoint = torch.load(f"{folder}{filename}{extension}", weights_only=False)
    mv = checkpoint["model_variables"]
    model = cd.CloudDiffusionModel(mv)
    model.network.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model



def save_movies(sequences, 
                model, 
                filename, 
                title=None,
                interval=15, 
                folder="animations/", 
                extension=".mp4"):
    movies_list = []
    N = sequences.shape[1]
    for n in range(N):
        num_frames = sequences.shape[0]
        tensor_sequence = []

        for i in range(num_frames):
            image = torch.clamp(model.denormalize(sequences[i,n,...].squeeze()).to("cpu"), min=0, max=1)
            tensor_sequence.append(image)

        # Create figure and axes for the animation
        fig, ax = plt.subplots()
        if not title==None:
            ax.set_title(title)
        plt.setp(ax, xticks=[], yticks=[])
        im = ax.imshow(tensor_sequence[0], cmap='gray', vmin=0, vmax=1)  # Initial frame
        

        # Animation update function
        def update(i):
            im.set_array(tensor_sequence[i])
            return [im]

        # Create the animation
        ani = animation.FuncAnimation(fig=fig, func=update, frames=num_frames, interval=interval)

        # Save the animation as an mp4
        full_filepath = f"{folder}{filename}_{n+1}{extension}"
        ani.save(full_filepath, writer='ffmpeg') # requires ffmpeg
        movies_list.append(full_filepath)
    
    return movies_list