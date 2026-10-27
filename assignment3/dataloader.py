"""CAP 5410 - Assignment 3: MNIST data loading.

Downloads MNIST through torchvision, splits the 60,000 training images
into 48,000 for training and 12,000 for validation with a fixed seed,
and keeps the 10,000 test images apart for the final evaluation.
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

DATA_DIR = "./data"
SPLIT_SEED = 42          # same split for every experiment
N_TRAIN = 48_000
N_VAL = 12_000


def split_indices(seed=SPLIT_SEED):
    """Return (train_idx, val_idx): a fixed random 80/20 split of the 60,000 images."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N_TRAIN + N_VAL, generator=g)
    return perm[:N_TRAIN].tolist(), perm[N_TRAIN:].tolist()


def train_mean_std(train_idx):
    """Mean and standard deviation of the pixels in the training split only.

    The validation and test images are not used here, so no information
    about them leaks into the preprocessing.
    """
    raw = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True)
    pixels = raw.data[train_idx].float() / 255.0
    return pixels.mean().item(), pixels.std().item()


def get_loaders(batch_size, seed=SPLIT_SEED):
    """Build the train, validation and test DataLoaders.

    Preprocessing: ToTensor (0-255 -> 0-1) followed by normalization with
    the training-split mean and std. Only the training loader is shuffled,
    and its shuffle order comes from a seeded generator so that every run
    sees the batches in the same order.
    """
    train_idx, val_idx = split_indices(seed)
    mean, std = train_mean_std(train_idx)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    full_train = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    trainset = Subset(full_train, train_idx)
    valset = Subset(full_train, val_idx)

    g = torch.Generator().manual_seed(seed)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
    valloader = DataLoader(valset, batch_size=1000, shuffle=False, num_workers=0)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0)

    return trainloader, valloader, testloader, (mean, std)
