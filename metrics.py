import torch
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

def calculate_inception_score(images, splits=10, device='cuda'):
    """
    Calculate Inception Score (IS) for generated images.
    :param images: tensor of shape (N, C, H, W) in range [0, 1]
    :param splits: number of splits for IS calculation
    :param device: computation device
    :return: inception score
    """
    images = images.clamp(0, 1)  # Ensure in [0, 1]
    images = (images * 255).to(torch.uint8)  # Convert to uint8
    images = images.to(device)

    is_metric = InceptionScore(splits=splits).to(device)
    score = is_metric(images)
    return score[0]  # score is a tuple (mean, std)
from torchmetrics.image.fid import FrechetInceptionDistance

def calculate_fid(real_images, generated_images, device='cpu'):
    """
    Calculate Frechet Inception Distance (FID) between real and generated images.
    Inputs must be in [0, 1] float and are converted to uint8.
    """
    real_images = (real_images.clamp(0, 1) * 255).to(torch.uint8).to(device)
    generated_images = (generated_images.clamp(0, 1) * 255).to(torch.uint8).to(device)

    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)

    return fid.compute().item()

