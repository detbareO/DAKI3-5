import torch
from torchvision import models

FEATURE_DIM = 512
BACKBONE_ID = "torchvision resnet18, IMAGENET1K_V1 weights, fc=Identity"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)  # findes ikke på Jetson
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_backbone():
    if hasattr(models, "ResNet18_Weights"):
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        # gammel torchvision på Jetson
        net = models.resnet18(pretrained=True)
    net.fc = torch.nn.Identity()
    return net
