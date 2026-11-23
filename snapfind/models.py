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


class RetrievalNet(torch.nn.Module):
    # A_ft: backbone -> classifier. C: backbone -> hash-lag -> classifier. Classifier bruges kun under træning.

    def __init__(self, n_classes, bits=None):
        super().__init__()
        self.backbone = build_backbone()
        self.bits = bits
        if bits is None:
            self.hash_layer = None
            width = FEATURE_DIM
        else:
            self.hash_layer = torch.nn.Linear(FEATURE_DIM, bits)
            torch.nn.init.xavier_uniform_(self.hash_layer.weight)
            width = bits
        self.classifier = torch.nn.Linear(width, n_classes)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def embed(self, x):
        features = self.backbone(x)
        if self.hash_layer is None:
            return features
        # tanh giver værdier mellem -1 og 1. Fortegnet bliver til bit, når koden gemmes.
        return torch.tanh(self.hash_layer(features))

    def forward(self, x):
        embedding = self.embed(x)
        return embedding, self.classifier(embedding)


def encode_images(model, images):
    if isinstance(model, RetrievalNet):
        return model.embed(images)  # uden klassifikatoren
    return model(images)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    net = RetrievalNet(checkpoint["n_classes"], checkpoint["bits"])
    net.load_state_dict(checkpoint["weights"])
    return net
