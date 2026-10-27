"""Small fully connected conditional GAN for 28x28 EMNIST Letters images."""

import torch
from torch import nn


NUM_CLASSES = 26
IMAGE_SIZE = 28
NOISE_SIZE = 100


class Generator(nn.Module):
    """Generate a 28x28 grayscale image conditioned on a letter label."""

    def __init__(
        self,
        noise_size: int = NOISE_SIZE,
        num_classes: int = NUM_CLASSES,
        label_size: int = 32,
    ) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, label_size)
        self.network = nn.Sequential(
            nn.Linear(noise_size + label_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, IMAGE_SIZE * IMAGE_SIZE),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_features = self.label_embedding(labels)
        combined = torch.cat((noise, label_features), dim=1)
        return self.network(combined).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)


class Discriminator(nn.Module):
    """Estimate whether an image is real, given the requested letter label."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        label_size: int = 32,
    ) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, label_size)
        self.network = nn.Sequential(
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE + label_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        flattened = images.view(images.size(0), -1)
        label_features = self.label_embedding(labels)
        combined = torch.cat((flattened, label_features), dim=1)
        return self.network(combined).view(-1)


class CharacterClassifier(nn.Module):
    """CNN used only to measure the labels recognized in generated images."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
