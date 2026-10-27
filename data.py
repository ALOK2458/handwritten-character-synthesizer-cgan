"""Dataset and label helpers for torchvision's EMNIST Letters split."""

from pathlib import Path
from typing import Tuple

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cgan import NUM_CLASSES


LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def letter_to_label(letter: str) -> int:
    """Convert one uppercase English letter to the model's zero-based label."""
    normalized = letter.strip().upper()
    if len(normalized) != 1 or normalized not in LETTERS:
        raise ValueError("Choose one letter from A through Z.")
    return ord(normalized) - ord("A")


def label_to_letter(label: int) -> str:
    """Convert a model label in [0, 25] to an uppercase English letter."""
    if not 0 <= label < NUM_CLASSES:
        raise ValueError(f"Letter label must be between 0 and {NUM_CLASSES - 1}.")
    return LETTERS[label]


def get_transform() -> transforms.Compose:
    """Orient EMNIST images for normal reading and scale pixels to [-1, 1]."""
    return transforms.Compose(
        [
            transforms.Lambda(
                lambda image: image.rotate(-90, expand=True).transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def get_dataloader(
    data_dir: str = "data",
    train: bool = True,
    batch_size: int = 128,
    download: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Load EMNIST Letters, remapping its documented labels 1..26 to 0..25."""
    dataset = datasets.EMNIST(
        root=str(Path(data_dir)),
        split="letters",
        train=train,
        download=download,
        transform=get_transform(),
        target_transform=lambda target: target - 1,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=False,
    )


def get_datasets(data_dir: str = "data", download: bool = True) -> Tuple[object, object]:
    """Return training and test datasets using the same preprocessing."""
    train_dataset = datasets.EMNIST(
        root=str(Path(data_dir)),
        split="letters",
        train=True,
        download=download,
        transform=get_transform(),
        target_transform=lambda target: target - 1,
    )
    test_dataset = datasets.EMNIST(
        root=str(Path(data_dir)),
        split="letters",
        train=False,
        download=download,
        transform=get_transform(),
        target_transform=lambda target: target - 1,
    )
    return train_dataset, test_dataset
