"""Save and optionally display a grid of real EMNIST training examples."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data import get_dataloader, label_to_letter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="outputs/real_samples.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    loader = get_dataloader(data_dir=args.data_dir, batch_size=64)
    images, labels = next(iter(loader))
    figure, axes = plt.subplots(8, 8, figsize=(10, 10))
    for axis, image, label in zip(axes.flat, images, labels):
        axis.imshow(image.squeeze(0).mul(0.5).add(0.5).clamp(0, 1), cmap="gray")
        axis.set_title(label_to_letter(int(label)), fontsize=8)
        axis.axis("off")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    print(f"Saved a grid of real EMNIST samples to {output.resolve()}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
