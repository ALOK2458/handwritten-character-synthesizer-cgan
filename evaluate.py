"""Measure conditional accuracy using a separately trained real-image classifier."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from cgan import CharacterClassifier, Generator, NUM_CLASSES
from data import label_to_letter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/cgan_latest.pt")
    parser.add_argument("--classifier", default="checkpoints/classifier.pt")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--output", default="outputs/evaluation_samples.png")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be positive.")
    if not Path(args.checkpoint).is_file():
        parser.error(f"GAN checkpoint not found: {args.checkpoint}")
    if not Path(args.classifier).is_file():
        parser.error(
            f"Classifier not found: {args.classifier}. "
            "Train it first with `python train_classifier.py`."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan_checkpoint = torch.load(args.checkpoint, map_location=device)
    classifier_checkpoint = torch.load(args.classifier, map_location=device)
    if gan_checkpoint.get("num_classes") != NUM_CLASSES:
        raise ValueError("GAN checkpoint is not a 26-class EMNIST Letters model.")
    if classifier_checkpoint.get("num_classes") != NUM_CLASSES:
        raise ValueError("Classifier checkpoint is not a 26-class EMNIST Letters model.")

    noise_size = int(gan_checkpoint["noise_size"])
    generator = Generator(noise_size=noise_size).to(device)
    generator.load_state_dict(gan_checkpoint["generator_state_dict"])
    generator.eval()
    classifier = CharacterClassifier().to(device)
    classifier.load_state_dict(classifier_checkpoint["model_state_dict"])
    classifier.eval()

    sample_count = args.samples_per_class
    labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(sample_count)
    random_generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        labels.numel(), noise_size, generator=random_generator, device=device
    )
    predicted = []
    with torch.no_grad():
        for start in range(0, labels.numel(), 512):
            batch_images = generator(noise[start : start + 512], labels[start : start + 512])
            predicted.append(classifier(batch_images).argmax(dim=1))
    predicted_labels = torch.cat(predicted)
    accuracy = float((predicted_labels == labels).float().mean().item())
    print(
        f"Conditional accuracy: {accuracy:.2%} "
        f"({int((predicted_labels == labels).sum())}/{labels.numel()} generated images "
        "classified as their requested letter)."
    )
    print(
        "This is a classifier-based proxy, not a human quality rating; "
        f"classifier held-out accuracy on real test images was "
        f"{classifier_checkpoint.get('test_accuracy', float('nan')):.2%}."
    )

    count = min(8, sample_count)
    figure, axes = plt.subplots(NUM_CLASSES, count, figsize=(count * 1.4, 40))
    with torch.no_grad():
        display_labels = torch.arange(NUM_CLASSES, device=device).repeat_interleave(count)
        display_noise = torch.randn(
            display_labels.numel(),
            noise_size,
            generator=torch.Generator(device=device).manual_seed(args.seed + 1),
            device=device,
        )
        display_images = generator(display_noise, display_labels).cpu()
    for row in range(NUM_CLASSES):
        for column in range(count):
            axis = axes[row, column] if count > 1 else axes[row]
            image_index = row * count + column
            axis.imshow(
                display_images[image_index, 0].mul(0.5).add(0.5).clamp(0, 1),
                cmap="gray",
            )
            if column == 0:
                axis.set_ylabel(label_to_letter(row), rotation=0, labelpad=12)
            axis.axis("off")
    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120)
    plt.close(figure)
    print(f"Saved generated-image grid to {output.resolve()}")


if __name__ == "__main__":
    main()
