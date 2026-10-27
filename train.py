"""Train the conditional GAN on EMNIST Letters and save checkpoints and plots."""

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn

from cgan import Discriminator, Generator, NOISE_SIZE, NUM_CLASSES
from data import get_dataloader, label_to_letter


def save_sample_grid(
    generator: Generator,
    fixed_noise: torch.Tensor,
    fixed_labels: torch.Tensor,
    device: torch.device,
    output_path: Path,
) -> None:
    generator.eval()
    with torch.no_grad():
        generated = generator(fixed_noise.to(device), fixed_labels.to(device)).cpu()
    figure, axes = plt.subplots(4, 8, figsize=(12, 6))
    for index, axis in enumerate(axes.flat):
        image = generated[index, 0].mul(0.5).add(0.5).clamp(0, 1)
        axis.imshow(image, cmap="gray")
        axis.set_title(label_to_letter(int(fixed_labels[index])), fontsize=9)
        axis.axis("off")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    generator.train()


def save_loss_plot(history: List[Tuple[float, float]], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot([row[0] for row in history], label="Discriminator")
    axis.plot([row[1] for row in history], label="Generator")
    axis.set(xlabel="Epoch", ylabel="BCE loss", title="cGAN training losses")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-size", type=int, default=NOISE_SIZE)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be positive and --batch-size must be at least 2.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}.")

    loader = get_dataloader(
        args.data_dir, train=True, batch_size=args.batch_size, download=True
    )
    generator = Generator(noise_size=args.noise_size).to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer_g = torch.optim.Adam(
        generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999)
    )

    checkpoints = Path(args.checkpoint_dir)
    outputs = Path(args.output_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    fixed_noise = torch.randn(32, args.noise_size)
    fixed_labels = torch.arange(NUM_CLASSES).repeat(2)[:32]
    history: List[Tuple[float, float]] = []

    for epoch in range(1, args.epochs + 1):
        discriminator_total = 0.0
        generator_total = 0.0
        batch_count = 0
        for real_images, real_labels in loader:
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)
            batch_size = real_images.size(0)
            if batch_size < 2:
                continue

            real_targets = torch.ones(batch_size, device=device) * 0.9
            fake_targets = torch.zeros(batch_size, device=device)

            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(real_images, real_labels)
            real_loss = criterion(real_logits, real_targets)

            noise = torch.randn(batch_size, args.noise_size, device=device)
            fake_labels = torch.randint(NUM_CLASSES, (batch_size,), device=device)
            with torch.no_grad():
                fake_images = generator(noise, fake_labels)
            fake_logits = discriminator(fake_images, fake_labels)
            fake_loss = criterion(fake_logits, fake_targets)
            discriminator_loss = (real_loss + fake_loss) / 2
            discriminator_loss.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, args.noise_size, device=device)
            fake_labels = torch.randint(NUM_CLASSES, (batch_size,), device=device)
            generated_images = generator(noise, fake_labels)
            generated_logits = discriminator(generated_images, fake_labels)
            generator_loss = criterion(generated_logits, torch.ones(batch_size, device=device))
            generator_loss.backward()
            optimizer_g.step()

            discriminator_total += float(discriminator_loss.item())
            generator_total += float(generator_loss.item())
            batch_count += 1

        if batch_count == 0:
            raise RuntimeError("The training dataset produced no usable batches.")
        mean_d = discriminator_total / batch_count
        mean_g = generator_total / batch_count
        history.append((mean_d, mean_g))
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"D loss: {mean_d:.4f} | G loss: {mean_g:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "noise_size": args.noise_size,
            "num_classes": NUM_CLASSES,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "loss_history": history,
        }
        torch.save(checkpoint, checkpoints / "cgan_latest.pt")
        if epoch % 5 == 0 or epoch == args.epochs:
            torch.save(checkpoint, checkpoints / f"cgan_epoch_{epoch:03d}.pt")
            save_sample_grid(
                generator,
                fixed_noise,
                fixed_labels,
                device,
                outputs / f"generated_epoch_{epoch:03d}.png",
            )
            save_loss_plot(history, outputs / "training_losses.png")

    print(f"Training finished. Checkpoint: {(checkpoints / 'cgan_latest.pt').resolve()}")
    print(f"Loss plot: {(outputs / 'training_losses.png').resolve()}")


if __name__ == "__main__":
    main()
