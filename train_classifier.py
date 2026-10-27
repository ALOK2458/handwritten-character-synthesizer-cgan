"""Train a small EMNIST classifier for an honest generated-label metric."""

import argparse
from pathlib import Path

import torch
from torch import nn

from cgan import CharacterClassifier, NUM_CLASSES
from data import get_datasets


def evaluate(model: CharacterClassifier, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            correct += int((model(images).argmax(dim=1) == labels).sum().item())
            total += labels.numel()
    if total == 0:
        raise RuntimeError("The test dataset is empty.")
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("--epochs must be positive and --batch-size must be at least 2.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training classifier on {device}.")
    train_dataset, test_dataset = get_datasets(args.data_dir, download=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = CharacterClassifier(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        if batches == 0:
            raise RuntimeError("The training dataset produced no usable batches.")

        accuracy = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"loss: {total_loss / batches:.4f} | "
            f"held-out real-image accuracy: {accuracy:.2%}"
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "num_classes": NUM_CLASSES,
                    "test_accuracy": accuracy,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_dir / "classifier.pt",
            )

    print(
        f"Best held-out real-image accuracy: {best_accuracy:.2%}. "
        f"Classifier saved to {(checkpoint_dir / 'classifier.pt').resolve()}"
    )


if __name__ == "__main__":
    main()
