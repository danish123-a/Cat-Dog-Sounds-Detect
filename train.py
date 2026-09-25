"""Training and evaluation routines for the audio CNN."""

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CLASS_NAMES,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
)
from dataset import create_data_loaders
from model import AudioCNN


def get_device() -> torch.device:
    """Select CUDA when available, otherwise use the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[int, int, float]:
    """Return correct predictions, sample count, and accuracy percentage."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs.to(device))
            labels = labels.to(device)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total if total else 0.0
    return correct, total, accuracy


def train_model(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> tuple[AudioCNN, float]:
    """Train the CNN, evaluate it, and save a reusable checkpoint."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    torch.manual_seed(RANDOM_SEED)
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, test_loader = create_data_loaders(batch_size=batch_size)
    model = AudioCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {average_loss:.4f}")

    correct, total, accuracy = evaluate(model, test_loader, device)
    print(f"Test accuracy: {accuracy:.2f}% ({correct}/{total})")

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": list(CLASS_NAMES),
        },
        checkpoint_path,
    )
    print(f"Saved model to: {checkpoint_path}")
    return model, accuracy
