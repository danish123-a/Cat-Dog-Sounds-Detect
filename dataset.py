"""Dataset and data-loader construction for WAV classification."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from audio import audio_to_mel
from config import (
    BATCH_SIZE,
    CLASS_TO_LABEL,
    DATA_DIR,
    RANDOM_SEED,
    TEST_SIZE,
)


class AudioDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load and cache cat/dog Mel spectrograms from the dataset folders."""

    def __init__(self, data_dir: str | Path = DATA_DIR) -> None:
        self.samples: list[torch.Tensor] = []
        self.labels: list[int] = []
        data_dir = Path(data_dir)

        for class_name, label in CLASS_TO_LABEL.items():
            class_dir = data_dir / f"{class_name}s"
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Dataset folder not found: {class_dir}")

            audio_paths = sorted(class_dir.glob("*.wav"))
            for audio_path in audio_paths:
                self.samples.append(audio_to_mel(audio_path))
                self.labels.append(label)

        if not self.samples:
            raise ValueError(f"No .wav files found under {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.samples[index], self.labels[index]


def create_data_loaders(
    data_dir: str | Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    test_size: float = TEST_SIZE,
) -> tuple[DataLoader, DataLoader]:
    """Create reproducible training and testing data loaders."""
    dataset = AudioDataset(data_dir)
    if len(dataset) < 2:
        raise ValueError("At least two audio files are required for a train/test split.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    test_count = max(1, round(len(dataset) * test_size))
    train_count = len(dataset) - test_count
    if train_count == 0:
        raise ValueError("The dataset is too small for the requested test split.")

    train_set, test_set = random_split(
        dataset,
        [train_count, test_count],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
