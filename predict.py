"""Load a trained checkpoint and classify an audio file."""

from pathlib import Path

import torch

from audio import audio_to_mel
from config import CHECKPOINT_PATH, CLASS_NAMES
from model import AudioCNN
from train import get_device


def predict_audio(
    file_path: str | Path,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
) -> str:
    """Return the predicted class name for one WAV file."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}. Train the model first."
        )

    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = checkpoint.get("class_names", list(CLASS_NAMES))
    model = AudioCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mel = audio_to_mel(file_path).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted_index = model(mel).argmax(dim=1).item()
    return class_names[predicted_index]
