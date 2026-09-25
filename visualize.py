"""Display the cat or dog image associated with a prediction."""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from config import CAT_IMAGE_PATH, DOG_IMAGE_PATH


def show_prediction(prediction: str) -> None:
    """Show the corresponding cat/dog image when one is available."""
    image_paths = {"Cat": CAT_IMAGE_PATH, "Dog": DOG_IMAGE_PATH}
    image_path: Path | None = image_paths.get(prediction)
    if image_path is None:
        raise ValueError(f"Unknown class name: {prediction}")
    if not image_path.is_file():
        print(f"Image not found; skipping display: {image_path}")
        return

    with Image.open(image_path) as image:
        plt.imshow(image)
    plt.title(f"Prediction: {prediction}")
    plt.axis("off")
    plt.show()
