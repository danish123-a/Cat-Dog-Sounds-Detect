"""Shared configuration for the cat-vs-dog audio classifier."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / ".kaggle" / "DvC"
CHECKPOINT_PATH = PROJECT_DIR / "artifacts" / "audio_cnn.pth"
DEFAULT_PREDICTION_AUDIO = DATA_DIR / "Cats" / "cat0001.wav"

CLASS_NAMES = ("Cat", "Dog")
CLASS_TO_LABEL = {name: index for index, name in enumerate(CLASS_NAMES)}

SAMPLE_RATE = 22_050
N_MELS = 128
MAX_FRAMES = 300
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
RANDOM_SEED = 42

CAT_IMAGE_PATH = PROJECT_DIR / "Cat03.jpg"
DOG_IMAGE_PATH = PROJECT_DIR / "images.jpg"
