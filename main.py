"""Command-line entry point for training and audio prediction."""

import argparse
from pathlib import Path

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DEFAULT_PREDICTION_AUDIO,
    EPOCHS,
    LEARNING_RATE,
)
from predict import predict_audio
from train import train_model
from visualize import show_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cat vs. dog audio classifier")
    parser.add_argument("--train", action="store_true", help="train a new model")
    parser.add_argument(
        "--predict",
        type=Path,
        default=DEFAULT_PREDICTION_AUDIO,
        help=f"classify a WAV audio file (default: {DEFAULT_PREDICTION_AUDIO})",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="do not display the matching class image after prediction",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    should_train = args.train or not CHECKPOINT_PATH.exists()

    if should_train:
        train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    prediction = predict_audio(args.predict)
    print(f"Prediction: {prediction}")
    if not args.no_image:
        show_prediction(prediction)


if __name__ == "__main__":
    main()
