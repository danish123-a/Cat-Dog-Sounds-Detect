# Cat vs. Dog Audio CNN

The Python code is split into small modules so data preparation, the model, training, and inference can be worked on separately.

## Project files

- `config.py` — paths and shared settings
- `audio.py` — WAV loading and Mel-spectrogram preprocessing
- `dataset.py` — dataset and train/test data loaders
- `model.py` — CNN architecture
- `train.py` — training, evaluation, and checkpoint saving
- `predict.py` — inference from a saved checkpoint
- `visualize.py` — optional cat/dog image display
- `main.py` — command-line entry point
- `gradio_app.py` — colorful two-column browser interface

The dataset is expected at `.kaggle/DvC/Cats` and `.kaggle/DvC/Dogs` relative to this project. Training saves the checkpoint to `artifacts/audio_cnn.pth`.

## Run

Install the packages in `requirements.txt`. Install the PyTorch and torchaudio builds that match your CPU/CUDA setup if needed. Then, from this folder:

- Train: `python main.py --train`
- Predict the default sample `.kaggle/DvC/Cats/cat0001.wav` and automatically show the matching cat/dog image (trains first if there is no checkpoint): `python main.py`
- Predict a different audio file: `python main.py --predict .kaggle/DvC/Dogs/dog0001.wav`
- Predict without displaying an image: `python main.py --predict .kaggle/DvC/Cats/cat0001.wav --no-image`

You can also run `python train.py` to train directly.

## Gradio interface

Install `gradio` from `requirements.txt`, then launch the browser interface from this folder:

- Start the app: `python gradio_app.py`
- Upload a WAV file in the left column, listen to it, then click **Identify the sound**.
- The right column shows the matching cat/dog picture, the audio that was classified, and a highlighted prediction.

The included `.kaggle/DvC/Cats/cat0001.wav` is preloaded as a sample. Train first with `python main.py --train` if `artifacts/audio_cnn.pth` does not exist.
