"""Audio loading and fixed-size Mel-spectrogram preprocessing."""

from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from config import MAX_FRAMES, N_MELS, SAMPLE_RATE


def audio_to_mel(file_path: str | Path) -> torch.Tensor:
    """Convert a WAV file into a [1, n_mels, max_frames] tensor."""
    audio_array, sample_rate = sf.read(str(file_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio_array).transpose(0, 1)
    waveform = waveform.mean(dim=0, keepdim=True)

    if waveform.numel() == 0:
        raise ValueError(f"Audio file is empty: {file_path}")

    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
    )
    mel = mel_transform(waveform)

    frame_count = mel.shape[-1]
    if frame_count < MAX_FRAMES:
        mel = F.pad(mel, (0, MAX_FRAMES - frame_count))
    else:
        mel = mel[..., :MAX_FRAMES]

    return mel
