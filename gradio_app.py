"""Colorful Gradio interface for cat/dog audio prediction."""

from pathlib import Path

import gradio as gr

from config import CAT_IMAGE_PATH, DATA_DIR, DEFAULT_PREDICTION_AUDIO, DOG_IMAGE_PATH
from predict import predict_audio

VOICE_SAMPLES_DIR = Path(__file__).resolve().parent / "voice_samples(cats and dogs)"
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


CUSTOM_CSS = """
:root {
    --page: #f4f6ff;
    --ink: #202445;
    --muted: #737b9c;
    --purple: #7057f5;
    --pink: #ff6eaa;
}
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    background:
        radial-gradient(ellipse at 5% 0%, rgba(255, 184, 215, .28), transparent 34%),
        radial-gradient(ellipse at 95% 4%, rgba(138, 126, 255, .22), transparent 34%),
        var(--page) !important;
    color: var(--ink);
}
#hero {
    padding: 34px 36px 26px;
    border-radius: 28px;
    margin: 18px 0 22px;
    color: white;
    background: linear-gradient(120deg, #5545d9 0%, #8460f5 52%, #ed71b5 100%);
    box-shadow: 0 18px 42px rgba(92, 75, 205, .22);
}
#hero h1 { margin: 0 0 8px; font-size: 2.25rem; letter-spacing: -.04em; }
#hero p { margin: 0; color: rgba(255,255,255,.88); font-size: 1.05rem; }
.panel {
    padding: 23px !important;
    border: 1px solid rgba(108, 112, 160, .12) !important;
    border-radius: 24px !important;
    background: rgba(255,255,255,.91) !important;
    box-shadow: 0 12px 34px rgba(40, 48, 92, .08) !important;
}
.panel-title { margin: 0 0 4px; color: #272a4a; font-size: 1.25rem; font-weight: 750; }
.panel-note { margin: 0 0 18px; color: var(--muted); font-size: .92rem; }
#upload-audio { border: 2px dashed #b9b1ff !important; border-radius: 18px !important; }
#submit-prediction {
    min-height: 54px;
    border: 0 !important;
    border-radius: 16px !important;
    color: white !important;
    background: linear-gradient(100deg, #6751ec, #a14ee7 62%, #e75fa5) !important;
    box-shadow: 0 10px 22px rgba(121, 76, 222, .24);
    font-size: 1rem !important;
    font-weight: 750 !important;
    transition: transform .15s ease, box-shadow .15s ease;
}
#submit-prediction:hover { transform: translateY(-2px); box-shadow: 0 14px 26px rgba(121, 76, 222, .3); }
#result-image { overflow: hidden; border-radius: 19px !important; background: #f2f2fb !important; }
#prediction-result { min-height: 90px; }
.result-card {
    display: flex; align-items: center; gap: 15px;
    padding: 17px 19px; border-radius: 18px;
    background: linear-gradient(110deg, #f1edff, #fff0f7);
    border: 1px solid #e6ddff;
}
.result-emoji { font-size: 2.1rem; }
.result-label { color: #6d7190; text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; font-weight: 800; }
.result-name { color: #4f3fc6; font-size: 1.55rem; font-weight: 850; }
.result-help { margin-top: 3px; color: #777d98; font-size: .88rem; }
.error-card { padding: 16px; border-radius: 14px; color: #8f3151; background: #fff0f4; border: 1px solid #ffd1df; }
footer { display: none !important; }
@media (max-width: 700px) {
    #hero { padding: 26px 23px; }
    #hero h1 { font-size: 1.8rem; }
    .panel { padding: 17px !important; }
}
"""


def get_sample_audio_choices() -> list[tuple[str, str]]:
    """Read named cat/dog samples from the supplied folders and fill up to four."""
    choices: list[tuple[str, str]] = []
    for class_name, sample_folder in (
        ("Cat", VOICE_SAMPLES_DIR / "cats_voice"),
        ("Dog", VOICE_SAMPLES_DIR / "dogs_voice"),
    ):
        audio_files = sorted(
            path
            for path in sample_folder.iterdir()
            if sample_folder.is_dir() and path.is_file()
            and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
        ) if sample_folder.is_dir() else []

        # Use the supplied sample folder first, then the existing class dataset
        # only when needed to provide four selectable examples per animal.
        dataset_folder = DATA_DIR / f"{class_name}s"
        if len(audio_files) < 4 and dataset_folder.is_dir():
            known_paths = {path.resolve() for path in audio_files}
            extra_files = sorted(
                path for path in dataset_folder.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
                and path.resolve() not in known_paths
            )
            audio_files.extend(extra_files[:4 - len(audio_files)])

        for sample_number, audio_path in enumerate(audio_files, start=1):
            label = f"{class_name} audio {sample_number} — {audio_path.name}"
            choices.append((label, str(audio_path)))
    return choices


def select_sample_audio(audio_path: str | None) -> str | None:
    """Load a built-in sample into the single audio upload/playback control."""
    return audio_path


def classify_audio(audio_path: str | None) -> tuple[str | None, str | None, str]:
    """Run inference and return the matching image, playable audio, and result card."""
    if not audio_path:
        return None, None, '<div class="error-card">Choose a WAV audio file first.</div>'

    try:
        prediction = predict_audio(audio_path)
    except FileNotFoundError as error:
        return None, audio_path, f'<div class="error-card">{error}</div>'
    except Exception as error:
        return None, audio_path, (
            '<div class="error-card">Could not process this audio. '
            f'Please upload a WAV file.<br><small>{error}</small></div>'
        )

    image_path: Path = CAT_IMAGE_PATH if prediction == "Cat" else DOG_IMAGE_PATH
    if prediction == "Cat":
        emoji, note = "🐱", "The model thinks this audio belongs to a cat."
    else:
        emoji, note = "🐶", "The model thinks this audio belongs to a dog."

    result_html = (
        '<div class="result-card">'
        f'<div class="result-emoji">{emoji}</div>'
        '<div><div class="result-label">Prediction</div>'
        f'<div class="result-name">{prediction}!</div>'
        f'<div class="result-help">{note}</div></div></div>'
    )
    return str(image_path), audio_path, result_html


def build_app() -> gr.Blocks:
    """Build and return the interactive two-column application."""
    with gr.Blocks(
        title="PawSound • Cat or Dog?",
    ) as app:
        gr.HTML(
            """<div id="hero">
                <h1>🐾 Cat & Dog Sounds</h1>
                <p>Drop in a cat or dog sound and let the model guess which one it is.</p>
            </div>"""
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_classes=["panel"]):
                gr.HTML('<div class="panel-title">1. Add your audio</div>')
                gr.HTML('<p class="panel-note">Choose a cat/dog sample or upload your own audio in the player below.</p>')
                sample_choices = get_sample_audio_choices()
                sample_values = [choice[1] for choice in sample_choices]
                default_sample = (
                    str(DEFAULT_PREDICTION_AUDIO)
                    if str(DEFAULT_PREDICTION_AUDIO) in sample_values
                    else (sample_values[0] if sample_values else None)
                )
                sample_selector = gr.Dropdown(
                    choices=sample_choices,
                    value=default_sample,
                    label="Cat and dog audio samples",
                    info="Choose a named sample, or upload your own audio below.",
                )
                audio_input = gr.Audio(
                    value=default_sample,
                    sources=["upload"],
                    type="filepath",
                    label="Selected audio preview · upload or drop audio here",
                )
                sample_selector.change(
                    fn=select_sample_audio,
                    inputs=sample_selector,
                    outputs=audio_input,
                    show_progress="hidden",
                )
                submit_button = gr.Button(
                    "✨  Identify the sound",
                    variant="primary",
                    elem_id="submit-prediction",
                )

            with gr.Column(scale=6, elem_classes=["panel"]):
                gr.HTML('<div class="panel-title">2. Your result</div>')
                gr.HTML('<p class="panel-note">The matching animal picture and your audio appear here.</p>')
                result_image = gr.Image(
                    label="Predicted animal",
                    type="filepath",
                    height=300,
                    interactive=False,
                    elem_id="result-image",
                )
                result_audio = gr.Audio(
                    type="filepath",
                    label="Audio that was classified",
                    interactive=False,
                )
                result_text = gr.HTML(
                    '<div class="result-card"><div class="result-emoji">✨</div>'
                    '<div><div class="result-label">Prediction</div>'
                    '<div class="result-name">Ready when you are</div>'
                    '<div class="result-help">Upload a sound and press Identify.</div>'
                    '</div></div>',
                    elem_id="prediction-result",
                )
                submit_button.click(
                    fn=classify_audio,
                    inputs=audio_input,
                    outputs=[result_image, result_audio, result_text],
                    show_progress="minimal",
                )

        gr.HTML(
            '<div style="text-align:center;color:#858aa5;padding:18px 0 28px;'
            'font-size:.85rem">Audio stays in this local app · WAV files recommended</div>'
        )

    return app


if __name__ == "__main__":
    build_app().launch(
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="pink",
            neutral_hue="slate",
        ),
        css=CUSTOM_CSS,
    )
