# 🐾 Cat & Dog Sound Classifier

A deep learning audio-classification application that predicts whether an uploaded audio sample belongs to a **cat 🐱 or a dog 🐶**.

The project uses **PyTorch and Torchaudio** for audio processing and model inference, with an interactive **Gradio** interface for testing the model.

---

## 🎬 Application Preview

![Cat and Dog Sound Classifier](assets/demo.png)

The interface allows the user to:

- select a sample cat or dog sound,
- upload a custom audio file,
- preview the selected audio,
- run the trained model,
- view the predicted animal,
- listen to the audio that was classified.

---

## 🚀 Features

- 🐱 Cat sound classification
- 🐶 Dog sound classification
- 🎵 Custom audio upload
- 🎧 Audio preview
- 📊 Audio preprocessing with Torchaudio
- 🧠 PyTorch-based classification model
- 🖼️ Predicted animal image display
- ⚡ Interactive Gradio web interface
- 📁 Built-in cat and dog audio samples

---

## 🧠 Project Workflow

```text
Cat / Dog Audio
       ↓
Load Audio
       ↓
Audio Preprocessing
       ↓
Convert Audio to Model Input
       ↓
Trained PyTorch Model
       ↓
Prediction
       ↓
Cat 🐱 or Dog 🐶
       ↓
Display Result in Gradio
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyTorch | Deep learning model |
| Torchaudio | Audio loading and processing |
| SoundFile | Audio file handling |
| NumPy | Numerical processing |
| Pillow | Image handling |
| Matplotlib | Visualization |
| Gradio | Interactive web interface |

---

## 📂 Project Structure

A typical structure for this project is:

```text
cat-dog-sound-classifier/
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
│
├── model/
│   └── cat_dog_model.pth
│
├── assets/
│   ├── demo.png
│   ├── cat.jpg
│   └── dog.jpg
│
├── samples/
│   ├── cats/
│   └── dogs/
│
└── data/
    ├── Cats/
    └── Dogs/
```

> Change the file and folder names above if your actual project uses different names.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cat-dog-sound-classifier.git
```

Move into the project directory:

```bash
cd cat-dog-sound-classifier
```

---

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
torch
torchaudio
soundfile
numpy
pillow
matplotlib
gradio
```

---

## ▶️ Run the Application

Run:

```bash
python app.py
```

Gradio will provide a local URL similar to:

```text
http://127.0.0.1:7860
```

Open the URL in your browser.

---

## 🎧 How to Use

1. Select one of the available sample audio files  
   **or**
2. Upload your own cat/dog audio file.
3. Preview the audio.
4. Click:

```text
✨ Identify the sound
```

5. The model processes the audio.
6. The predicted class is displayed.

Example:

```text
Prediction
🐱 Cat!
```

or

```text
Prediction
🐶 Dog!
```

---

## 🔊 Audio Loading

Audio files are loaded using Torchaudio:

```python
waveform, sample_rate = torchaudio.load(audio_path)
```

The returned values contain:

```text
waveform     → audio signal values
sample_rate  → number of audio samples per second
```

Example:

```text
waveform shape = [channels, samples]
```

The waveform is then preprocessed into the format expected by the trained model.

---

## 🧠 Model

The classifier is implemented using **PyTorch**.

The model receives processed audio features and outputs scores for two classes:

```text
0 → Cat
1 → Dog
```

Example inference flow:

```python
model.eval()

with torch.no_grad():
    output = model(audio_input)

prediction = torch.argmax(output, dim=1)
```

---

## 📊 Classes

The current model supports two classes:

| Label | Animal |
|---:|---|
| `0` | 🐱 Cat |
| `1` | 🐶 Dog |

---

## 🖥️ Interface

The Gradio interface contains two main sections:

### 1. Add your audio

Users can select an example sound or upload their own audio.

### 2. Your result

The application displays:

```text
Predicted animal image
        +
Classified audio
        +
Final prediction
```

---

## 📸 Example Result

### Input

```text
Cat audio
```

### Output

```text
🐱 Cat!
```

The application also displays the predicted animal image and the audio waveform/player.

---

## 🔮 Future Improvements

Possible improvements include:

- 🎙️ Real-time microphone recording
- 📈 Prediction confidence score
- 🐦 Support for more animal classes
- 📊 Training and validation graphs
- 🎵 Support for additional audio formats
- ⚡ Faster inference
- 🌐 Online deployment
- 📱 Mobile-friendly interface

---

## 👨‍💻 Author

**Mohd Danish**

AI / Machine Learning Developer

---

## ⭐ Support

If you find this project useful, consider giving the repository a **⭐ Star**.
