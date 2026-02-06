# Granite Speech Pipeline

A Streamlit web application for automatic speech recognition (ASR) and translation using IBM's Granite Speech models.

## Features

- **Model Selection**: Choose between Granite Speech 3.3 8B or 2B models
- **Transcription**: Convert speech to text
- **Translation**: Translate speech to French, German, Spanish, or Portuguese
- **Supported Formats**: WAV, MP3, M4A, OGG, FLAC, WebM, AAC
- **Performance Metrics**: View model, audio duration, word count, and eval duration
- **JSON Export**: Download results as JSON

## Hardware Support

Automatically detects and uses the best available device:
- Apple Silicon (MPS)
- NVIDIA GPU (CUDA)
- CPU (fallback)

## Installation

```bash
# Set up Python virtual environment
python3.12 -m venv streamlit_env

# Activate the virtual environment
source streamlit_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

## Usage

1. Select a model (8B for higher quality, 2B for faster inference)
2. Upload an audio file (WAV, MP3, M4A, OGG, FLAC, WebM, AAC)
3. Choose a prompt (transcribe or translate)
4. Click "Transcribe" to process
5. View results and metrics
6. Download results as JSON

## Development

```bash
ruff check .          # Lint
ruff format .         # Format
pyright               # Typecheck
pytest                # Test
```

## Resources

- [Granite Speech Models](https://huggingface.co/collections/ibm-granite/granite-speech)
- [Technical Report](https://arxiv.org/abs/2505.08699)
