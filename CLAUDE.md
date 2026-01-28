# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Streamlit web application for automatic speech recognition (ASR) and translation using IBM's [Granite Speech](https://huggingface.co/collections/ibm-granite/granite-speech) models. Users upload audio files (WAV/MP3) to transcribe speech or translate to French, German, Spanish, or Portuguese.

## Directory Structure

The application is a single file Streamlit app (`streamlit_app.py`).

## Main Dependencies

- `transformers` - Hugging Face model loading (`AutoModelForSpeechSeq2Seq`, `AutoProcessor`, `PreTrainedTokenizerBase`)
- `torch` - Tensor operations
- `torchaudio` - Audio loading and resampling
- `streamlit` - Web user interface framework (`UploadedFile` for type hints)

## Architecture

### Components in `streamlit_app.py`

1. **Device Detection** (`get_device()`) - Auto-selects MPS (Apple Silicon) → CUDA → CPU
2. **Model Loading** (`load_model()`) - Loads Granite Speech 3.3 from Hugging Face with `@st.cache_resource` caching; uses bfloat16 on GPU/MPS, float32 on CPU; displays device name during loading
3. **Audio Preprocessing** (`load_and_preprocess_audio()`) - Converts audio to 16kHz mono (model requirement), handles stereo-to-mono conversion and resampling; includes try-finally for temp file cleanup
4. **Inference** (`transcribe_audio()`) - Constructs chat-formatted prompts with tokenizer chat template, runs generation with greedy decoding (num_beams=1)
5. **Formatting Helpers** (`format_duration()`, `format_size()`) - Convert nanoseconds and bytes to human-readable strings
6. **Metrics Display** (`display_metrics()`) - Displays all metrics using `st.metric` components in a grid layout

### Constants

- `MODEL_OPTIONS` - Dict mapping display names to Hugging Face model IDs
- `SYSTEM_PROMPT` - System prompt with knowledge cutoff and today's date
- `PROMPT_CHOICES` - List of transcription/translation prompts
- `SUPPORTED_FORMATS` - `["wav", "mp3"]`
- `DOWNLOAD_FIELDS` - Fields included in JSON export

### Type Imports

```python
from typing import Any
from streamlit.runtime.uploaded_file_manager import UploadedFile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, PreTrainedTokenizerBase
```

### Model Options

Granite Speech model is configurable via a radio button widget.

- [Granite Speech 3.3 8b](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)
- [Granite Speech 3.3 2b](https://huggingface.co/ibm-granite/granite-speech-3.3-2b)

### Supported Languages

- English (transcription)
- French (translation)
- German (translation)
- Spanish (translation)
- Portuguese (translation)

### UI Flow

```
Select model → Upload audio → Select prompt → Transcribe → View results → Download (clears results)
```

### Download

Include these items in the response JSON file for download.

- model (string): Model name
- response (string): The model's generated text response
- total_duration (integer): Time spent generating the response in nanoseconds
- load_duration (integer): Time spent loading the model in nanoseconds
- prompt_eval_count (integer): Number of audio segments processed
- prompt_eval_duration (integer): Time spent decoding audio in nanoseconds
- eval_count (integer): Number of words in the transcript
- eval_duration (integer): Time spent transcribing in nanoseconds

Use `time.perf_counter_ns()` to measure duration and return time in nanoseconds.

### Metrics

Display metrics using `st.metric` components in a grid layout:

**Model name as subheader**

**Row 1 - Key Results (4 columns)**
- Total Time
- Words
- Speed (with realtime factor delta)
- Audio Length

**Row 2 - Audio Info (3 columns)**
- File Size
- Format
- Segments

**Row 3 - Timing Breakdown (3 columns)**
- Model Load
- Audio Processing
- Transcription

### Behavior

- Results display immediately after transcription
- Clicking Download clears results (no session state persistence)
- Model loading shows device name in spinner (e.g., "Loading model on MPS...")

### Error Handling

- Invalid audio files raise `ValueError` with context message
- Temp files cleaned up via try-finally block regardless of success/failure
- Audio format validation handled by Streamlit file uploader

## Standards

- Type hints required on all functions
- pytest for testing (fixtures in `tests/conftest.py`)
- PEP 8 with 100 character lines
- pylint for static code analysis
- Error handling with try-finally for resource cleanup

## Test Data

Sample audio file available at `tests/data/audio/sample_10s.mp3` for testing.

## Commands

```bash
# Setup
python3.12 -m venv streamlit_env
source streamlit_env/bin/activate
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## Resources

- [Granite Speech Models](https://huggingface.co/collections/ibm-granite/granite-speech)
- [Technical Report](https://arxiv.org/abs/2505.08699)
- [Finetune on custom data](https://github.com/ibm-granite/granite-speech-models/blob/main/notebooks/fine_tuning_granite_speech.ipynb)
- [Two-Pass Spoken Question Answering](https://github.com/ibm-granite/granite-speech-models/blob/main/notebooks/two_pass_spoken_qa.ipynb)
