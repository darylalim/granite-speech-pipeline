# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app for automatic speech recognition and translation using IBM's [Granite Speech](https://huggingface.co/collections/ibm-granite/granite-speech) models.

## Setup

```bash
python3.12 -m venv streamlit_env
source streamlit_env/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Commands

- **Lint**: `ruff check .`
- **Format**: `ruff format .`
- **Typecheck**: `pyright`
- **Test**: `pytest`

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- `RuntimeError` for known transcription failures (no custom exception class)
- isort with combine-as-imports (configured in `pyproject.toml`)

## Dependencies

- `transformers` - Hugging Face model loading
- `torch` - Tensor operations
- `torchaudio` - Audio loading and resampling
- `streamlit` - Web user interface
- `ruff` — linting/formatting (dev)
- `pyright` — type checking (dev)
- `pytest` — testing (dev)

## Configuration

`pyproject.toml` — ruff isort (`combine-as-imports`) and pyright (`pythonVersion = "3.12"`).

## Architecture

### Entry Point

`streamlit_app.py` - single-file app.

### Models

Granite Speech models selected via `st.radio`:

- [Granite Speech 3.3 2b](https://huggingface.co/ibm-granite/granite-speech-3.3-2b)
- [Granite Speech 3.3 8b](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)

### Languages

- English (transcription)
- French (translation)
- German (translation)
- Spanish (translation)
- Portuguese (translation)

### Performance

- Use best available device: MPS > CUDA > CPU
- `@st.cache_resource` to cache models
- `@torch.inference_mode()` on inference functions
- `io.BytesIO` for in-memory audio loading (no temp files)
- `dtype` is bfloat16 on MPS and CUDA, float32 on CPU
- `time.perf_counter()` for timing (fractional seconds)

### Audio Formats

Supported: wav, mp3, m4a, ogg, flac, webm, aac

### Error Handling

- `RuntimeError` caught explicitly for transcription failures
- Unexpected exceptions shown with `st.exception()` for debugging

### JSON Download

Fields in the downloadable JSON via `st.download_button`:

- `model` (string) — model name
- `audio_duration` (float) — audio duration in seconds
- `transcript` (string) — generated text
- `num_words` (int) — word count
- `eval_duration` (float) — transcription time in seconds (rounded to 2 decimal places)

### Metrics

`st.metric` displays all JSON fields except transcript.

### Tests

`tests/test_streamlit_app.py` — unit tests for device detection, supported formats, audio loading, and error handling. Run with `pytest`.

## Resources

- [Granite Speech Models](https://huggingface.co/collections/ibm-granite/granite-speech)
- [Technical Report](https://arxiv.org/abs/2505.08699)
- [Finetune on custom data](https://github.com/ibm-granite/granite-speech-models/blob/main/notebooks/fine_tuning_granite_speech.ipynb)
- [Two-Pass Spoken Question Answering](https://github.com/ibm-granite/granite-speech-models/blob/main/notebooks/two_pass_spoken_qa.ipynb)
