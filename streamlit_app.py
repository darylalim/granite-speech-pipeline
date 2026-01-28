import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import torch
import torchaudio
from streamlit.runtime.uploaded_file_manager import UploadedFile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, PreTrainedTokenizerBase

MODEL_OPTIONS = {
    "Granite Speech 3.3 8b": "ibm-granite/granite-speech-3.3-8b",
    "Granite Speech 3.3 2b": "ibm-granite/granite-speech-3.3-2b",
}
SYSTEM_PROMPT = f"""Knowledge Cutoff Date: April 2024.
Today's Date: {datetime.today().strftime("%B %d, %Y")}.
You are Granite, developed by IBM. You are a helpful AI assistant"""
PROMPT_CHOICES = [
    "Transcribe the speech to text",
    "Translate the speech to French",
    "Translate the speech to German",
    "Translate the speech to Spanish",
    "Translate the speech to Portuguese",
]
SUPPORTED_FORMATS = ["wav", "mp3"]
DOWNLOAD_FIELDS = ["model", "response", "total_duration", "load_duration",
                   "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration"]


def get_device() -> str:
    """Detect the best available device: MPS -> CUDA -> CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner=False)
def load_model(
    model_id: str, device: str,
) -> tuple[AutoModelForSpeechSeq2Seq, AutoProcessor, PreTrainedTokenizerBase, int]:
    """Load the Granite Speech model and processor."""
    start = time.perf_counter_ns()
    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, device_map=device, dtype=dtype)
    return model, processor, processor.tokenizer, time.perf_counter_ns() - start


def load_and_preprocess_audio(audio_file: UploadedFile) -> dict[str, Any]:
    """Load audio file and convert to 16kHz mono."""
    start = time.perf_counter_ns()
    audio_bytes = audio_file.getvalue()
    suffix = Path(audio_file.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        wav, sr = torchaudio.load(tmp_path, normalize=True)
        duration = wav.shape[1] / sr
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {e}") from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    return {
        "wav": wav, "file_size": len(audio_bytes), "file_format": suffix[1:].upper(),
        "audio_duration": duration, "prompt_eval_count": 1,
        "prompt_eval_duration": time.perf_counter_ns() - start,
    }


def transcribe_audio(
    wav: torch.Tensor,
    prompt: str,
    model: AutoModelForSpeechSeq2Seq,
    processor: AutoProcessor,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
) -> dict[str, Any]:
    """Transcribe or translate audio using the Granite Speech model."""
    start = time.perf_counter_ns()
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<|audio|>{prompt}"},
    ]
    text_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = processor(text_prompt, wav, device=device, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)
    text = tokenizer.decode(outputs[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return {
        "response": text,
        "eval_count": len(text.split()),
        "eval_duration": time.perf_counter_ns() - start,
    }


def format_duration(ns: int) -> str:
    """Format nanoseconds as human-readable string."""
    if ns >= 1e9:
        return f"{ns / 1e9:.2f} s"
    return f"{ns / 1e6:.2f} ms" if ns >= 1e6 else f"{ns / 1e3:.2f} µs"


def format_size(b: int) -> str:
    """Format bytes as human-readable string."""
    if b >= 1024 * 1024:
        return f"{b / (1024 * 1024):.2f} MB"
    return f"{b / 1024:.2f} KB" if b >= 1024 else f"{b} B"


def display_metrics(r: dict[str, Any]) -> None:
    """Display metrics summary using st.metric components."""
    wps = r["eval_count"] / (r["eval_duration"] / 1e9) if r["eval_duration"] > 0 else 0
    realtime_factor = r["audio_duration"] / (r["eval_duration"] / 1e9) if r["eval_duration"] > 0 else 0

    st.subheader(r["model"].split("/")[-1])

    # Row 1: Key results
    cols = st.columns(4)
    cols[0].metric("Total Time", format_duration(r["total_duration"]))
    cols[1].metric("Words", r["eval_count"])
    cols[2].metric("Speed", f"{wps:.1f} w/s", delta=f"{realtime_factor:.1f}x realtime")
    cols[3].metric("Audio Length", f"{r['audio_duration']:.1f}s")

    st.divider()

    # Row 2: Audio info
    cols = st.columns(3)
    cols[0].metric("File Size", format_size(r["file_size"]))
    cols[1].metric("Format", r["file_format"])
    cols[2].metric("Segments", r["prompt_eval_count"])

    # Row 3: Timing breakdown
    cols = st.columns(3)
    cols[0].metric("Model Load", format_duration(r["load_duration"]))
    cols[1].metric("Audio Processing", format_duration(r["prompt_eval_duration"]))
    cols[2].metric("Transcription", format_duration(r["eval_duration"]))


def main() -> None:
    st.set_page_config(page_title="Granite Speech Pipeline", page_icon="🎙️")
    st.title("🎙️ Granite Speech Pipeline")
    st.write("Upload an audio file and try one of the prompts.")

    model_choice = st.radio("Select model", list(MODEL_OPTIONS.keys()), horizontal=True)
    device = get_device()
    with st.spinner(f"Loading model on {device.upper()}..."):
        model, processor, tokenizer, load_duration = load_model(MODEL_OPTIONS[model_choice], device)

    audio_file = st.file_uploader("Upload audio file", type=SUPPORTED_FORMATS,
                                   help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")
    if audio_file:
        st.audio(audio_file, format=f"audio/{Path(audio_file.name).suffix[1:]}")

    user_prompt = st.selectbox("Select prompt", PROMPT_CHOICES)

    if st.button("Transcribe", type="primary", disabled=not audio_file) and audio_file:
        with st.spinner("Transcribing audio..."):
            total_start = time.perf_counter_ns()
            audio = load_and_preprocess_audio(audio_file)
            result = transcribe_audio(audio["wav"], user_prompt, model, processor, tokenizer, device)
            r = {
                "model": MODEL_OPTIONS[model_choice], **result,
                "total_duration": time.perf_counter_ns() - total_start,
                "load_duration": load_duration, "audio_filename": audio_file.name,
                **{k: audio[k] for k in ["prompt_eval_count", "prompt_eval_duration",
                                          "file_size", "file_format", "audio_duration"]},
            }

        st.subheader("Transcription")
        st.write(r["response"])
        display_metrics(r)
        st.download_button("Download", json.dumps({k: r[k] for k in DOWNLOAD_FIELDS}, indent=2),
                           f"{Path(r['audio_filename']).stem}_transcription.json", "application/json")


if __name__ == "__main__":
    main()
