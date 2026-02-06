import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import torch
import torchaudio
from streamlit.runtime.uploaded_file_manager import UploadedFile
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    PreTrainedTokenizerBase,
)

MODEL_OPTIONS = {
    "Granite Speech 3.3 8b": "ibm-granite/granite-speech-3.3-8b",
    "Granite Speech 3.3 2b": "ibm-granite/granite-speech-3.3-2b",
}
SYSTEM_PROMPT_TEMPLATE = """Knowledge Cutoff Date: April 2024.
Today's Date: {today}.
You are Granite, developed by IBM. You are a helpful AI assistant"""
PROMPT_CHOICES = [
    "Transcribe the speech to text",
    "Translate the speech to French",
    "Translate the speech to German",
    "Translate the speech to Spanish",
    "Translate the speech to Portuguese",
]
SUPPORTED_FORMATS = ["wav", "mp3", "m4a", "ogg", "flac", "webm", "aac"]


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner=False)
def load_model(
    model_id: str,
    device: str,
) -> tuple[AutoModelForSpeechSeq2Seq, AutoProcessor, PreTrainedTokenizerBase]:
    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, device_map=device, dtype=dtype
    )
    return model, processor, processor.tokenizer


def load_and_preprocess_audio(audio_file: UploadedFile) -> tuple[torch.Tensor, float]:
    try:
        wav, sr = torchaudio.load(io.BytesIO(audio_file.getvalue()))
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {e}") from e

    duration = wav.shape[1] / sr
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav, duration


@torch.inference_mode()
def transcribe_audio(
    wav: torch.Tensor,
    prompt: str,
    model: AutoModelForSpeechSeq2Seq,
    processor: AutoProcessor,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
) -> tuple[str, float]:
    start = time.perf_counter()
    today = datetime.today().strftime("%B %d, %Y")
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(today=today)},
        {"role": "user", "content": f"<|audio|>{prompt}"},
    ]
    text_prompt = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text_prompt, wav, device=device, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, num_beams=1)
    transcript = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    return transcript, round(time.perf_counter() - start, 2)


def display_metrics(result: dict[str, Any]) -> None:
    cols = st.columns(4)
    cols[0].metric("Model", result["model"].split("/")[-1])
    cols[1].metric("Audio Duration", f"{result['audio_duration']:.2f}s")
    cols[2].metric("Words", result["num_words"])
    cols[3].metric("Eval Duration", f"{result['eval_duration']}s")


def main() -> None:
    st.set_page_config(page_title="Granite Speech Pipeline", page_icon="🎙️")
    st.title("🎙️ Granite Speech Pipeline")
    st.write("Upload an audio file and try one of the prompts.")

    model_choice = st.radio("Select model", list(MODEL_OPTIONS.keys()), horizontal=True)
    device = get_device()
    with st.spinner(f"Loading model on {device.upper()}..."):
        model, processor, tokenizer = load_model(MODEL_OPTIONS[model_choice], device)

    audio_file = st.file_uploader(
        "Upload audio file",
        type=SUPPORTED_FORMATS,
        help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
    )
    if audio_file:
        st.audio(audio_file, format=f"audio/{Path(audio_file.name).suffix[1:]}")

    user_prompt = st.selectbox("Select prompt", PROMPT_CHOICES)

    if st.button("Transcribe", type="primary", disabled=not audio_file) and audio_file:
        with st.spinner("Transcribing audio..."):
            try:
                wav, audio_duration = load_and_preprocess_audio(audio_file)
                transcript, eval_duration = transcribe_audio(
                    wav, user_prompt, model, processor, tokenizer, device
                )
                result = {
                    "model": MODEL_OPTIONS[model_choice],
                    "audio_duration": audio_duration,
                    "transcript": transcript,
                    "num_words": len(transcript.split()),
                    "eval_duration": eval_duration,
                }
            except RuntimeError as e:
                st.error(str(e))
                return
            except Exception as e:
                st.exception(e)
                return

        st.subheader("Transcription")
        st.write(result["transcript"])
        display_metrics(result)
        st.download_button(
            "Download",
            json.dumps(result, indent=2),
            f"{Path(audio_file.name).stem}_transcription.json",
            "application/json",
        )


if __name__ == "__main__":
    main()
