import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_ID = "ibm-granite/granite-speech-3.3-2b"
TODAY_DATE = datetime.today().strftime("%B %d, %Y")
SYSTEM_PROMPT = f"""Knowledge Cutoff Date: April 2024.
Today's Date: {TODAY_DATE}.
You are Granite, developed by IBM. You are a helpful AI assistant"""

PROMPT_CHOICES = [
    "Transcribe the speech to text",
    "Translate the speech to French",
    "Translate the speech to German",
    "Translate the speech to Spanish",
    "Translate the speech to Portuguese",
]

SUPPORTED_FORMATS = ["wav", "mp3"]


def get_device() -> str:
    """Detect the best available device: MPS -> CUDA -> CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


@st.cache_resource(show_spinner="Loading Granite Speech model...")
def load_model():
    """Load the Granite Speech model and processor."""
    device = get_device()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        device_map=device,
        dtype=dtype,
    )

    return model, processor, tokenizer, device


def load_and_preprocess_audio(audio_file) -> tuple[torch.Tensor, int]:
    """Load audio file and convert to 16kHz mono."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp:
        tmp.write(audio_file.getvalue())
        tmp_path = tmp.name

    wav, sr = torchaudio.load(tmp_path, normalize=True)

    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    Path(tmp_path).unlink(missing_ok=True)

    return wav, sr


def transcribe_audio(
    wav: torch.Tensor,
    user_prompt: str,
    model,
    processor,
    tokenizer,
    device: str,
) -> str:
    """Transcribe or translate audio using the Granite Speech model."""
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<|audio|>{user_prompt}"},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    model_inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)

    model_outputs = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False,
        num_beams=1,
    )

    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = model_outputs[0, num_input_tokens:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return output_text


def main():
    st.set_page_config(page_title="Granite Speech Pipeline", page_icon="🎙️")

    st.title("🎙️ Granite Speech Pipeline")
    st.write("Upload an audio file and try one of the prompts.")

    model, processor, tokenizer, device = load_model()

    audio_file = st.file_uploader(
        "Upload audio file",
        type=SUPPORTED_FORMATS,
        help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
    )

    if audio_file is not None:
        st.audio(audio_file, format=f"audio/{Path(audio_file.name).suffix[1:]}")

    user_prompt = st.selectbox("Select prompt", options=PROMPT_CHOICES)

    if "transcription" not in st.session_state:
        st.session_state.transcription = None
    if "audio_filename" not in st.session_state:
        st.session_state.audio_filename = None

    if st.button("Transcribe", type="primary", disabled=audio_file is None):
        if audio_file is not None:
            with st.spinner("Transcribing audio..."):
                wav, sr = load_and_preprocess_audio(audio_file)
                text = transcribe_audio(wav, user_prompt, model, processor, tokenizer, device)
                st.session_state.transcription = text
                st.session_state.audio_filename = audio_file.name

    if st.session_state.transcription:
        st.subheader("Transcription")
        st.write(st.session_state.transcription)

        filename = f"{Path(st.session_state.audio_filename).stem}_transcription.txt"
        st.download_button(
            label="Download",
            data=st.session_state.transcription,
            file_name=filename,
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
