import io
import tempfile
from datetime import datetime
from pathlib import Path

import langid
import streamlit as st
import torch
import torchaudio
from punctuators.models import PunctCapSegModelONNX
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

SUPPORTED_FORMATS = ["wav", "mp3", "flac", "ogg", "m4a", "wma"]

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
        dtype=dtype
    )

    return model, processor, tokenizer, device

@st.cache_resource(show_spinner="Loading punctuation model...")
def load_punctuation_model():
    """Load the punctuation and capitalization model."""
    return PunctCapSegModelONNX.from_pretrained("pcs_en")

def load_and_preprocess_audio(audio_file) -> tuple[torch.Tensor, int]:
    """Load audio file and convert to 16kHz mono."""
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp:
        tmp.write(audio_file.getvalue())
        tmp_path = tmp.name

    # Load audio
    wav, sr = torchaudio.load(tmp_path, normalize=True)

    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    # Clean up temp file
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
    # Build chat messages
    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<|audio|>{user_prompt}"},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    # Process inputs
    model_inputs = processor(prompt, wav, device=device, return_tensors="pt").to(device)

    # Generate output
    model_outputs = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False,
        num_beams=1,
    )

    # Decode output (exclude input tokens)
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = model_outputs[0, num_input_tokens:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return output_text

def apply_punctuation(text: str, pc_model) -> str:
    """Apply punctuation and capitalization for English text."""
    result = pc_model.infer([text])
    text = " ".join(result[0])
    text = text.replace("<unk>", " ").replace("<Unk>", " ")
    return text

def format_as_srt(text: str) -> str:
    """Format transcription as SRT subtitle format."""
    lines = []
    sentences = text.replace("? ", "?\n").replace(". ", ".\n").replace("! ", "!\n").split("\n")
    sentences = [s.strip() for s in sentences if s.strip()]

    for i, sentence in enumerate(sentences, start=1):
        # Simple time estimation (3 seconds per sentence)
        start_time = (i - 1) * 3
        end_time = i * 3
        start_str = f"{start_time // 3600:02d}:{(start_time % 3600) // 60:02d}:{start_time % 60:02d},000"
        end_str = f"{end_time // 3600:02d}:{(end_time % 3600) // 60:02d}:{end_time % 60:02d},000"

        lines.append(f"{i}")
        lines.append(f"{start_str} --> {end_str}")
        lines.append(sentence)
        lines.append("")

    return "\n".join(lines)

def get_download_filename(original_name: str, extension: str) -> str:
    """Generate download filename based on original audio filename."""
    stem = Path(original_name).stem
    return f"{stem}_transcription.{extension}"

def main():
    st.set_page_config(page_title="Granite Speech Pipeline", page_icon="🎙️")

    st.title("🎙️ Granite Speech Pipeline")
    st.write("Upload an audio file and try one of the prompts.")

    # Load models
    model, processor, tokenizer, device = load_model()
    pc_model = load_punctuation_model()

    st.caption(f"Running on: **{device.upper()}**")

    # File upload
    audio_file = st.file_uploader(
        "Upload audio file",
        type=SUPPORTED_FORMATS,
        help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
    )

    # Audio playback
    if audio_file is not None:
        st.audio(audio_file, format=f"audio/{Path(audio_file.name).suffix[1:]}")

    # Prompt selection
    user_prompt = st.selectbox("Select prompt", options=PROMPT_CHOICES)

    # Initialize session state for transcription
    if "transcription" not in st.session_state:
        st.session_state.transcription = None
    if "audio_filename" not in st.session_state:
        st.session_state.audio_filename = None

    # Transcribe button
    if st.button("Transcribe", type="primary", disabled=audio_file is None):
        if audio_file is not None:
            with st.spinner("Transcribing audio..."):
                # Load and preprocess audio
                wav, sr = load_and_preprocess_audio(audio_file)

                # Transcribe
                text = transcribe_audio(wav, user_prompt, model, processor, tokenizer, device)

                # Apply punctuation for English
                if langid.classify(text)[0] == "en":
                    text = apply_punctuation(text, pc_model)

                # Store in session state
                st.session_state.transcription = text
                st.session_state.audio_filename = audio_file.name

    # Display transcription
    if st.session_state.transcription:
        st.subheader("Transcription")
        st.text_area(
            "Result",
            value=st.session_state.transcription,
            height=200,
            label_visibility="collapsed",
        )

        # Download section
        st.subheader("Download")
        col1, col2 = st.columns([1, 2])

        with col1:
            download_format = st.selectbox(
                "Format",
                options=["txt", "srt"],
                help="TXT for plain text, SRT for subtitle format",
            )

        with col2:
            # Prepare download content
            if download_format == "txt":
                download_content = st.session_state.transcription
                mime_type = "text/plain"
            else:
                download_content = format_as_srt(st.session_state.transcription)
                mime_type = "text/srt"

            filename = get_download_filename(st.session_state.audio_filename, download_format)

            st.download_button(
                label=f"Download as .{download_format}",
                data=download_content,
                file_name=filename,
                mime=mime_type,
            )


if __name__ == "__main__":
    main()
