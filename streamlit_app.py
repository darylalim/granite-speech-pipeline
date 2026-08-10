# Copyright 2026 Daryl Lim
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import math
import warnings
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, NotRequired, TypedDict

# mlx 0.32.0 dropped mlx/core/*.pyi while still shipping py.typed, so ty resolves
# the package but not the compiled `core` extension. Drop the suppression once
# upstream ships stubs again.
import mlx.core as mx  # ty: ignore[unresolved-import]
import streamlit as st
import torch
import torchaudio
from mlx import nn
from mlx_audio.stt.utils import load_model as _load_stt_model
from mlx_audio.vad.utils import load_model as _load_mlx_vad_model
from silero_vad import get_speech_timestamps, load_silero_vad
from streamlit.runtime.uploaded_file_manager import UploadedFile
from torchcodec.decoders import AudioDecoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings(
    "ignore", message="An output with one or more elements was resized"
)
MODEL_ID = "divydeep/granite-speech-4.1-2b-mlx-8bit"
# Pinned: MODEL_ID is a personal-account conversion, not an org-maintained
# repo, so an unpinned main could change weights under us on any cold cache.
MODEL_REVISION = "8b45a92464686505ee42658cdbbd1946c2762b5b"
GUARDIAN_MODEL_ID = "ibm-granite/granite-guardian-hap-125m"
# Silero VAD v6 in MLX form. Its 16 kHz weights are bit-exact with the
# `silero-vad` PyPI checkpoint the PyTorch fallback loads, so the two backends
# agree by construction rather than by luck — which is exactly what the pin
# below protects.
MLX_VAD_REPO = "mlx-community/silero-vad-v6"
# Pinned for a different reason than MODEL_REVISION, which guards against a
# personal account re-uploading. This repo is org-maintained, but mlx_audio's
# VAD loader takes `strict=False` and *cannot* do otherwise: the v6 conversion
# legitimately ships no `vad_8k.*` keys, so a strict load would reject it
# outright. Non-strict loading means renamed `vad_16k.*` keys would leave those
# layers randomly initialised — and the model would still build, still pass
# _supports_mlx_vad, still run clean, and return confident nonsense. That is
# the same silent-garbage mechanism documented for the 8 kHz branch in
# silero_vad(), and a content hash is the only thing that rules it out.
MLX_VAD_REVISION = "2ebf4a5e10726a2e78ddd4d70eedfb6f1c33eb06"
SOURCE_LANGUAGES = [
    "English",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Japanese",
]
EN_TARGETS = [
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Italian",
    "Japanese",
    "Mandarin Chinese",
]
TRANSCRIBE_PROMPT = "can you transcribe the speech into a written format?"
SUPPORTED_FORMATS = [
    "wav",
    "flac",
    "m4a",
    "mp3",
    "ogg",
    "aac",
    "mp4",
    "mov",
    "webm",
    "mkv",
]
VIDEO_FORMATS = {"mp4", "mov", "webm", "mkv"}
SAMPLE_RATE = 16000
# Silero's fixed framing at 16 kHz: a 512-sample (32 ms) decision chunk, each
# fed to the network together with the 64 samples preceding it.
VAD_CHUNK_SAMPLES = 512
VAD_CONTEXT_SAMPLES = 64
# Chunks per batched encoder pass. Caps peak memory rather than tuning speed:
# the STFT intermediate is (batch, 4, 258) float32, so 2048 chunks (~65s of
# audio) holds ~17 MB in flight. Measured +43 MB peak over the streaming path
# on 15 min of audio, and larger batches stopped helping once the GPU saturated.
VAD_ENCODER_BATCH_CHUNKS = 2048
# Ceiling for the VAD-off path, where the whole clip is one inference. Set by
# measured peak MLX memory (fresh process per run, 8-bit 2B weights = 3.3 GB
# resident): 30s -> 8.4 GB, 60s -> 9.5 GB, 120s -> 14.5 GB, 300s -> 17.1 GB.
# The old 300s was calibrated against the 1B model; on this one it puts a 16 GB
# Mac into swap. 120s caps the worst case around 14.5 GB. Correctness is not the
# binding constraint here — ~350s is where the 4096-position context runs out —
# memory is. Drop to 60s if 16 GB machines still struggle.
MAX_VAD_OFF_DURATION_S = 120
# Longest segment fed to a single inference when VAD is on. Two bounds apply,
# and the tighter one wins by a wide margin:
#   context — the model emits ~10.2 audio tokens/s against 4096 positions, so
#             with max_tokens=512 reserved the hard ceiling is ~350s;
#   quality — translation degrades far sooner. Measured on 40s of continuous
#             speech, counting segments whose German output is actually German:
#             10s -> 2/4, 8s -> 6/6, 6s -> 8/8, 5s -> 12/12, 4s -> 12/16.
#             Above ~20s every target language collapses to source-language
#             passthrough; below ~5s the splits land mid-utterance and cost
#             accuracy again.
# 8s is the widest cap that still translated everything, so it fragments the
# audio (and multiplies inference count) as little as possible.
MAX_SEGMENT_DURATION_S = 8.0
# Floor for splitting an over-long span. From the same measurements: 5s -> 12/12
# but 4s -> 12/16, because short parts land mid-utterance. A span only slightly
# over the cap must not be halved into sub-floor slivers — the overshoot costs
# less accuracy than the split would.
MIN_SEGMENT_DURATION_S = 5.0
# Where a single inference stops translating and echoes the source verbatim.
# Only reachable with VAD off, since the cap above keeps every VAD segment well
# under it.
TRANSLATION_PASSTHROUGH_S = 20.0
TOXICITY_THRESHOLD = 0.5
TOXICITY_SCORE_PRECISION = 4


class PipelineResult(TypedDict):
    transcript: str
    is_toxic: NotRequired[bool]
    toxicity_score: NotRequired[float]


def is_video(filename: str) -> bool:
    return Path(filename).suffix.lower().lstrip(".") in VIDEO_FORMATS


def build_tasks(source: str) -> dict[str, str]:
    tasks: dict[str, str] = {"Transcribe": TRANSCRIBE_PROMPT}
    targets = EN_TARGETS if source == "English" else ["English"]
    for target in targets:
        tasks[target] = f"translate the speech to {target}"
    return tasks


def apply_keywords(prompt: str, keywords: list[str]) -> str:
    if not keywords:
        return prompt
    return f"{prompt} Keywords: {', '.join(keywords)}"


def produces_english(source: str, task: str) -> bool:
    """Can this (source, task) pair put English text on a result card?

    Transcription is English exactly when the audio is, and a translation into
    English always is. Beyond that: when the audio is English, *any* target can
    come back as untranslated English passthrough (see the segment-length note
    in CLAUDE.md), so those count too — the guardian scoring a genuinely French
    card is a far cheaper mistake than skipping it on English text that turned
    out to be toxic.

    Non-English sources need no such widening: passthrough there emits the
    source language, which is not English by definition.
    """
    if task == "Transcribe":
        return source == "English"
    return task == "English" or source == "English"


def compute_safety_tasks(
    selected_tasks: list[str], source: str, use_toxicity_check: bool
) -> set[str]:
    if not use_toxicity_check:
        return set()
    return {name for name in selected_tasks if produces_english(source, name)}


def result_title(source: str, task: str) -> str:
    if task == "Transcribe":
        return f"Transcribe ({source})"
    return task


def result_slug(source: str, task: str) -> str:
    if task == "Transcribe":
        return f"transcribe_{source.lower().replace(' ', '_')}"
    return task.lower().replace(" ", "_")


def format_timestamp(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"


_MLX_VAD_BRANCH_ATTRS = (
    "stft_conv",
    "conv1",
    "conv2",
    "conv3",
    "conv4",
    "lstm",
    "final_conv",
)
_MLX_VAD_BRANCH_CONFIG_ATTRS = ("pad", "cutoff", "chunk_size", "context_size")
_MLX_VAD_CONFIG_ATTRS = (
    "threshold",
    "min_speech_duration_ms",
    "min_silence_duration_ms",
    "speech_pad_ms",
)


def _supports_mlx_vad(model: Any) -> bool:
    """Probe the mlx_audio surface the batched VAD path reaches into.

    Same exposure, and same mitigation, as _supports_encoder_hoist: the fast
    path drives the branch submodules directly and borrows the private
    _probs_to_timestamps, so a rename upstream must degrade to the PyTorch
    reference rather than crash the run.

    Unlike that probe this one also checks two config *values*, not just names.
    The pin freezes the repo's config.json but not mlx_audio's BranchConfig
    defaults, which fill in any field the config omits — so a dependency
    upgrade can still move the framing out from under us. Neither this path nor
    mlx_audio's own _probs_to_timestamps (which hardcodes a 512-sample stride)
    could serve different framing correctly, and getting it wrong yields
    plausible garbage rather than an error, so refuse instead.
    """
    branch = getattr(model, "vad_16k", None)
    branch_config = getattr(branch, "config", None)
    config = getattr(model, "config", None)
    if branch is None or branch_config is None or config is None:
        return False
    if not (
        hasattr(model, "_probs_to_timestamps")
        and hasattr(model, "dtype")
        and all(hasattr(branch, name) for name in _MLX_VAD_BRANCH_ATTRS)
        and all(hasattr(branch_config, name) for name in _MLX_VAD_BRANCH_CONFIG_ATTRS)
        and all(hasattr(config, name) for name in _MLX_VAD_CONFIG_ATTRS)
    ):
        return False
    return (
        branch_config.chunk_size == VAD_CHUNK_SAMPLES
        and branch_config.context_size == VAD_CONTEXT_SAMPLES
    )


def _mlx_vad_windows(audio: mx.array) -> mx.array:
    """One (context + chunk) window per decision chunk, as a strided view.

    Frames the audio exactly as mlx_audio's streaming loop does — right-pad up
    to a chunk multiple, prepend the zero context — but the windows overlap by
    64 samples, so materialising them would copy ~12% more audio than it needs
    to. as_strided keeps them a view over the one padded buffer.
    """
    n_samples = audio.shape[-1]
    tail = (VAD_CHUNK_SAMPLES - n_samples % VAD_CHUNK_SAMPLES) % VAD_CHUNK_SAMPLES
    # Both edges in one pass: padding then concatenating the context would
    # allocate two full copies of the waveform, which on a 500 MB upload is a
    # lot of transient memory for a function that exists to avoid a copy.
    audio = mx.pad(audio, [(VAD_CONTEXT_SAMPLES, tail)])
    return mx.as_strided(
        audio,
        shape=(
            (n_samples + tail) // VAD_CHUNK_SAMPLES,
            VAD_CONTEXT_SAMPLES + VAD_CHUNK_SAMPLES,
        ),
        strides=(VAD_CHUNK_SAMPLES, 1),
    )


def _mlx_vad_probabilities(model: Any, audio: mx.array) -> mx.array:
    """Per-chunk speech probabilities, encoder batched across chunks.

    The STFT + conv encoder is stateless per chunk — it only ever sees that
    chunk's 576-sample window — so it can run on a whole batch at once. Only
    the LSTM is sequential, and nn.LSTM unrolls a full (1, T, 128) sequence
    inside one graph, so carrying (hidden, cell) across batches reproduces
    stepping chunk by chunk exactly. Net effect: one model call per 32 ms of
    audio becomes one per ~65s.

    Measured 2.9-3.5x faster than the PyTorch path across 10s-15min of audio,
    and identical where it counts: max |delta p| of 2.5e-06 over 9375 chunks,
    zero decisions flipped at the threshold, byte-identical spans out.

    mlx_audio's own get_speech_timestamps is *slower* than PyTorch here (0.66x),
    which is the whole reason this exists: at 309K parameters the per-call
    dispatch cost dominates the arithmetic, so call count is the only lever.
    """
    branch = model.vad_16k
    windows = _mlx_vad_windows(audio)
    pad, cutoff = branch.config.pad, branch.config.cutoff
    # PyTorch's ReflectionPad1d(right=pad), inlined: out[L + i] = in[L - 2 - i].
    reflect = mx.arange(windows.shape[-1] - 2, windows.shape[-1] - pad - 2, -1)

    outputs: list[mx.array] = []
    hidden = cell = None
    for start in range(0, windows.shape[0], VAD_ENCODER_BATCH_CHUNKS):
        batch = windows[start : start + VAD_ENCODER_BATCH_CHUNKS]
        x = mx.concatenate([batch, mx.take(batch, reflect, axis=-1)], axis=-1)
        x = branch.stft_conv(x[..., None])
        real, imag = x[..., :cutoff], x[..., cutoff:]
        x = mx.sqrt(real * real + imag * imag)
        x = nn.relu(branch.conv1(x))
        x = nn.relu(branch.conv2(x))
        x = nn.relu(branch.conv3(x))
        x = nn.relu(branch.conv4(x))
        if x.shape[1] != 1:
            # The batch axis is reused as the LSTM's sequence axis, which holds
            # only while the conv stack collapses each window to a single frame.
            # Different STFT framing leaves more (hop_length=64 gives 2), and
            # taking frame 0 would silently drop the rest while the reference
            # averages them. Refuse, and let the caller fall back to PyTorch.
            raise ValueError(
                f"VAD encoder produced {x.shape[1]} frames per chunk, expected 1"
            )
        hidden_seq, cell_seq = branch.lstm(x[:, 0, :][None], hidden=hidden, cell=cell)
        outputs.append(
            mx.squeeze(mx.sigmoid(branch.final_conv(nn.relu(hidden_seq))), axis=-1)[0]
        )
        hidden, cell = hidden_seq[:, -1, :], cell_seq[:, -1, :]
        mx.async_eval(outputs[-1], hidden, cell)
    return mx.concatenate(outputs) if outputs else mx.zeros((0,))


def _spans_in_seconds(
    speech_timestamps: list[dict[str, int]], sample_rate: int
) -> list[tuple[float, float]]:
    """Shared sample-offset -> seconds tail for both backends.

    The two paths are interchangeable only if they convert identically, so this
    is one function rather than two copies that could drift apart.
    """
    return [
        (ts["start"] / sample_rate, ts["end"] / sample_rate) for ts in speech_timestamps
    ]


def _mlx_vad_spans(
    wav: torch.Tensor, model: Any, sample_rate: int
) -> list[tuple[float, float]]:
    # Model.__call__ casts every input to model.dtype; this path bypasses it by
    # driving the branch directly, so do the cast here — MLX would otherwise
    # silently promote float32 audio against a float16 checkpoint and diverge
    # from the reference it claims to reproduce.
    audio = mx.array(wav.detach().reshape(-1).to(torch.float32).numpy()).astype(
        model.dtype
    )
    probabilities = _mlx_vad_probabilities(model, audio)
    mx.eval(probabilities)
    speech_timestamps = model._probs_to_timestamps(
        probabilities,
        audio_len=audio.shape[-1],
        sample_rate=sample_rate,
        threshold=model.config.threshold,
        min_speech_duration_ms=model.config.min_speech_duration_ms,
        min_silence_duration_ms=model.config.min_silence_duration_ms,
        speech_pad_ms=model.config.speech_pad_ms,
        return_seconds=False,
    )
    return _spans_in_seconds(speech_timestamps, sample_rate)


def _torch_vad_spans(
    wav: torch.Tensor, model: torch.nn.Module, sample_rate: int
) -> list[tuple[float, float]]:
    speech_timestamps = get_speech_timestamps(
        wav.squeeze(), model, sampling_rate=sample_rate
    )
    return _spans_in_seconds(speech_timestamps, sample_rate)


@st.cache_resource(show_spinner=False)
def _torch_vad_model() -> torch.nn.Module:
    """Cached PyTorch VAD, shared by every fallback site.

    load_silero_vad() is not memoised — it re-runs torch.jit.load on each call —
    so calling it inline would re-deserialise the model on every pipeline run of
    a session that permanently falls back, and leave a second copy resident
    alongside whatever load_vad_model already cached.
    """
    return load_silero_vad()


def silero_vad(
    wav: torch.Tensor, model: Any, sample_rate: int = SAMPLE_RATE
) -> list[tuple[float, float]]:
    if isinstance(model, torch.nn.Module):
        return _torch_vad_spans(wav, model, sample_rate)
    # The v6 conversion ships no 8 kHz weights, but mlx_audio builds that branch
    # regardless and loads non-strictly, so calling it at 8 kHz runs randomly
    # initialised layers and returns confident nonsense instead of failing.
    # Anything but 16 kHz goes to the PyTorch model, which carries both branches.
    if sample_rate != SAMPLE_RATE:
        return _torch_vad_spans(wav, _torch_vad_model(), sample_rate)
    try:
        return _mlx_vad_spans(wav, model, sample_rate)
    except Exception as e:  # noqa: BLE001 - any drift falls back to PyTorch
        # hasattr only catches renames. A changed signature or return shape gets
        # here instead, and would otherwise kill the run outright; warn so a
        # silent 3x slowdown does not become an undiagnosable mystery.
        warnings.warn(
            f"MLX VAD failed ({type(e).__name__}: {e}); falling back to the "
            "slower PyTorch model. mlx_audio's VAD internals have probably "
            "changed.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _torch_vad_spans(wav, _torch_vad_model(), sample_rate)


def _split_long_segment(
    start: float, end: float, max_duration: float
) -> list[dict[str, float]]:
    """Split an over-long span into equal parts, each at most max_duration.

    Equal parts rather than max_duration-sized chunks plus a remainder: a
    30.1s span becomes 3 x ~10s, not 2 x 15s + one 0.1s sliver that would
    transcribe to nothing.
    """
    duration = end - start
    if duration <= max_duration:
        return [{"start": start, "end": end}]
    parts = math.ceil(duration / max_duration)
    # Back off the part count rather than emit parts below the accuracy floor:
    # buffering pushes a natural 8.0s span to 8.6s, which would otherwise halve
    # into 2 x 4.3s. The floor cannot exceed the cap itself, or a caller passing
    # a small max_duration would stop getting any split at all.
    floor = min(MIN_SEGMENT_DURATION_S, max_duration)
    while parts > 1 and duration / parts < floor:
        parts -= 1
    step = duration / parts
    return [
        {"start": start + i * step, "end": start + (i + 1) * step} for i in range(parts)
    ]


def get_speech_segments(
    wav: torch.Tensor,
    model: Any,
    sample_rate: int = SAMPLE_RATE,
    max_segment_duration: float = MAX_SEGMENT_DURATION_S,
) -> list[dict[str, float]]:
    duration = wav.shape[-1] / sample_rate
    vad_segments = silero_vad(wav, model, sample_rate)
    if not vad_segments:
        return _split_long_segment(0.0, duration, max_segment_duration)
    start_buffer = 0.3
    end_buffer = 0.3
    min_gap = 0.5
    segments: list[dict[str, float]] = []
    for start, end in vad_segments:
        buffered_start = max(0.0, start - start_buffer)
        buffered_end = min(duration, end + end_buffer)
        # Merge across short gaps, but never past the cap — continuous speech
        # would otherwise collapse into one unbounded segment.
        if (
            segments
            and buffered_start - segments[-1]["end"] < min_gap
            and max(segments[-1]["end"], buffered_end) - segments[-1]["start"]
            <= max_segment_duration
        ):
            segments[-1]["end"] = max(segments[-1]["end"], buffered_end)
        else:
            # The buffers can pull this span's start behind the previous
            # segment's end. Merging used to absorb that overlap unconditionally
            # (a negative gap always satisfies the min_gap test), so the cap is
            # what first made this reachable: clamp instead, or the same audio
            # gets transcribed twice and timestamps run backwards.
            if segments:
                buffered_start = max(buffered_start, segments[-1]["end"])
                if buffered_start >= buffered_end:
                    # Wholly covered by the previous segment already.
                    continue
            segments.append({"start": buffered_start, "end": buffered_end})
    # A single VAD span can exceed the cap on its own (unbroken speech, or the
    # no-gap fallback above), so split whatever is still over.
    return [
        part
        for seg in segments
        for part in _split_long_segment(seg["start"], seg["end"], max_segment_duration)
    ]


def _verdict(probability: float) -> tuple[bool, float]:
    """Shared toxic/score verdict so the two callers cannot drift apart."""
    return (
        probability > TOXICITY_THRESHOLD,
        round(probability, TOXICITY_SCORE_PRECISION),
    )


@st.cache_resource(show_spinner=False)
def load_model(model_id: str, revision: str | None = None) -> Any:
    return _load_stt_model(model_id, revision=revision)


def load_and_preprocess_audio(audio_file: UploadedFile) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(io.BytesIO(audio_file.getvalue()))
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {e}") from e

    # Caught here rather than downstream: an empty waveform survives VAD (which
    # falls back to a zero-length full-audio segment) and only dies inside the
    # mel filterbank, as an opaque reshape error on zero rows.
    if wav.numel() == 0:
        raise RuntimeError("No audio detected: the file decoded to zero samples.")

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav


def audio_duration_seconds(audio_file: UploadedFile) -> float | None:
    try:
        decoder = AudioDecoder(audio_file.getvalue())
        duration = decoder.metadata.duration_seconds
    except (RuntimeError, ValueError, OSError):
        return None
    if duration is None or duration <= 0:
        return None
    return float(duration)


@st.cache_resource(show_spinner=False)
def load_guardian_model(model_id: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return model, tokenizer


@st.cache_resource(show_spinner=False)
def load_vad_model() -> Any:
    """Load the MLX Silero VAD v6 model, degrading to the PyTorch build.

    Two ways the fast path can be unavailable, and both land on the same
    fallback: the repo may not resolve at all (offline, or renamed), or it may
    load into a class whose internals have moved. Warn in either case — the
    PyTorch path is correct but ~3x slower, and a silent downgrade would be
    invisible until someone benchmarked it.
    """
    try:
        model = _load_mlx_vad_model(MLX_VAD_REPO, revision=MLX_VAD_REVISION)
    except Exception as e:  # noqa: BLE001 - any load failure falls back
        warnings.warn(
            f"Could not load {MLX_VAD_REPO} ({e}); using the PyTorch VAD.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _torch_vad_model()
    if not _supports_mlx_vad(model):
        warnings.warn(
            "mlx_audio's VAD internals have moved; using the PyTorch VAD.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _torch_vad_model()
    return model


def check_safety(
    text: str,
    model: Any,
    tokenizer: Any,
) -> tuple[bool, float]:
    # Guardian (RoBERTa) caps at 512 tokens. For inputs longer than that,
    # chunk into 510-token windows (reserving 2 slots for CLS/SEP) and take
    # the max — otherwise truncation would silently drop late content.
    max_content_tokens = 510
    encoding = tokenizer(
        text, return_tensors="pt", truncation=False, add_special_tokens=False
    )
    input_ids = encoding["input_ids"][0]
    if len(input_ids) <= max_content_tokens:
        chunks = [text]
    else:
        chunks = [
            tokenizer.decode(
                input_ids[i : i + max_content_tokens], skip_special_tokens=True
            )
            for i in range(0, len(input_ids), max_content_tokens)
        ]

    max_probability = 0.0
    for chunk in chunks:
        inputs = tokenizer([chunk], padding=True, truncation=True, return_tensors="pt")
        logits = model(**inputs).logits
        probability = torch.softmax(logits, dim=1)[0, 1].item()
        max_probability = max(max_probability, probability)
    return _verdict(max_probability)


def transcribe_audio(
    wav: torch.Tensor,
    prompt: str,
    model: Any,
) -> str:
    audio_np = wav.squeeze().numpy()
    output = model.generate(audio=audio_np, prompt=prompt, max_tokens=512)
    return output.text


# model.generate() re-runs mel extraction and the 16-layer conformer encoder on
# every call, but that work depends only on the audio — not the prompt. Running
# N tasks over one segment therefore pays for N encodes where 1 would do, and
# the encoder is ~33% of a call. These two helpers split generate() so the
# encode can be hoisted out of the task loop.
#
# They reach into mlx_audio internals, so `_supports_encoder_hoist` gates them
# and run_pipeline falls back to the public `transcribe_audio` path when a
# future version renames something. The fallback is correct, just slower.
#
# The name probe only catches renames. A changed signature or return shape
# passes it and raises at call time, so run_pipeline also wraps the first
# `_encode_segment` and disables the hoist on any exception — with a
# RuntimeWarning, because an unexplained 28% slowdown is near-undiagnosable.
_HOIST_ATTRS = (
    "_extract_features",
    "get_audio_features",
    "_build_prompt",
    "_build_inputs_embeds",
    "_tokenizer",
)


def _supports_encoder_hoist(model: Any) -> bool:
    return all(hasattr(model, attr) for attr in _HOIST_ATTRS)


def _encode_segment(wav: torch.Tensor, model: Any) -> tuple[Any, int]:
    input_features, num_audio_tokens = model._extract_features(wav.squeeze().numpy())
    audio_features = model.get_audio_features(input_features)
    mx.eval(audio_features)
    return audio_features, num_audio_tokens


def _generate_from_features(
    audio_features: Any,
    num_audio_tokens: int,
    prompt: str,
    model: Any,
    max_tokens: int = 512,
) -> str:
    """Decode one prompt against pre-encoded audio.

    Sampler settings mirror mlx_audio's own generate() defaults (greedy), so
    output is identical to the non-hoisted path.
    """
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    prompt_ids = model._build_prompt(num_audio_tokens, prompt)
    inputs_embeds = model._build_inputs_embeds(prompt_ids, audio_features)
    mx.eval(inputs_embeds)

    eos_token_id = model._tokenizer.eos_token_id
    tokens: list[Any] = []
    for token, _ in generate_step(
        prompt=prompt_ids,
        input_embeddings=inputs_embeds.squeeze(0),
        model=model,
        max_tokens=max_tokens,
        sampler=make_sampler(0.0, top_p=1.0, min_p=0.0, top_k=0),
        prefill_step_size=2048,
    ):
        if token == eos_token_id:
            break
        tokens.append(token)
    return model._tokenizer.decode(tokens, skip_special_tokens=True)


def _aggregate_segment_safety(
    texts: list[str], model: Any, tokenizer: Any
) -> tuple[bool, float]:
    max_probability = 0.0
    for text in texts:
        if not text.strip():
            continue
        _, probability = check_safety(text, model, tokenizer)
        max_probability = max(max_probability, probability)
    return _verdict(max_probability)


@torch.inference_mode()
def run_pipeline(
    wav: torch.Tensor,
    tasks: dict[str, str],
    safety_tasks: set[str],
    model: Any,
    vad_model: Any = None,
    guardian_model: Any | None = None,
    guardian_tokenizer: Any | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    use_segmentation: bool = True,
    keywords: list[str] | None = None,
) -> dict[str, PipelineResult]:
    if use_segmentation:
        assert vad_model is not None, "vad_model required when use_segmentation=True"
        segments = get_speech_segments(wav, vad_model)
    else:
        duration = wav.shape[-1] / SAMPLE_RATE
        segments = [{"start": 0.0, "end": duration}]
    if keywords is None:
        keywords = []

    # Keywords bias transcription only — that is what the UI promises, and
    # appending them to a translation prompt measurably degrades it (the model
    # drops back to untranslated, truncated source text).
    prompts = {
        task: apply_keywords(prompt, keywords) if task == "Transcribe" else prompt
        for task, prompt in tasks.items()
    }

    # Segments outer, tasks inner: the audio encoding depends only on the
    # segment, so this encodes each segment once and reuses it across every
    # task instead of re-encoding per (task, segment) pair.
    hoist = _supports_encoder_hoist(model)
    raw_texts: dict[str, list[str]] = {task: [] for task in tasks}
    lines: dict[str, list[str]] = {task: [] for task in tasks}
    total_steps = max(len(segments) * len(tasks), 1)
    step = 0

    for seg in segments:
        start_sample = int(seg["start"] * SAMPLE_RATE)
        end_sample = int(seg["end"] * SAMPLE_RATE)
        chunk = wav[:, start_sample:end_sample]
        if hoist:
            try:
                encoded = _encode_segment(chunk, model)
            except Exception as e:  # noqa: BLE001 - any drift disables the hoist
                # _supports_encoder_hoist only proves the names exist. A changed
                # signature or return shape gets this far, and without the catch
                # it kills the whole run. Warn rather than degrade quietly: the
                # public path is correct, just ~28% slower, and a silent
                # slowdown is close to undiagnosable.
                warnings.warn(
                    f"Encoder hoist failed ({type(e).__name__}: {e}); falling "
                    "back to the slower per-task encode. mlx_audio internals "
                    "have probably changed.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                hoist = False
                encoded = None
        else:
            encoded = None
        ts_start = format_timestamp(seg["start"])
        ts_end = format_timestamp(seg["end"])
        for task, actual_prompt in prompts.items():
            if on_progress:
                on_progress(step, total_steps, task)
            step += 1
            if encoded is not None:
                text = _generate_from_features(*encoded, actual_prompt, model)
            else:
                text = transcribe_audio(chunk, actual_prompt, model)
            raw_texts[task].append(text)
            lines[task].append(f"[{ts_start} - {ts_end}] {text}")

    results: dict[str, PipelineResult] = {}
    if on_progress:
        # The guardian pass below is real work. Without this the bar would sit
        # frozen at (total-1)/total for its whole duration, still labelled with
        # the task that already finished.
        checking = bool(safety_tasks) and guardian_model is not None
        on_progress(total_steps, total_steps, "safety check" if checking else "results")
    for task in tasks:
        result: PipelineResult = {"transcript": "\n".join(lines[task])}
        if (
            task in safety_tasks
            and guardian_model is not None
            and guardian_tokenizer is not None
        ):
            # Per-segment so long transcripts don't get silently truncated by
            # the guardian's 512-token cap; report the worst segment's score.
            is_toxic, score = _aggregate_segment_safety(
                raw_texts[task], guardian_model, guardian_tokenizer
            )
            result["is_toxic"] = is_toxic
            result["toxicity_score"] = score
        results[task] = result
    return results


def _row_sizes(n: int) -> list[int]:
    """Split n result cards into rows of at most 3, as evenly as possible.

    e.g. 4 → [2, 2] instead of [3, 1]; 7 → [3, 2, 2] instead of [3, 3, 1].
    """
    if n <= 0:
        return []
    rows = -(-n // 3)
    base, extra = divmod(n, rows)
    return [base + 1] * extra + [base] * (rows - extra)


def _labeled_toggle(label: str, help: str, key: str, value: bool = True) -> bool:
    label_col, toggle_col = st.columns([15, 1], vertical_alignment="center")
    with label_col:
        st.markdown(f"**{label}**", help=help)
    with toggle_col:
        return st.toggle(label, value=value, label_visibility="collapsed", key=key)


def _render_result_card(
    source: str,
    task: str,
    result: PipelineResult,
    stem: str,
) -> None:
    transcript = result["transcript"]
    title = result_title(source, task)
    is_transcription = task == "Transcribe"
    slug = result_slug(source, task)
    with st.container(border=True, height="stretch"):
        st.subheader(title)
        st.text(transcript)
        if "is_toxic" in result:
            score = f"score: {result['toxicity_score']:.1%}"
            if result["is_toxic"]:
                st.warning(
                    f"Toxic content detected ({score})", icon=":material/warning:"
                )
            else:
                st.success(f"Content is safe ({score})", icon=":material/check_circle:")
        download_help = (
            "Download transcription" if is_transcription else "Download translation"
        )
        st.download_button(
            "",
            transcript,
            f"{stem}_{slug}.txt",
            "text/plain",
            key=f"dl_txt_{source}_{task}",
            icon=":material/download:",
            help=download_help,
        )


def main() -> None:
    st.set_page_config(
        page_title="Granite Speech Studio",
        page_icon=":material/graphic_eq:",
        layout="centered",
    )

    st.title("Granite Speech Studio", text_alignment="center")
    st.markdown(
        "Transcribe and translate audio and video files with the "
        "[IBM Granite Speech 4.1 2B model]"
        "(https://huggingface.co/ibm-granite/granite-speech-4.1-2b).",
        text_alignment="center",
    )

    upload_tab, record_tab = st.tabs(["Upload", "Record"])
    with upload_tab:
        uploaded = st.file_uploader(
            "Upload audio file",
            type=SUPPORTED_FORMATS,
            help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}",
            label_visibility="collapsed",
        )
    with record_tab:
        recorded = st.audio_input("Record audio", label_visibility="collapsed")

    audio_file = uploaded or recorded

    if audio_file:
        if is_video(audio_file.name):
            st.video(audio_file)
        else:
            st.audio(audio_file)
        st.caption(audio_file.name if uploaded else "Recorded audio")

    source_value = st.segmented_control(
        "Source language",
        options=SOURCE_LANGUAGES,
        default="English",
        label_visibility="collapsed",
    )
    source: str = source_value if isinstance(source_value, str) else "English"

    available_tasks = build_tasks(source)
    selected_value = st.pills(
        "Tasks",
        options=list(available_tasks.keys()),
        selection_mode="multi",
        default=["Transcribe"],
        label_visibility="collapsed",
        key=f"tasks_{source}",
    )
    selected_tasks: list[str] = [t for t in selected_value if isinstance(t, str)]

    use_segmentation = _labeled_toggle(
        "VAD segmentation",
        help=(
            "Splits audio into speech segments with timestamps using "
            "Silero VAD. Disable for short utterances or to process "
            "the whole audio in one pass."
        ),
        key="use_segmentation",
    )

    vad_off_too_long = False
    if audio_file is not None and not use_segmentation:
        # Single-slot cache: getvalue() copies the full byte buffer each rerun,
        # so memoize the duration and recompute only when the file changes. One
        # slot can't grow, so no eviction is needed.
        cache_id = (audio_file.name, audio_file.size)
        cached = st.session_state.get("_duration")
        if cached is None or cached[0] != cache_id:
            cached = (cache_id, audio_duration_seconds(audio_file))
            st.session_state["_duration"] = cached
        duration = cached[1]
        if duration is not None and duration > MAX_VAD_OFF_DURATION_S:
            vad_off_too_long = True
            st.warning(
                f"Enable VAD segmentation: audio is longer than "
                f"{MAX_VAD_OFF_DURATION_S // 60} minutes. A single inference "
                "that long needs well over 14 GB of memory, and translation "
                "falls back to untranslated source text on long audio.",
                icon=":material/warning:",
            )
        elif any(task != "Transcribe" for task in selected_tasks) and (
            duration is None or duration > TRANSLATION_PASSTHROUGH_S
        ):
            # Short clips still translate fine with VAD off, so warn rather
            # than disable — but past the passthrough point the result card
            # looks finished while holding untranslated source text.
            st.warning(
                "Enable VAD segmentation to translate this clip: with it off "
                "the whole clip is one inference, and past "
                f"~{TRANSLATION_PASSTHROUGH_S:.0f}s the model returns "
                "untranslated source text instead of reporting an error.",
                icon=":material/warning:",
            )

    st.markdown(
        "**Keywords**",
        help=(
            "Up to 15 keywords to be boosted during transcription. "
            "Boosted terms are more likely to appear in the output."
        ),
    )
    keywords = st.multiselect(
        "Keywords",
        options=[],
        accept_new_options=True,
        max_selections=15,
        placeholder="Add keywords...",
        label_visibility="collapsed",
        key="keywords",
    )

    use_toxicity_check = _labeled_toggle(
        "Toxicity check",
        help=(
            "Checks output that may be English for toxic content via Granite "
            "Guardian. English audio can come back untranslated, so every task "
            "is checked when the source is English; non-English sources are "
            "checked only for translations into English."
        ),
        key="use_toxicity_check",
    )

    input_key = (
        (
            audio_file.name,
            audio_file.size,
            source,
            tuple(selected_tasks),
            use_segmentation,
            tuple(sorted(keywords)),
            use_toxicity_check,
        )
        if audio_file
        else None
    )
    if input_key != st.session_state.get("_last_input_key"):
        for key in ("results", "result_stem", "result_source"):
            st.session_state.pop(key, None)
        st.session_state["_last_input_key"] = input_key

    can_run = (
        audio_file is not None and len(selected_tasks) > 0 and not vad_off_too_long
    )

    with st.container(horizontal_alignment="right"):
        run_clicked = st.button(
            "Transcribe",
            type="primary",
            disabled=not can_run,
        )

    if run_clicked and can_run:
        assert audio_file is not None
        progress = st.progress(0, text="Starting pipeline...")
        try:
            with st.spinner("Loading speech model..."):
                model = load_model(MODEL_ID, MODEL_REVISION)
            wav = load_and_preprocess_audio(audio_file)

            if use_segmentation:
                with st.spinner("Loading VAD model..."):
                    vad_model = load_vad_model()
            else:
                vad_model = None

            def update_progress(i: int, total: int, task: str) -> None:
                progress.progress(i / total, text=f"Processing: {task}...")

            tasks_to_run = {name: available_tasks[name] for name in selected_tasks}
            safety_tasks = compute_safety_tasks(
                selected_tasks, source, use_toxicity_check
            )

            if safety_tasks:
                with st.spinner("Loading safety model..."):
                    guardian_model, guardian_tokenizer = load_guardian_model(
                        GUARDIAN_MODEL_ID
                    )
            else:
                guardian_model, guardian_tokenizer = None, None

            pipeline_results = run_pipeline(
                wav,
                tasks_to_run,
                safety_tasks,
                model,
                vad_model,
                guardian_model,
                guardian_tokenizer,
                on_progress=update_progress,
                use_segmentation=use_segmentation,
                keywords=keywords,
            )
            st.session_state.results = pipeline_results
            if uploaded:
                stem = Path(audio_file.name).stem
            else:
                # Local wall-clock is what a user expects in a filename.
                stem = datetime.now().strftime(  # noqa: DTZ005
                    "recording_%Y%m%d_%H%M%S"
                )
            st.session_state.result_stem = stem
            st.session_state.result_source = source
        except RuntimeError as e:
            st.error(str(e))
            return
        except Exception as e:  # noqa: BLE001 - top-level UI error boundary
            st.exception(e)
            return
        finally:
            # Also on the error paths, or a half-filled bar labelled with the
            # task that failed stays on screen next to the error message.
            progress.empty()
        st.toast("Pipeline complete!")

    if "results" in st.session_state:
        results = st.session_state.results
        stem = st.session_state.result_stem
        source_used = st.session_state.result_source
        task_names = list(results.keys())

        idx = 0
        for row_size in _row_sizes(len(task_names)):
            cols = st.columns(row_size)
            for col, task_name in zip(
                cols, task_names[idx : idx + row_size], strict=True
            ):
                with col:
                    _render_result_card(
                        source_used, task_name, results[task_name], stem
                    )
            idx += row_size


if __name__ == "__main__":
    main()
