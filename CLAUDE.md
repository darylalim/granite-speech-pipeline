# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit application for transcription and translation using IBM Granite Speech on Apple Silicon with MLX. Multi-task pipeline processing, VAD-based audio segmentation, and English toxicity detection.

## Setup

```bash
uv sync
uv run streamlit run streamlit_app.py
```

Requires FFmpeg's shared libs at runtime (`brew install ffmpeg` on macOS) — `streamlit_app.py` imports `torchcodec`, which `dlopen`s FFmpeg at import time, so a fresh env without it crashes on startup.

## Commands

- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck**: `uv run ty check`
- **Test**: `uv run pytest`

When working with Python, invoke the relevant `/astral:<skill>` (`/astral:uv`, `/astral:ty`, `/astral:ruff`) for uv, ty, and ruff to ensure best practices are followed.

## Hooks & CI

- `.claude/` hooks: a **Stop hook** (`check-on-stop.sh`) auto-runs `ruff check` + `ty check` + `pytest -q` and blocks turn completion on failure when `.py` files changed — no need to hand-run them at end of turn. A **PreToolUse hook** blocks edits to `uv.lock` (use `uv add`/`remove`/`lock`/`sync`). A **PostToolUse hook** (`ruff-on-edit.sh`) auto-`ruff format`s edited `.py` files (format only, not `check --fix`).
- CI (`.github/workflows/ci.yml`): GitHub Actions on macOS, Python **3.12 and 3.13**, enforcing `uv sync --locked`, `ruff check`, `ruff format --check`, `ty check`, and `pytest`.

## Code Style

- snake_case for functions/variables, PascalCase for classes
- Type annotations on all parameters and returns
- `RuntimeError` for known transcription failures (no custom exception class)
- isort with combine-as-imports (configured in `pyproject.toml`)

## Dependencies

- `mlx-audio` — MLX-based speech model loading and inference (Apple Silicon)
- `transformers` — guardian model loading (toxicity detection)
- `torch` — tensor operations (VAD, guardian model)
- `torchaudio` — audio loading and resampling for the whole pipeline (16 kHz mono via `load_and_preprocess_audio`), including VAD preprocessing
- `torchcodec` — audio decoding backend for torchaudio; also used directly via `AudioDecoder` for header-only duration reads
- `silero-vad` — Voice Activity Detection for audio segmentation
- `streamlit` — web user interface
- `ruff` — linting/formatting (dev)
- `ty` — type checking (dev)
- `pytest` — testing (dev)

## Configuration

`pyproject.toml` — ruff isort (`combine-as-imports`) and ty (`python-version = "3.12"`).

`.streamlit/config.toml` — IBM Carbon-inspired `[theme]` with `[theme.light]` and `[theme.dark]` palettes (colors only, no remote web fonts). Mode-agnostic settings (radii, link/border options) live in `[theme]`; the per-mode color palettes enable the settings-menu light/dark toggle.

## Architecture

`streamlit_app.py` — single-file app.

### Functions

- `build_tasks` — returns ordered `{task_name: prompt}` dict for a given source language
- `apply_keywords` — appends `Keywords: kw1, kw2, ...` suffix to a prompt when the keywords list is non-empty. `run_pipeline` applies it to the `Transcribe` task **only**: appending keywords to a translation prompt makes the model return untranslated, truncated source text (measured on French), and the UI help text scopes the feature to transcription anyway
- `produces_english` — predicate: **can** `(source, task)` yield English output (drives safety check). True for English-source transcription, any X→English translation, and — because English audio can come back as untranslated passthrough — *every* target when the source is English. Non-English sources need no widening: passthrough there emits the source language
- `_supports_encoder_hoist` / `_encode_segment` / `_generate_from_features` — split `model.generate()` so the mel + 16-layer conformer encode (~33% of a call, and prompt-independent) happens once per segment instead of once per (segment, task). Reaches into `mlx_audio` internals (`_extract_features`, `get_audio_features`, `_build_prompt`, `_build_inputs_embeds`, `_tokenizer`), so `_supports_encoder_hoist` gates it and `run_pipeline` falls back to plain `transcribe_audio` if a future version renames any of them. Sampler settings mirror mlx_audio's greedy defaults, so output is byte-identical; measured 28% faster on 4 tasks × 5 segments
- `compute_safety_tasks` — filters `selected_tasks` to those that may produce English output; returns empty set when toxicity check is off
- `result_title` — display title for result cards (transcription shows source; translation shows target)
- `result_slug` — filename slug for downloads (transcription includes source; translation uses target)
- `is_video` — predicate by extension, drives `st.video` vs `st.audio` preview
- `format_timestamp` — formats seconds to `M:SS` or `H:MM:SS`
- `audio_duration_seconds` — returns clip duration via `torchcodec.decoders.AudioDecoder.metadata.duration_seconds` (header read only, no full decode); returns `None` if the format can't be parsed. Used to gate the Run button when VAD is off on long audio (threshold `MAX_VAD_OFF_DURATION_S = 120`).
- `_verdict` — shared `probability → (is_toxic, rounded_score)` tail for `check_safety` and `_aggregate_segment_safety`, so the threshold comparison and rounding can't drift apart between them
- `_aggregate_segment_safety` — calls `check_safety` per non-empty segment text and returns `(is_toxic, max_score)`; aggregation is max so any toxic segment flags the whole transcript
- `_row_sizes` — splits `n` result cards into rows of at most 3 as evenly as possible (4 → `[2, 2]`, 7 → `[3, 2, 2]`); used by the result grid in `main()` to avoid orphan cards
- `silero_vad` — runs Silero VAD on waveform, returns `(start, end)` tuples in seconds
- `_split_long_segment` — splits an over-long span into equal parts each within the cap (equal parts, not cap-sized chunks plus a remainder, so a 30.1s span becomes 3 × ~10s rather than 2 × 15s + a 0.1s sliver). Backs the part count off rather than emit parts under `MIN_SEGMENT_DURATION_S`: buffering turns a natural 8.0s span into 8.6s, which would otherwise halve into 2 × 4.3s — the measured-worse bucket (4s → 12/16) versus simply tolerating the small overshoot. The floor is clamped to `max_duration` so a caller passing a cap below it still gets a split
- `get_speech_segments` — post-processes VAD output with buffering, merging, and a `max_segment_duration` cap (default `MAX_SEGMENT_DURATION_S`). Merging across sub-`min_gap` gaps stops at the cap, and any span still over it — an unbroken VAD span, or the no-speech full-audio fallback — is split by `_split_long_segment`. Without the cap, continuous speech (podcast, lecture, dictation) merged into one unbounded segment. When the cap **refuses** a merge, the new segment's start is clamped to the previous segment's end (and skipped entirely if wholly covered): the 0.3s start/end buffers make consecutive spans overlap by 0.5s, and merging used to absorb that unconditionally because a negative gap always satisfies the `min_gap` test. The cap is what first made the overlap reachable — unclamped it re-transcribed ~0.5s at every boundary and emitted timestamps that ran backwards
- `load_vad_model` — cached Silero VAD model loader
- `load_model` — cached speech-model loader (MLX, via `mlx_audio`); takes an optional `revision` forwarded to `mlx_audio`, called with `MODEL_REVISION` so the personal-account conversion can't change under us
- `load_guardian_model` — cached guardian loader; returns `(model, tokenizer)`
- `load_and_preprocess_audio` — loads from `io.BytesIO` via `torchaudio`, downmixes to mono, resamples to `SAMPLE_RATE` (16 kHz); raises `RuntimeError` on failure
- `transcribe_audio` — single `model.generate` call (`max_tokens=512`) on a waveform slice
- `check_safety` — guardian toxicity check; returns `(is_toxic, rounded_score)`. Chunks inputs >510 tokens into windows and takes the max so long text isn't truncated by the guardian's 512-token cap
- `_labeled_toggle` — bold faux-label + right-aligned `st.toggle` in a `st.columns([15, 1])` pair; renders the VAD and Toxicity controls
- `run_pipeline` — takes `tasks: dict[str, str]` (task→prompt), `safety_tasks: set[str]`, and `use_segmentation: bool`; when segmentation is on, runs VAD then transcribes each segment; when off, treats the full audio as a single segment. **Segments outer, tasks inner** so each segment is encoded once and reused across tasks (see `_encode_segment`); one decode per task per segment, each using that task's own prompt. `on_progress` is called once per (segment, task) pair, so the bar advances smoothly rather than in `len(tasks)` jumps. Emits timestamped output and runs the safety check per segment for tasks in `safety_tasks` (empty segments skipped); the worst per-segment score is reported on the result card so long transcripts aren't silently truncated by the guardian's 512-token cap.

  A CoT-AST fast path (combined `"Can you transcribe the speech, and then translate it to {target}?"` prompt, output split on `[Transcription]`/`[Translation]` tags) was removed in favour of the above: neither `granite-4.0-1b-speech-8bit` nor `granite-speech-4.1-2b` emits those tags under any prompt phrasing, so the parse always failed and the fallback made the Transcribe+1-translation path cost *three* inferences per segment instead of two. Don't reintroduce it without first verifying the tags against real inference.

### Models

- [Granite Speech 4.1 2b 8bit (MLX)](https://huggingface.co/divydeep/granite-speech-4.1-2b-mlx-8bit) — transcription and translation (MLX, 8-bit quantized). A **community** conversion of [ibm-granite/granite-speech-4.1-2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) (pinned in its config to base revision `de575db`), not an mlx-community artifact — mlx-community publishes only the NAR port. Chosen over the older `mlx-community/granite-4.0-1b-speech-8bit`, which truncated every translation after the first clause and ignored keyword biasing.
- [Granite Guardian HAP 125m](https://huggingface.co/ibm-granite/granite-guardian-hap-125m) — English toxicity detection (runs on CPU)
- [Silero VAD](https://github.com/snakers4/silero-vad) — Voice Activity Detection for speech segmentation (runs on CPU)

### Languages

- Source languages: English, French, German, Spanish, Portuguese, Japanese (model-supported ASR set)
- Transcription: available for any source language
- Translation: English source → French, German, Spanish, Portuguese, Italian, Japanese, Mandarin Chinese; non-English source → English only (matches model's En↔X capability)

Known model behaviour: **translation quality is a function of segment length.** Past ~20s the model stops translating and echoes the source language verbatim; it does not error, so the failure is silent and the result card looks finished. `MAX_SEGMENT_DURATION_S = 8.0` exists to keep every segment under that cliff — measured on 40s of continuous speech, counting segments whose German output is actually German: 10s → 2/4, **8s → 6/6**, 6s → 8/8, 5s → 12/12, 4s → 12/16 (too-short splits land mid-utterance and lose accuracy again). 8s is the widest cap that still translated everything, minimising fragmentation and inference count. (The older 4.0-1b failed long input differently, silently truncating after the first clause.) With VAD **off** no cap applies — the whole clip is one inference, so translation on anything but a short clip will passthrough; that path is for transcription.

### UI Layout (top to bottom)

- **Page config** — `st.set_page_config` sets `page_title`, `page_icon` (`:material/graphic_eq:`), and `layout="centered"`
- **Title + description** — center-aligned (`text_alignment="center"`) `st.title` plus `st.markdown` linking to the IBM Granite Speech 4.1 2B model card
- **Audio input** — `st.tabs` with Upload (`st.file_uploader`) first, then Record (`st.audio_input`); labels hidden via `label_visibility="collapsed"`
- **Audio/video preview** — `st.video` for video containers, `st.audio` otherwise; selected via `is_video(filename)`. `st.caption` shows filename or "Recorded audio".
- **Source language** — `st.segmented_control` (single-select), `English` default; drives the task option list via `build_tasks(source)`
- **Task selection** — `st.pills` with `selection_mode="multi"`, label hidden, `Transcribe` preselected; widget keyed by source so options reset when source changes
- **VAD segmentation** — `st.columns([15, 1], vertical_alignment="center")`: `st.markdown("**VAD segmentation**", help=...)` (bold faux-label) on the left, `st.toggle` defaulting to `True` on the right. When off, VAD model load is skipped, `run_pipeline` treats the full audio as a single segment, and `MAX_SEGMENT_DURATION_S` does not apply (see the translation-length note under Languages). Part of `_last_input_key` so toggling invalidates cached results. When VAD is off and the audio is longer than `MAX_VAD_OFF_DURATION_S` (2 min), an `st.warning` is rendered and the Run button is disabled. The binding constraint is **memory, not context** — measured peak MLX memory for one inference: 30s -> 8.4 GB, 60s -> 9.5 GB, 120s -> 14.5 GB, 300s -> 17.1 GB (the old 300s limit was calibrated for the 1B model and swaps a 16 GB Mac). The 4096-position context does not run out until ~350s. Separately, when VAD is off and any translation task is selected on audio past `TRANSLATION_PASSTHROUGH_S` (20s, or of unknown duration), a second `st.warning` fires — the Run button stays enabled, since short clips still translate fine on that path
- **Keywords** — `st.markdown("**Keywords**", help=...)` bold faux-label followed by `st.multiselect` with `accept_new_options=True`, `max_selections=15`, `label_visibility="collapsed"`, and placeholder `"Add keywords..."`. When non-empty, `apply_keywords` appends `Keywords: kw1, kw2, ...` to the transcription prompt before inference (translation prompts are left alone — see `apply_keywords` above). Part of `_last_input_key` (as `tuple(sorted(keywords))`) so changes invalidate cached results.
- **Toxicity check** — `st.columns([15, 1], vertical_alignment="center")`: `st.markdown("**Toxicity check**", help=...)` (bold faux-label) on the left, `st.toggle` defaulting to `True` on the right. When off, `compute_safety_tasks` returns an empty set so guardian model load and per-task safety check are both skipped. Part of `_last_input_key` so toggling invalidates cached results. The help text must stay in sync with `produces_english` — it previously claimed "non-English output is skipped regardless of this setting", which the passthrough widening made false.
- **Run button** — `st.button("Transcribe", type="primary")` placed in a right-aligned `st.container(horizontal_alignment="right")` (content-width, no spacer column); disabled until audio is loaded and at least one task is selected
- **Results** — pipeline results, stem, and source captured at run time in `st.session_state`; displayed in a side-by-side column grid (up to 3 columns) via `_render_result_card` helper. Each card is a bordered `st.container(border=True, height="stretch")` so cards in the same row share equal height (no ragged bottom edge when transcript lengths differ)
- **Run feedback** — during a run, an `st.progress` bar (`"Starting pipeline..."` → `"Processing: {task}..."`) with per-model `st.spinner`s for the speech/VAD/safety loads, then `st.toast("Pipeline complete!")` on success. `progress.empty()` lives in a `finally` so the error paths clear it too — otherwise a half-filled bar labelled with the failing task stays on screen beside the error. (`on_progress` fires before each unit of work so the label names what is in flight, which means the bar tops out at `(total-1)/total`; harmless, since `empty()` removes it either way.)
- **Safety** — results show `st.success` (safe, `:material/check_circle:`) or `st.warning` (toxic, `:material/warning:`) banner with toxicity score whenever output *may* be English — English-source transcription, any X→English translation, and every target when the source is English (passthrough risk; see `produces_english`). Output is flagged toxic when the score exceeds `TOXICITY_THRESHOLD = 0.5`; the reported score is rounded to `TOXICITY_SCORE_PRECISION = 4` decimal places

### Audio Formats

Audio: wav, flac, m4a, mp3, ogg, aac. Video containers (audio track extracted via torchcodec): mp4, mov, webm, mkv. `SUPPORTED_FORMATS` is the combined accepted-extension list passed to `st.file_uploader`; `VIDEO_FORMATS` is its video subset, driving conditional preview (`st.video` vs `st.audio`). Upload size limit raised to 500 MB in `.streamlit/config.toml`.

### Performance

- Speech model runs via MLX on Apple Silicon GPU (8-bit quantized, ~3.3GB resident; peak scales with segment length — ~5.5GB at the 8s cap, 17GB for a 300s single pass)
- Audio encoding hoisted out of the task loop — one conformer pass per segment shared across tasks (28% faster on 4 tasks x 5 segments)
- Deferred model loading — speech and VAD models load on first pipeline run, not on page load
- `@st.cache_resource` to cache models
- `@torch.inference_mode()` on safety check and pipeline (for guardian model)
- `io.BytesIO` for in-memory audio loading (no temp files)
- Audio downmixed to mono and resampled to `SAMPLE_RATE = 16000` Hz on load
- `audio_duration_seconds` cached in a single `st.session_state["_duration"]` slot holding `((name, size), duration)` so the upload buffer isn't re-copied on every rerun (matters for 500 MB uploads); the slot is overwritten when the file changes, so it can't grow unbounded (no eviction needed)
- Guardian model runs on CPU with default dtype (125M params)
- Silero VAD model runs on CPU (~3MB)
- `max_tokens=512` per segment (prevents truncation on long speech)

### Error Handling

- `RuntimeError` caught explicitly for transcription failures
- Unexpected exceptions shown with `st.exception()`

### Downloads

- **Per-task Text** — plain transcript as `.txt`, icon-only download button (`:material/download:`) with context-aware tooltip ("Download transcription" or "Download translation")

### Tests

`tests/test_streamlit_app.py` — unit tests for constants, helpers (`build_tasks`, `apply_keywords`, `produces_english`, `compute_safety_tasks`, `result_title`, `result_slug`, `is_video`, `format_timestamp`, `_row_sizes`, `silero_vad`, `get_speech_segments` (buffering, merging, and the `max_segment_duration` cap — merge stops at the cap, long spans split into equal contiguous parts, no-speech fallback capped too, **segments never overlap and stay monotonic across cap values** when the cap refuses a merge), `_split_long_segment` (the `MIN_SEGMENT_DURATION_S` floor: a span just over the cap stays whole, long spans still split, and a cap below the floor still splits), `audio_duration_seconds`, `_aggregate_segment_safety`), model loaders (including the `MODEL_REVISION` pin), audio loading (wav + mp4 video), safety check, transcription, pipeline execution (multi-task, multi-segment, VAD on/off, keywords applied to transcription only, per-segment safety with max aggregation), and result card rendering. `TestGetSpeechSegments._run` takes an explicit `max_segment_duration` so cap tests don't depend on the shipped constant's value. Repetitive cases use `pytest.mark.parametrize`. Shared `pipeline_mocks` fixture supplies the four pipeline mocks via a `NamedTuple` so tests can unpack with `*pipeline_mocks`. `TestRunPipeline` patches `get_speech_segments` via an autouse `pytest.fixture` with a default single-segment fixture, and overrides it per-test for multi-segment cases. Decorator wrappers (`@st.cache_resource`, `@torch.inference_mode`) are bypassed via module-level `_load_model`, `_load_guardian_model`, `_load_vad_model`, `_run_pipeline` aliases pointing at `.__wrapped__`. Test fixtures in `tests/data/audio/` (`sample_10s.wav`, `sample_10s_video.mp4`).

`tests/test_app_ui.py` — end-to-end UI tests driving the real `main()` widget tree via `streamlit.testing.v1.AppTest` (complements `test_streamlit_app.py`, which mocks `st` entirely and never exercises the UI wiring). Covers default widget state, the bold VAD/Toxicity/Keywords faux-labels, Run-button gating on audio + task presence, source-change task reset, the VAD-off long-audio warning (text + `:material/warning:` icon), the single-slot `_duration` cache (overwritten, not accumulated, on file swap), the `[theme.light]`/`[theme.dark]` config (both palettes present, every theme key a registered Streamlit option), and end-to-end Runs that render a single result card and the multi-card result grid. AppTest re-executes `streamlit_app.py` in a fresh namespace on every `.run()`, so mocks must patch the **shared upstream imports** (`mlx_audio.stt.utils.load_model` for inference, `torchcodec.decoders.AudioDecoder` for header-only duration reads) rather than `streamlit_app.*` — patching `streamlit_app` attributes does not cross AppTest's script-runner boundary, and clicking Run without an upstream patch would load the real ~3.3GB speech model. An autouse fixture clears `st.cache_resource` before each test, since AppTest does not reset Streamlit's resource cache between instances. Pure-UI cases avoid clicking Run entirely.

## Resources

- [Granite Speech 4.1 2b](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) — upstream model card
- [Granite Speech 4.1 2b 8bit (MLX)](https://huggingface.co/divydeep/granite-speech-4.1-2b-mlx-8bit) — the conversion loaded at runtime
- [Granite Guardian HAP 125m](https://huggingface.co/ibm-granite/granite-guardian-hap-125m)
- [Granite Speech Models](https://huggingface.co/collections/ibm-granite/granite-speech)
- [Technical Report](https://arxiv.org/abs/2505.08699)
- [Finetune on custom data](https://github.com/ibm-granite/granite-speech-models/blob/main/notebooks/fine_tuning_granite_speech.ipynb)
- [Two-Pass Spoken Question Answering](https://github.com/ibm-granite/granite-speech-models/blob/main/notebooks/two_pass_spoken_qa.ipynb)
