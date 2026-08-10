from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import mlx.core as mx  # ty: ignore[unresolved-import]
import numpy as np
import pytest
import torch

from streamlit_app import (
    _HOIST_ATTRS,
    EN_TARGETS,
    GUARDIAN_MODEL_ID,
    MAX_SEGMENT_DURATION_S,
    MAX_VAD_OFF_DURATION_S,
    MIN_SEGMENT_DURATION_S,
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_LANGUAGES,
    SUPPORTED_FORMATS,
    TRANSCRIBE_PROMPT,
    VIDEO_FORMATS,
    PipelineResult,
    _aggregate_segment_safety,
    _encode_segment,
    _generate_from_features,
    _render_result_card,
    _row_sizes,
    _split_long_segment,
    _supports_encoder_hoist,
    apply_keywords,
    audio_duration_seconds,
    build_tasks,
    check_safety,
    compute_safety_tasks,
    format_timestamp,
    get_speech_segments,
    is_video,
    load_and_preprocess_audio,
    load_guardian_model,
    load_model,
    load_vad_model,
    produces_english,
    result_slug,
    result_title,
    run_pipeline,
    silero_vad,
    transcribe_audio,
)

# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

AUDIO_DIR = Path(__file__).parent / "data" / "audio"
NON_ENGLISH_SOURCES = tuple(s for s in SOURCE_LANGUAGES if s != "English")

# Unwrap st.cache_resource / torch.inference_mode wrappers so tests call the
# originals without going through Streamlit/torch machinery.
_load_model = load_model.__wrapped__  # ty: ignore[unresolved-attribute]
_load_guardian_model = load_guardian_model.__wrapped__  # ty: ignore[unresolved-attribute]
_load_vad_model = load_vad_model.__wrapped__  # ty: ignore[unresolved-attribute]
_run_pipeline = run_pipeline.__wrapped__  # ty: ignore[unresolved-attribute]


def make_upload(
    path: Path | None = None, raw: bytes = b"", name: str = "bad.wav"
) -> MagicMock:
    upload = MagicMock()
    if path is not None:
        upload.name = path.name
        upload.getvalue.return_value = path.read_bytes()
    else:
        upload.name = name
        upload.getvalue.return_value = raw
    return upload


def classification_calls(tokenizer: MagicMock) -> list:
    """Filter a guardian-tokenizer mock to the classification-shape calls
    only. check_safety also calls the tokenizer once per check to measure
    input length (truncation=False, add_special_tokens=False); the actual
    classification call uses padding=True."""
    return [c for c in tokenizer.call_args_list if c.kwargs.get("padding") is True]


class PipelineMocks(NamedTuple):
    model: MagicMock
    vad: MagicMock
    guardian: MagicMock
    guardian_tokenizer: MagicMock


@pytest.fixture
def pipeline_mocks() -> PipelineMocks:
    # spec= keeps the mock from auto-creating the mlx_audio internals that
    # _supports_encoder_hoist probes for, so these tests drive the public
    # model.generate() fallback path and can assert on generate.call_count.
    # The hoisted path has its own tests below.
    model = MagicMock(spec=["generate"])
    model.generate.return_value = MagicMock(text="decoded text")
    guardian_tokenizer = MagicMock()
    guardian_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    guardian = MagicMock()
    guardian.return_value.logits = torch.tensor([[5.0, -5.0]])
    return PipelineMocks(model, MagicMock(), guardian, guardian_tokenizer)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_model_id() -> None:
    assert MODEL_ID == "divydeep/granite-speech-4.1-2b-mlx-8bit"


def test_guardian_model_id() -> None:
    assert GUARDIAN_MODEL_ID == "ibm-granite/granite-guardian-hap-125m"


def test_source_languages() -> None:
    assert SOURCE_LANGUAGES[0] == "English"
    assert set(SOURCE_LANGUAGES) == {"English", *NON_ENGLISH_SOURCES}


def test_en_targets() -> None:
    assert "English" not in EN_TARGETS
    assert set(EN_TARGETS) == {
        "French",
        "German",
        "Spanish",
        "Portuguese",
        "Italian",
        "Japanese",
        "Mandarin Chinese",
    }


def test_supported_formats() -> None:
    assert set(SUPPORTED_FORMATS) == {
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
    }
    assert VIDEO_FORMATS.issubset(set(SUPPORTED_FORMATS))


def test_max_vad_off_duration_is_two_minutes() -> None:
    assert MAX_VAD_OFF_DURATION_S == 120


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestBuildTasks:
    def test_english_source(self) -> None:
        tasks = build_tasks("English")
        assert tasks["Transcribe"] == TRANSCRIBE_PROMPT
        assert "English" not in tasks
        assert set(tasks) - {"Transcribe"} == set(EN_TARGETS)
        # Insertion order drives result-card order: Transcribe comes first.
        assert next(iter(tasks)) == "Transcribe"

    @pytest.mark.parametrize("source", NON_ENGLISH_SOURCES)
    def test_non_english_source_has_two_tasks(self, source: str) -> None:
        assert set(build_tasks(source)) == {"Transcribe", "English"}

    @pytest.mark.parametrize("source", SOURCE_LANGUAGES)
    def test_translation_prompts_format(self, source: str) -> None:
        for task, prompt in build_tasks(source).items():
            if task != "Transcribe":
                assert prompt == f"translate the speech to {task}"


@pytest.mark.parametrize(
    "prompt, keywords, expected",
    [
        pytest.param("p1", [], "p1", id="no_keywords"),
        pytest.param("p", ["IBM"], "p Keywords: IBM", id="single"),
        pytest.param(
            "p",
            ["IBM", "MLX", "Granite"],
            "p Keywords: IBM, MLX, Granite",
            id="multiple",
        ),
    ],
)
def test_apply_keywords(prompt: str, keywords: list[str], expected: str) -> None:
    assert apply_keywords(prompt, keywords) == expected


PRODUCES_ENGLISH_CASES = [
    ("English", "Transcribe", True),
    *[(src, "Transcribe", False) for src in NON_ENGLISH_SOURCES],
    *[(src, "English", True) for src in NON_ENGLISH_SOURCES],
    # English audio: every target can come back as English passthrough, so all
    # of them must be scored.
    *[("English", target, True) for target in EN_TARGETS],
]


@pytest.mark.parametrize("source, task, expected", PRODUCES_ENGLISH_CASES)
def test_produces_english(source: str, task: str, expected: bool) -> None:
    assert produces_english(source, task) is expected


@pytest.mark.parametrize(
    "selected, source, toggle, expected",
    [
        pytest.param(
            ["Transcribe", "French"], "English", False, set(), id="toggle_off"
        ),
        pytest.param(
            ["Transcribe"], "English", True, {"Transcribe"}, id="english_transcribe"
        ),
        pytest.param(
            ["Transcribe"], "French", True, set(), id="non_english_transcribe"
        ),
        pytest.param(
            ["English"], "French", True, {"English"}, id="translation_to_english"
        ),
        pytest.param(
            # English audio: translations can passthrough as English, so they
            # are scored too rather than silently skipped.
            ["Transcribe", "French", "German"],
            "English",
            True,
            {"Transcribe", "French", "German"},
            id="english_source_scores_every_task",
        ),
        pytest.param(
            # French audio: Transcribe emits French and passthrough would too,
            # so only the English translation is scored.
            ["Transcribe", "English"],
            "French",
            True,
            {"English"},
            id="non_english_source_scores_english_target_only",
        ),
    ],
)
def test_compute_safety_tasks(
    selected: list[str], source: str, toggle: bool, expected: set[str]
) -> None:
    assert compute_safety_tasks(selected, source, toggle) == expected


@pytest.mark.parametrize(
    "source, task, expected",
    [
        pytest.param(
            "English", "Transcribe", "Transcribe (English)", id="en_transcribe"
        ),
        pytest.param("German", "Transcribe", "Transcribe (German)", id="de_transcribe"),
        pytest.param(
            "Japanese", "Transcribe", "Transcribe (Japanese)", id="ja_transcribe"
        ),
        pytest.param("English", "French", "French", id="en_to_fr"),
        pytest.param("German", "English", "English", id="de_to_en"),
    ],
)
def test_result_title(source: str, task: str, expected: str) -> None:
    assert result_title(source, task) == expected


@pytest.mark.parametrize(
    "source, task, expected",
    [
        pytest.param("English", "Transcribe", "transcribe_english", id="en_transcribe"),
        pytest.param("German", "Transcribe", "transcribe_german", id="de_transcribe"),
        pytest.param("English", "French", "french", id="en_to_fr"),
        pytest.param("French", "English", "english", id="fr_to_en"),
        pytest.param(
            "English", "Mandarin Chinese", "mandarin_chinese", id="multi_word_target"
        ),
    ],
)
def test_result_slug(source: str, task: str, expected: str) -> None:
    assert result_slug(source, task) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [
        pytest.param(0.0, "0:00", id="zero"),
        pytest.param(15.0, "0:15", id="seconds_only"),
        pytest.param(62.0, "1:02", id="minutes_and_seconds"),
        pytest.param(3661.0, "1:01:01", id="hours"),
        pytest.param(15.7, "0:15", id="fractional_truncated"),
    ],
)
def test_format_timestamp(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("clip.mp4", True),
        ("clip.mov", True),
        ("clip.webm", True),
        ("clip.mkv", True),
        ("CLIP.MP4", True),
        ("recordings/2026/clip.mp4", True),
        ("sound.wav", False),
        ("sound.flac", False),
        ("sound.m4a", False),
        ("sound.mp3", False),
        ("sound.ogg", False),
        ("sound.aac", False),
        ("Sound.WAV", False),
        ("filename", False),
        ("recordings/2026/sound.wav", False),
    ],
)
def test_is_video(filename: str, expected: bool) -> None:
    assert is_video(filename) is expected


@pytest.mark.parametrize(
    "n, expected",
    [
        pytest.param(0, [], id="zero"),
        pytest.param(1, [1], id="one"),
        pytest.param(2, [2], id="two"),
        pytest.param(3, [3], id="three"),
        pytest.param(4, [2, 2], id="four_balanced"),
        pytest.param(5, [3, 2], id="five"),
        pytest.param(6, [3, 3], id="six"),
        pytest.param(7, [3, 2, 2], id="seven_balanced"),
        pytest.param(8, [3, 3, 2], id="eight"),
    ],
)
def test_row_sizes(n: int, expected: list[int]) -> None:
    assert _row_sizes(n) == expected


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------


class TestSileroVad:
    def test_returns_tuples_in_seconds(self) -> None:
        mock_timestamps = [
            {"start": 16000, "end": 48000},
            {"start": 64000, "end": 96000},
        ]
        with patch("streamlit_app.get_speech_timestamps", return_value=mock_timestamps):
            result = silero_vad(torch.zeros(1, 160000), MagicMock())
        assert result == [(1.0, 3.0), (4.0, 6.0)]

    def test_empty_audio_returns_empty_list(self) -> None:
        with patch("streamlit_app.get_speech_timestamps", return_value=[]):
            assert silero_vad(torch.zeros(1, 16000), MagicMock()) == []

    def test_passes_model_and_sample_rate(self) -> None:
        model = MagicMock()
        wav = torch.zeros(1, 16000)
        with patch("streamlit_app.get_speech_timestamps", return_value=[]) as mock_fn:
            silero_vad(wav, model)
        assert torch.equal(mock_fn.call_args[0][0], wav.squeeze())
        assert mock_fn.call_args[0][1] is model
        assert mock_fn.call_args[1]["sampling_rate"] == 16000


class TestGetSpeechSegments:
    @staticmethod
    def _run(
        wav: torch.Tensor,
        vad_segments: list[tuple[float, float]],
        max_segment_duration: float = MAX_SEGMENT_DURATION_S,
    ) -> list[dict[str, float]]:
        with patch("streamlit_app.silero_vad", return_value=vad_segments):
            return get_speech_segments(
                wav, MagicMock(), max_segment_duration=max_segment_duration
            )

    def test_adds_start_and_end_buffer(self) -> None:
        result = self._run(torch.zeros(1, 160000), [(1.0, 2.0)])
        assert result[0]["start"] == pytest.approx(0.7)
        assert result[0]["end"] == pytest.approx(2.3)

    def test_clamps_start_buffer_to_zero(self) -> None:
        result = self._run(torch.zeros(1, 160000), [(0.1, 1.0)])
        assert result[0]["start"] == 0.0

    def test_clamps_end_buffer_to_duration(self) -> None:
        # 2 seconds at 16 kHz
        result = self._run(torch.zeros(1, 32000), [(0.5, 1.9)])
        assert result[0]["end"] == 2.0

    def test_merges_close_segments(self) -> None:
        result = self._run(torch.zeros(1, 160000), [(1.0, 2.0), (2.3, 3.0)])
        assert len(result) == 1
        assert result[0]["start"] == pytest.approx(0.7)
        assert result[0]["end"] == pytest.approx(3.3)

    def test_keeps_distant_segments_separate(self) -> None:
        result = self._run(torch.zeros(1, 160000), [(1.0, 2.0), (5.0, 6.0)])
        assert len(result) == 2

    def test_no_speech_falls_back_to_full_audio(self) -> None:
        # 10 seconds at 16 kHz, covered end to end but still subject to the cap.
        result = self._run(torch.zeros(1, 160000), [])
        assert result[0]["start"] == 0.0
        assert result[-1]["end"] == pytest.approx(10.0)
        assert all(
            seg["end"] - seg["start"] <= MAX_SEGMENT_DURATION_S + 1e-6 for seg in result
        )

    def test_merge_stops_at_max_duration(self) -> None:
        # Both spans are within min_gap, so the old code merged them into one
        # 9.3s segment. With a 5s cap the merge must not happen.
        result = self._run(
            torch.zeros(1, 160000), [(1.0, 5.0), (5.2, 9.0)], max_segment_duration=5.0
        )
        assert len(result) == 2

    def test_continuous_speech_does_not_collapse_into_one_segment(self) -> None:
        # 60s of speech with sub-min_gap breaks: previously one 60s segment.
        vad = [(float(i), i + 0.9) for i in range(60)]
        result = self._run(torch.zeros(1, 16000 * 60), vad, max_segment_duration=10.0)
        assert len(result) > 1
        assert all(seg["end"] - seg["start"] <= 10.0 + 1e-6 for seg in result)

    def test_single_long_span_is_split_into_equal_parts(self) -> None:
        # One unbroken 30s span, 10s cap -> 3 equal parts, not 10+10+10 by
        # chance: 31s would be 4 x 7.75s rather than 3 x 10s + 1s.
        result = self._run(
            torch.zeros(1, 16000 * 40), [(0.3, 31.3)], max_segment_duration=10.0
        )
        assert len(result) == 4
        durations = [seg["end"] - seg["start"] for seg in result]
        assert all(d == pytest.approx(durations[0]) for d in durations)
        assert all(d <= 10.0 + 1e-6 for d in durations)

    def test_split_parts_are_contiguous_and_cover_the_span(self) -> None:
        result = self._run(
            torch.zeros(1, 16000 * 40), [(0.3, 31.3)], max_segment_duration=10.0
        )
        assert result[0]["start"] == pytest.approx(0.0)
        assert result[-1]["end"] == pytest.approx(31.6)
        for earlier, later in pairwise(result):
            assert earlier["end"] == pytest.approx(later["start"])

    def test_no_speech_fallback_is_also_capped(self) -> None:
        result = self._run(torch.zeros(1, 16000 * 25), [], max_segment_duration=10.0)
        assert len(result) == 3
        assert all(seg["end"] - seg["start"] <= 10.0 + 1e-6 for seg in result)

    def test_short_segments_are_untouched(self) -> None:
        result = self._run(
            torch.zeros(1, 160000), [(1.0, 2.0)], max_segment_duration=10.0
        )
        assert result == [{"start": pytest.approx(0.7), "end": pytest.approx(2.3)}]

    def test_segments_never_overlap_when_the_cap_blocks_a_merge(self) -> None:
        # Buffering pulls each span's start 0.3s back and pushes its end 0.3s
        # on, so consecutive spans overlap by 0.5s. Merging used to absorb that
        # unconditionally; once the cap can refuse the merge, the overlap has to
        # be clamped or the same audio is transcribed twice and the rendered
        # timestamps run backwards.
        vad = [(float(i), i + 0.9) for i in range(60)]
        result = self._run(torch.zeros(1, 16000 * 60), vad, max_segment_duration=8.0)
        assert len(result) > 1
        for earlier, later in pairwise(result):
            assert later["start"] >= earlier["end"] - 1e-6

    def test_segments_are_monotonic_across_cap_values(self) -> None:
        vad = [(float(i), i + 0.9) for i in range(40)]
        for cap in (5.0, 8.0, 10.0, 15.0):
            result = self._run(
                torch.zeros(1, 16000 * 40), vad, max_segment_duration=cap
            )
            for earlier, later in pairwise(result):
                assert later["start"] >= earlier["end"] - 1e-6, f"overlap at cap={cap}"
            assert all(seg["end"] > seg["start"] for seg in result)

    def test_span_just_over_the_cap_is_not_split_below_the_floor(self) -> None:
        # 8.3s at an 8.0s cap would halve into 2 x 4.15s, landing in the bucket
        # the measurements call out as worse (4s -> 12/16) than the overshoot.
        parts = _split_long_segment(0.0, 8.3, 8.0)
        assert len(parts) == 1
        assert parts[0] == {"start": 0.0, "end": 8.3}

    def test_long_spans_still_split_above_the_floor(self) -> None:
        parts = _split_long_segment(0.0, 24.0, 8.0)
        assert len(parts) == 3
        assert all(
            p["end"] - p["start"] >= MIN_SEGMENT_DURATION_S - 1e-6 for p in parts
        )

    def test_floor_never_defeats_a_cap_smaller_than_it(self) -> None:
        # A caller passing max_duration below the floor must still get a split.
        parts = _split_long_segment(0.0, 12.0, 4.0)
        assert len(parts) == 3
        assert all(p["end"] - p["start"] <= 4.0 + 1e-6 for p in parts)

    def test_default_cap_is_below_the_measured_translation_cliff(self) -> None:
        # 8s is the widest cap measured to translate every segment (10s already
        # drops to 2/4), so anchor on the measurement rather than on the ~20s
        # cliff — a loose bound here would go green at a known-broken cap.
        assert 0 < MAX_SEGMENT_DURATION_S <= 8.0


# ---------------------------------------------------------------------------
# Audio IO
# ---------------------------------------------------------------------------


class TestLoadAndPreprocessAudio:
    @pytest.mark.parametrize("filename", ["sample_10s.wav", "sample_10s_video.mp4"])
    def test_loads_real_audio(self, filename: str) -> None:
        wav = load_and_preprocess_audio(make_upload(AUDIO_DIR / filename))
        assert wav.shape[0] == 1
        assert wav.shape[1] > 0

    def test_zero_sample_audio_raises_a_readable_error(self) -> None:
        # Caught at load, not downstream: an empty waveform passes VAD (which
        # falls back to a zero-length segment) and would otherwise die as an
        # opaque reshape error inside the mel filterbank.
        with (
            patch(
                "streamlit_app.torchaudio.load", return_value=(torch.zeros(1, 0), 16000)
            ),
            pytest.raises(RuntimeError, match="No audio detected"),
        ):
            load_and_preprocess_audio(make_upload(AUDIO_DIR / "sample_10s.wav"))

    def test_invalid_audio_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError, match="Failed to load audio file"):
            load_and_preprocess_audio(make_upload(raw=b"not audio data"))


class TestAudioDurationSeconds:
    @pytest.mark.parametrize("filename", ["sample_10s.wav", "sample_10s_video.mp4"])
    def test_duration_matches_fixture(self, filename: str) -> None:
        duration = audio_duration_seconds(make_upload(AUDIO_DIR / filename))
        assert duration is not None
        assert 9.5 < duration < 10.5

    def test_invalid_audio_returns_none(self) -> None:
        assert audio_duration_seconds(make_upload(raw=b"not audio data")) is None


# ---------------------------------------------------------------------------
# Model loaders (test the unwrapped originals)
# ---------------------------------------------------------------------------


@patch("streamlit_app.st")
class TestLoadModel:
    @patch("streamlit_app._load_stt_model")
    def test_calls_load_stt_model_and_returns_result(
        self, mock_load: MagicMock, _mock_st: MagicMock
    ) -> None:
        result = _load_model("test-model")
        mock_load.assert_called_once_with("test-model", revision=None)
        assert result == mock_load.return_value

    @patch("streamlit_app._load_stt_model")
    def test_forwards_revision_pin(
        self, mock_load: MagicMock, _mock_st: MagicMock
    ) -> None:
        _load_model(MODEL_ID, MODEL_REVISION)
        mock_load.assert_called_once_with(MODEL_ID, revision=MODEL_REVISION)


@patch("streamlit_app.st")
class TestLoadGuardianModel:
    @patch("streamlit_app.AutoModelForSequenceClassification")
    @patch("streamlit_app.AutoTokenizer")
    def test_loads_model_and_tokenizer(
        self,
        mock_tokenizer_cls: MagicMock,
        mock_model_cls: MagicMock,
        _mock_st: MagicMock,
    ) -> None:
        result = _load_guardian_model("test-model")
        mock_tokenizer_cls.from_pretrained.assert_called_once_with("test-model")
        mock_model_cls.from_pretrained.assert_called_once_with("test-model")
        assert result == (
            mock_model_cls.from_pretrained.return_value,
            mock_tokenizer_cls.from_pretrained.return_value,
        )


@patch("streamlit_app.st")
class TestLoadVadModel:
    @patch("streamlit_app.load_silero_vad")
    def test_loads_model(self, mock_load: MagicMock, _mock_st: MagicMock) -> None:
        result = _load_vad_model()
        mock_load.assert_called_once()
        assert result == mock_load.return_value


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


class TestCheckSafety:
    @staticmethod
    def _mocks(logits: list[list[float]]) -> tuple[MagicMock, MagicMock]:
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        model = MagicMock()
        model.return_value.logits = torch.tensor(logits)
        return model, tokenizer

    def test_safe_content(self) -> None:
        model, tokenizer = self._mocks([[5.0, -5.0]])
        is_toxic, score = check_safety("safe text", model, tokenizer)
        assert is_toxic is False
        assert score < 0.5

    def test_toxic_content(self) -> None:
        model, tokenizer = self._mocks([[-5.0, 5.0]])
        is_toxic, score = check_safety("toxic text", model, tokenizer)
        assert is_toxic is True
        assert score > 0.5

    def test_boundary_is_not_toxic(self) -> None:
        model, tokenizer = self._mocks([[0.0, 0.0]])
        is_toxic, score = check_safety("text", model, tokenizer)
        assert score == 0.5
        assert is_toxic is False

    def test_tokenizer_called_with_classification_args(self) -> None:
        # check_safety now calls the tokenizer twice for short inputs:
        # once with the raw text (length check, no special tokens) and once
        # with [text] in the standard classification shape. Verify the
        # classification call is present.
        model, tokenizer = self._mocks([[5.0, -5.0]])
        check_safety("hello world", model, tokenizer)
        tokenizer.assert_any_call(
            ["hello world"], padding=True, truncation=True, return_tensors="pt"
        )

    def test_long_text_chunks_and_returns_max_score(self) -> None:
        # 1099 tokens → 3 chunks (510, 510, 79). Aggregation must surface
        # the toxic middle chunk, not average it away.
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([list(range(1, 1100))])}
        tokenizer.decode.return_value = "chunk text"
        safe = MagicMock(logits=torch.tensor([[5.0, -5.0]]))
        toxic = MagicMock(logits=torch.tensor([[-5.0, 5.0]]))
        model = MagicMock(side_effect=[safe, toxic, safe])

        is_toxic, score = check_safety("long text", model, tokenizer)
        assert is_toxic is True
        assert score > 0.5
        assert model.call_count == 3


class TestAggregateSegmentSafety:
    @staticmethod
    def _mocks(*score_pairs: tuple[float, float]) -> tuple[MagicMock, MagicMock]:
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        responses = [MagicMock(logits=torch.tensor([list(p)])) for p in score_pairs]
        return MagicMock(side_effect=responses), tokenizer

    def test_empty_list_returns_safe_zero(self) -> None:
        tokenizer = MagicMock()
        model = MagicMock()
        assert _aggregate_segment_safety([], model, tokenizer) == (False, 0.0)
        tokenizer.assert_not_called()
        model.assert_not_called()

    def test_skips_whitespace_only_segments(self) -> None:
        model, tokenizer = self._mocks((5.0, -5.0))
        is_toxic, _ = _aggregate_segment_safety(
            ["", "   ", "real text"], model, tokenizer
        )
        assert is_toxic is False
        assert model.call_count == 1

    def test_returns_max_probability(self) -> None:
        model, tokenizer = self._mocks((5.0, -5.0), (-5.0, 5.0), (5.0, -5.0))
        is_toxic, score = _aggregate_segment_safety(["a", "b", "c"], model, tokenizer)
        assert is_toxic is True
        assert score > 0.5


# ---------------------------------------------------------------------------
# Transcription primitive
# ---------------------------------------------------------------------------


class TestTranscribeAudio:
    def test_calls_model_generate_with_kwargs(self) -> None:
        model = MagicMock()
        model.generate.return_value = MagicMock(text="transcribed text")
        transcript = transcribe_audio(torch.zeros(1, 16000), "test prompt", model)
        kwargs = model.generate.call_args[1]
        assert kwargs["prompt"] == "test prompt"
        assert kwargs["max_tokens"] == 512
        assert transcript == "transcribed text"

    def test_audio_passed_as_squeezed_numpy(self) -> None:
        model = MagicMock()
        model.generate.return_value = MagicMock(text="text")
        transcribe_audio(torch.randn(1, 16000), "prompt", model)
        audio = model.generate.call_args[1]["audio"]
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (16000,)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

SEGMENTS_3S = [{"start": 0.0, "end": 1.5}, {"start": 1.5, "end": 3.0}]
TRANSCRIBE_PLUS_ONE = {"Transcribe": TRANSCRIBE_PROMPT, "French": "translate to French"}


class TestRunPipeline:
    @pytest.fixture(autouse=True)
    def _patch_default_segments(self) -> Iterator[None]:
        with patch(
            "streamlit_app.get_speech_segments",
            return_value=[{"start": 0.0, "end": 1.0}],
        ):
            yield

    def test_returns_dict_keyed_by_task(self, pipeline_mocks: PipelineMocks) -> None:
        tasks = {"Transcribe": "transcribe prompt", "French": "translate to French"}
        results = _run_pipeline(
            torch.zeros(1, 16000), tasks, {"Transcribe"}, *pipeline_mocks
        )
        assert set(results) == {"Transcribe", "French"}

    def test_each_result_has_transcript(self, pipeline_mocks: PipelineMocks) -> None:
        results = _run_pipeline(
            torch.zeros(1, 16000), {"Transcribe": "p"}, {"Transcribe"}, *pipeline_mocks
        )
        assert "transcript" in results["Transcribe"]

    def test_empty_tasks_returns_empty_dict(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        assert _run_pipeline(torch.zeros(1, 16000), {}, set(), *pipeline_mocks) == {}

    def test_uses_correct_prompt_per_task(self, pipeline_mocks: PipelineMocks) -> None:
        tasks = {
            "Transcribe": "transcribe prompt",
            "French": "translate to French",
            "German": "translate to German",
        }
        _run_pipeline(torch.zeros(1, 16000), tasks, {"Transcribe"}, *pipeline_mocks)
        prompts = [c[1]["prompt"] for c in pipeline_mocks.model.generate.call_args_list]
        assert prompts == [
            "transcribe prompt",
            "translate to French",
            "translate to German",
        ]

    def test_preserves_task_order(self, pipeline_mocks: PipelineMocks) -> None:
        tasks = {"Japanese": "p1", "Transcribe": "p2", "German": "p3"}
        results = _run_pipeline(
            torch.zeros(1, 16000), tasks, {"Transcribe"}, *pipeline_mocks
        )
        assert list(results) == ["Japanese", "Transcribe", "German"]

    def test_safety_fields_when_in_safety_tasks(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        results = _run_pipeline(
            torch.zeros(1, 16000), {"Transcribe": "p"}, {"Transcribe"}, *pipeline_mocks
        )
        result = results["Transcribe"]
        assert result["is_toxic"] is False
        assert "toxicity_score" in result

    def test_toxic_content_flagged(self, pipeline_mocks: PipelineMocks) -> None:
        pipeline_mocks.guardian.return_value.logits = torch.tensor([[-5.0, 5.0]])
        results = _run_pipeline(
            torch.zeros(1, 16000), {"Transcribe": "p"}, {"Transcribe"}, *pipeline_mocks
        )
        assert results["Transcribe"]["is_toxic"] is True
        assert results["Transcribe"]["toxicity_score"] > 0.5

    def test_safety_check_receives_transcript(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        _run_pipeline(
            torch.zeros(1, 16000), {"Transcribe": "p"}, {"Transcribe"}, *pipeline_mocks
        )
        calls = classification_calls(pipeline_mocks.guardian_tokenizer)
        assert len(calls) == 1
        assert calls[0].args == (["decoded text"],)
        assert calls[0].kwargs == {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }

    def test_tasks_not_in_safety_set_skip_safety_check(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        results = _run_pipeline(
            torch.zeros(1, 16000),
            {"French": "translate to French"},
            set(),
            *pipeline_mocks,
        )
        assert "is_toxic" not in results["French"]
        pipeline_mocks.guardian_tokenizer.assert_not_called()

    def test_mixed_tasks_safety_only_on_listed(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        results = _run_pipeline(
            torch.zeros(1, 16000),
            {"Transcribe": "p1", "French": "p2"},
            {"Transcribe"},
            *pipeline_mocks,
        )
        assert "is_toxic" in results["Transcribe"]
        assert "is_toxic" not in results["French"]

    def test_non_english_to_english_translation_gets_safety(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        results = _run_pipeline(
            torch.zeros(1, 16000),
            {"English": "translate to English"},
            {"English"},
            *pipeline_mocks,
        )
        assert "is_toxic" in results["English"]

    def test_multi_segment_transcript_has_timestamps(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        with patch("streamlit_app.get_speech_segments", return_value=SEGMENTS_3S):
            results = _run_pipeline(
                torch.zeros(1, 48000),
                {"Transcribe": "p"},
                {"Transcribe"},
                *pipeline_mocks,
            )
        transcript = results["Transcribe"]["transcript"]
        assert "[0:00 - 0:01]" in transcript
        assert "[0:01 - 0:03]" in transcript
        assert "\n" in transcript

    def test_multi_segment_safety_runs_per_segment(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        with patch("streamlit_app.get_speech_segments", return_value=SEGMENTS_3S):
            _run_pipeline(
                torch.zeros(1, 48000),
                {"Transcribe": "p"},
                {"Transcribe"},
                *pipeline_mocks,
            )
        calls = classification_calls(pipeline_mocks.guardian_tokenizer)
        assert len(calls) == 2
        for call in calls:
            assert call.args == (["decoded text"],)
            assert call.kwargs == {
                "padding": True,
                "truncation": True,
                "return_tensors": "pt",
            }

    def test_multi_segment_safety_reports_max_score(self) -> None:
        # A single toxic segment must flag the whole transcript, even when
        # surrounded by safe segments — i.e., aggregation is max, not mean.
        model = MagicMock(spec=["generate"])
        model.generate.return_value = MagicMock(text="decoded text")
        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        safe = MagicMock(logits=torch.tensor([[5.0, -5.0]]))
        toxic = MagicMock(logits=torch.tensor([[-5.0, 5.0]]))
        guardian = MagicMock(side_effect=[safe, toxic, safe])
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.0, "end": 2.0},
            {"start": 2.0, "end": 3.0},
        ]
        with patch("streamlit_app.get_speech_segments", return_value=segments):
            results = _run_pipeline(
                torch.zeros(1, 48000),
                {"Transcribe": "p"},
                {"Transcribe"},
                model,
                MagicMock(),
                guardian,
                tokenizer,
            )
        assert results["Transcribe"]["is_toxic"] is True
        assert results["Transcribe"]["toxicity_score"] > 0.5

    def test_safety_skips_empty_segment_text(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        # If the model returns "" for a segment, the guardian shouldn't be
        # called for it — saves an inference and avoids spurious scores.
        pipeline_mocks.model.generate.side_effect = [
            MagicMock(text=""),
            MagicMock(text="real transcript"),
        ]
        with patch("streamlit_app.get_speech_segments", return_value=SEGMENTS_3S):
            _run_pipeline(
                torch.zeros(1, 48000),
                {"Transcribe": "p"},
                {"Transcribe"},
                *pipeline_mocks,
            )
        assert len(classification_calls(pipeline_mocks.guardian_tokenizer)) == 1

    def test_segmentation_off_skips_vad(self, pipeline_mocks: PipelineMocks) -> None:
        with patch("streamlit_app.get_speech_segments") as mock_vad:
            _run_pipeline(
                torch.zeros(1, 16000),
                {"Transcribe": "p"},
                {"Transcribe"},
                pipeline_mocks.model,
                None,
                pipeline_mocks.guardian,
                pipeline_mocks.guardian_tokenizer,
                use_segmentation=False,
            )
        mock_vad.assert_not_called()

    def test_segmentation_off_uses_full_audio(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        results = _run_pipeline(
            torch.zeros(1, 48000),  # 3 seconds at 16 kHz
            {"Transcribe": "p"},
            {"Transcribe"},
            pipeline_mocks.model,
            None,
            pipeline_mocks.guardian,
            pipeline_mocks.guardian_tokenizer,
            use_segmentation=False,
        )
        assert pipeline_mocks.model.generate.call_count == 1
        assert "[0:00 - 0:03]" in results["Transcribe"]["transcript"]

    def test_transcribe_plus_one_translation_gets_no_special_casing(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        # Transcribe + exactly one translation used to trigger a combined
        # chain-of-thought prompt. The model never emitted the [Transcription]
        # / [Translation] tags that path parsed for, so it always fell back —
        # costing an extra inference per segment. Each task now runs its own
        # inference: 2 segments x 2 tasks = 4 calls, no more, no fewer.
        with patch("streamlit_app.get_speech_segments", return_value=SEGMENTS_3S):
            _run_pipeline(
                torch.zeros(1, 48000), TRANSCRIBE_PLUS_ONE, set(), *pipeline_mocks
            )
        assert pipeline_mocks.model.generate.call_count == 4
        prompts = [c[1]["prompt"] for c in pipeline_mocks.model.generate.call_args_list]
        # Segments outer, tasks inner: both tasks run against segment 1 before
        # segment 2, which is what lets the encoder be hoisted per segment.
        assert prompts == [TRANSCRIBE_PROMPT, "translate to French"] * 2

    def test_encoder_hoisted_when_model_supports_it(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        # A model exposing the mlx_audio internals takes the hoisted path:
        # one encode per segment, one decode per (segment, task) — not one
        # encode per pair.
        model = MagicMock()  # unspec'd: has every _HOIST_ATTRS attribute
        with (
            patch("streamlit_app.get_speech_segments", return_value=SEGMENTS_3S),
            patch(
                "streamlit_app._encode_segment", return_value=("features", 7)
            ) as encode,
            patch(
                "streamlit_app._generate_from_features", return_value="text"
            ) as decode,
        ):
            _run_pipeline(
                torch.zeros(1, 48000),
                TRANSCRIBE_PLUS_ONE,
                set(),
                model,
                *pipeline_mocks[1:],
            )
        assert encode.call_count == 2  # 2 segments
        assert decode.call_count == 4  # 2 segments x 2 tasks
        assert model.generate.call_count == 0  # public path not used

    def test_falls_back_to_generate_when_internals_absent(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        # pipeline_mocks.model is spec'd to `generate` only, so the hoist is
        # unavailable and the public path must still produce results.
        with patch("streamlit_app._encode_segment") as encode:
            results = _run_pipeline(
                torch.zeros(1, 16000), TRANSCRIBE_PLUS_ONE, set(), *pipeline_mocks
            )
        encode.assert_not_called()
        assert pipeline_mocks.model.generate.call_count == 2
        assert set(results) == {"Transcribe", "French"}

    @pytest.mark.parametrize(
        "attrs",
        [
            pytest.param(["generate"], id="none_present"),
            pytest.param(
                ["generate", "_extract_features", "get_audio_features"],
                id="partially_present",
            ),
        ],
    )
    def test_hoist_requires_every_internal(self, attrs: list[str]) -> None:
        assert _supports_encoder_hoist(MagicMock(spec=attrs)) is False

    def test_signature_drift_warns_and_falls_back(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        # All five names present, so the hasattr probe says yes, but the call
        # raises — the shape an upstream signature change actually takes.
        model = MagicMock()
        model._extract_features.side_effect = TypeError("unexpected keyword")
        model.generate.return_value = MagicMock(text="fallback text")
        assert _supports_encoder_hoist(model) is True
        with (
            patch("streamlit_app.get_speech_segments", return_value=SEGMENTS_3S),
            pytest.warns(RuntimeWarning, match="Encoder hoist failed"),
        ):
            results = _run_pipeline(
                torch.zeros(1, 48000),
                TRANSCRIBE_PLUS_ONE,
                set(),
                model,
                *pipeline_mocks[1:],
            )
        assert model.generate.call_count == 4  # 2 segments x 2 tasks
        assert "fallback text" in results["Transcribe"]["transcript"]

    def test_hoist_is_not_retried_after_it_fails(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        model = MagicMock()
        model._extract_features.side_effect = TypeError("boom")
        model.generate.return_value = MagicMock(text="t")
        with (
            patch("streamlit_app.get_speech_segments", return_value=SEGMENTS_3S),
            pytest.warns(RuntimeWarning),
        ):
            _run_pipeline(
                torch.zeros(1, 48000),
                TRANSCRIBE_PLUS_ONE,
                set(),
                model,
                *pipeline_mocks[1:],
            )
        # Disabled for the whole run, not re-attempted once per segment.
        assert model._extract_features.call_count == 1

    def test_progress_reaches_full_before_the_safety_pass(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        calls: list[tuple[int, int, str]] = []
        with patch("streamlit_app.get_speech_segments", return_value=SEGMENTS_3S):
            _run_pipeline(
                torch.zeros(1, 48000),
                TRANSCRIBE_PLUS_ONE,
                {"Transcribe"},
                *pipeline_mocks,
                on_progress=lambda i, total, task: calls.append((i, total, task)),
            )
        # The guardian pass is real work; the bar must not sit at (total-1)/total
        # and keep naming the task that already finished.
        assert calls[-1] == (4, 4, "safety check")
        assert calls[0] == (0, 4, "Transcribe")

    def test_progress_final_label_when_no_safety_pass(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        calls: list[tuple[int, int, str]] = []
        with patch("streamlit_app.get_speech_segments", return_value=SEGMENTS_3S):
            _run_pipeline(
                torch.zeros(1, 48000),
                TRANSCRIBE_PLUS_ONE,
                set(),
                *pipeline_mocks,
                on_progress=lambda i, total, task: calls.append((i, total, task)),
            )
        assert calls[-1] == (4, 4, "results")

    def test_keywords_applied_to_transcription_only(
        self, pipeline_mocks: PipelineMocks
    ) -> None:
        # The UI promises keywords are "boosted during transcription", and
        # appending them to a translation prompt makes the model fall back to
        # untranslated, truncated source text. Transcribe gets them; translation
        # targets must not.
        _run_pipeline(
            torch.zeros(1, 16000),
            TRANSCRIBE_PLUS_ONE,
            set(),
            *pipeline_mocks,
            keywords=["IBM", "MLX"],
        )
        prompts = [c[1]["prompt"] for c in pipeline_mocks.model.generate.call_args_list]
        assert prompts == [
            f"{TRANSCRIBE_PROMPT} Keywords: IBM, MLX",
            "translate to French",
        ]


# ---------------------------------------------------------------------------
# Encoder-hoist internals
# ---------------------------------------------------------------------------


class TestEncoderHoistInternals:
    """Drives the real _encode_segment / _generate_from_features bodies.

    Every other test patches both out, and the pipeline mocks are spec'd so the
    hoist is never taken — so without this class the code reaching into
    mlx_audio and mlx_lm internals never executes in CI, and an upstream
    signature change would ship green. _supports_encoder_hoist cannot help:
    it only checks that names exist.
    """

    @staticmethod
    def _stub_model() -> MagicMock:
        model = MagicMock()
        model._extract_features.return_value = (mx.zeros((1, 4, 8)), 7)
        model.get_audio_features.return_value = mx.zeros((1, 7, 8))
        model._build_prompt.return_value = mx.zeros((1, 5), dtype=mx.int32)
        model._build_inputs_embeds.return_value = mx.zeros((1, 5, 8))
        model._tokenizer.eos_token_id = 99
        model._tokenizer.decode.return_value = "stub transcript"
        return model

    def test_encode_segment_returns_features_and_token_count(self) -> None:
        model = self._stub_model()
        features, num_tokens = _encode_segment(torch.zeros(1, 16000), model)
        assert num_tokens == 7
        assert features.shape == (1, 7, 8)
        # mlx_audio takes a numpy array, not a torch tensor.
        (audio,) = model._extract_features.call_args[0]
        assert isinstance(audio, np.ndarray)

    def test_generate_from_features_stops_at_eos(self) -> None:
        model = self._stub_model()
        with patch(
            "mlx_lm.generate.generate_step",
            return_value=iter([(1, None), (2, None), (99, None), (3, None)]),
        ):
            text = _generate_from_features(mx.zeros((1, 7, 8)), 7, "prompt", model)
        assert text == "stub transcript"
        tokens = model._tokenizer.decode.call_args[0][0]
        assert tokens == [1, 2]  # stops at eos, and does not emit it
        assert model._tokenizer.decode.call_args[1] == {"skip_special_tokens": True}

    def test_generate_from_features_holds_the_mlx_lm_contract(self) -> None:
        model = self._stub_model()
        with patch(
            "mlx_lm.generate.generate_step", return_value=iter([(99, None)])
        ) as gen:
            _generate_from_features(mx.zeros((1, 7, 8)), 7, "a prompt", model)
        # These six kwargs are the contract with mlx_lm. A rename there is
        # exactly the drift the hasattr probe cannot see.
        assert set(gen.call_args[1]) == {
            "prompt",
            "input_embeddings",
            "model",
            "max_tokens",
            "sampler",
            "prefill_step_size",
        }
        model._build_prompt.assert_called_once_with(7, "a prompt")

    def test_real_granite_model_class_still_exposes_the_internals(self) -> None:
        # Guards the actual installed mlx_audio, which the stubs above cannot.
        granite = pytest.importorskip("mlx_audio.stt.models.granite_speech")
        missing = [
            attr
            for attr in _HOIST_ATTRS
            if attr != "_tokenizer"  # set per-instance at load time
            and not hasattr(granite.Model, attr)
        ]
        assert not missing, f"mlx_audio drift: {missing}"


# ---------------------------------------------------------------------------
# Result card rendering
# ---------------------------------------------------------------------------


@patch("streamlit_app.st")
class TestRenderResultCard:
    def test_renders_text(self, mock_st: MagicMock) -> None:
        result: PipelineResult = {"transcript": "hello"}
        _render_result_card("English", "Transcribe", result, "test")
        mock_st.text.assert_called_once_with("hello")

    def test_shows_safe_banner(self, mock_st: MagicMock) -> None:
        result: PipelineResult = {
            "transcript": "hello",
            "is_toxic": False,
            "toxicity_score": 0.1,
        }
        _render_result_card("English", "Transcribe", result, "test")
        msg = mock_st.success.call_args[0][0]
        assert "safe" in msg.lower()
        assert "10.0%" in msg
        assert mock_st.success.call_args.kwargs["icon"] == ":material/check_circle:"

    def test_shows_toxic_banner(self, mock_st: MagicMock) -> None:
        result: PipelineResult = {
            "transcript": "bad",
            "is_toxic": True,
            "toxicity_score": 0.9,
        }
        _render_result_card("English", "Transcribe", result, "test")
        msg = mock_st.warning.call_args[0][0]
        assert "toxic" in msg.lower()
        assert "90.0%" in msg
        assert mock_st.warning.call_args.kwargs["icon"] == ":material/warning:"

    def test_no_safety_banner_without_toxic_field(self, mock_st: MagicMock) -> None:
        result: PipelineResult = {"transcript": "bonjour"}
        _render_result_card("English", "French", result, "test")
        mock_st.success.assert_not_called()
        mock_st.warning.assert_not_called()

    def test_download_filename_for_translation(self, mock_st: MagicMock) -> None:
        result: PipelineResult = {"transcript": "hello world"}
        _render_result_card("English", "Mandarin Chinese", result, "audio")
        assert mock_st.download_button.call_args[0][2] == "audio_mandarin_chinese.txt"

    def test_download_filename_for_transcription_includes_source(
        self, mock_st: MagicMock
    ) -> None:
        result: PipelineResult = {"transcript": "hallo"}
        _render_result_card("German", "Transcribe", result, "audio")
        assert mock_st.download_button.call_args[0][2] == "audio_transcribe_german.txt"

    def test_download_tooltip_for_transcription(self, mock_st: MagicMock) -> None:
        result: PipelineResult = {"transcript": "hello"}
        _render_result_card("English", "Transcribe", result, "test")
        assert mock_st.download_button.call_args[1]["help"] == "Download transcription"

    def test_download_tooltip_for_translation(self, mock_st: MagicMock) -> None:
        result: PipelineResult = {"transcript": "bonjour"}
        _render_result_card("English", "French", result, "test")
        assert mock_st.download_button.call_args[1]["help"] == "Download translation"

    def test_renders_bordered_container(self, mock_st: MagicMock) -> None:
        result: PipelineResult = {"transcript": "hello"}
        _render_result_card("English", "Transcribe", result, "test")
        mock_st.container.assert_called_once_with(border=True, height="stretch")

    def test_transcription_card_title_includes_source(self, mock_st: MagicMock) -> None:
        result: PipelineResult = {"transcript": "hallo"}
        _render_result_card("German", "Transcribe", result, "test")
        mock_st.subheader.assert_called_once_with("Transcribe (German)")

    def test_translation_card_title_is_target(self, mock_st: MagicMock) -> None:
        result: PipelineResult = {"transcript": "bonjour"}
        _render_result_card("English", "French", result, "test")
        mock_st.subheader.assert_called_once_with("French")
