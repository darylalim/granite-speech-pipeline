"""End-to-end UI tests driving the real main() widget tree via AppTest.

These complement test_streamlit_app.py (which mocks `st` entirely and never
exercises the UI wiring). AppTest re-executes streamlit_app.py in a fresh
namespace on every .run(), so mocks must patch the SHARED upstream imports
(mlx_audio, torchcodec) rather than streamlit_app.* — patching streamlit_app
attributes does not cross AppTest's script-runner boundary, and clicking Run
without an upstream patch would load the real ~3.3GB speech model.
"""

import tomllib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit import config
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parent.parent / "streamlit_app.py"
AUDIO_DIR = Path(__file__).parent / "data" / "audio"
CONFIG = Path(__file__).parent.parent / ".streamlit" / "config.toml"


@pytest.fixture
def audio_bytes() -> bytes:
    return (AUDIO_DIR / "sample_10s.wav").read_bytes()


@pytest.fixture
def app_config() -> dict:
    """The parsed .streamlit/config.toml. Read as UTF-8 explicitly, which TOML
    mandates and Streamlit itself passes — `read_text()` would otherwise decode
    the file's comments with whatever the platform locale happens to be."""
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def _app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=60)


@pytest.fixture(autouse=True)
def _clear_streamlit_caches() -> Iterator[None]:
    # AppTest does not reset Streamlit's process-global @st.cache_resource store
    # between instances, so a cached (mocked or real) model could leak across
    # tests — which would make the Run-path loader.assert_called() guard
    # order-dependent. Clear it before every test for isolation.
    st.cache_resource.clear()
    yield


def test_default_state() -> None:
    """App renders without exception in the documented default widget state."""
    at = _app().run()
    assert not at.exception
    assert at.title[0].value == "Granite Speech Studio"
    assert at.segmented_control[0].value == "English"
    assert at.pills[0].value == ["Transcribe"]
    assert at.pills[0].options == [
        "Transcribe",
        "French",
        "German",
        "Spanish",
        "Portuguese",
        "Italian",
        "Japanese",
        "Mandarin Chinese",
    ]
    assert {t.key for t in at.toggle} == {"use_segmentation", "use_toxicity_check"}
    assert all(t.value is True for t in at.toggle)
    # Run is disabled until audio is loaded.
    assert at.button[0].disabled is True


def test_faux_labels_are_bold() -> None:
    """The VAD / Toxicity / Keywords pseudo-labels render as bold markdown so
    they read as form labels (the real widget labels are collapsed)."""
    at = _app().run()
    values = {m.value for m in at.markdown}
    assert "**VAD segmentation**" in values
    assert "**Toxicity check**" in values
    assert "**Keywords**" in values


def test_run_button_gating(audio_bytes: bytes) -> None:
    """Run enables once audio + a task are present and re-disables when tasks
    are cleared."""
    at = _app().run()
    at.file_uploader[0].set_value(("sample_10s.wav", audio_bytes, "audio/wav"))
    at.run()
    assert at.button[0].disabled is False
    assert at.caption[0].value == "sample_10s.wav"

    at.pills[0].set_value([])
    at.run()
    assert at.button[0].disabled is True


def test_source_change_resets_tasks() -> None:
    """A non-English source narrows tasks to Transcribe + English translation
    and resets the selection to Transcribe."""
    at = _app().run()
    at.segmented_control[0].set_value("French")
    at.run()
    assert at.pills[0].options == ["Transcribe", "English"]
    assert at.pills[0].value == ["Transcribe"]


def test_vad_off_long_audio_disables_run(audio_bytes: bytes) -> None:
    """With VAD off and audio over MAX_VAD_OFF_DURATION_S, Run is disabled and a
    warning is shown. Matched on the stable prefix rather than the minute count
    so retuning the constant doesn't break this test."""
    with patch("torchcodec.decoders.AudioDecoder") as decoder:
        decoder.return_value.metadata.duration_seconds = 600.0
        at = _app().run()
        at.file_uploader[0].set_value(("long.wav", audio_bytes, "audio/wav"))
        at.run()
        at.toggle(key="use_segmentation").set_value(False)
        at.run()
    assert at.button[0].disabled is True
    vad_warning = next(
        (w for w in at.warning if w.value.startswith("Enable VAD segmentation")), None
    )
    assert vad_warning is not None
    assert vad_warning.icon == ":material/warning:"


def test_duration_cache_is_single_slot(audio_bytes: bytes) -> None:
    """With VAD off, the duration cache is a single `_duration` slot holding the
    current file's ((name, size), duration) — swapping files overwrites it rather
    than accumulating an entry per file, so it can't grow unbounded."""
    size = len(audio_bytes)
    at = _app().run()
    at.toggle(key="use_segmentation").set_value(False)
    at.file_uploader[0].set_value(("a.wav", audio_bytes, "audio/wav"))
    at.run()
    assert at.session_state["_duration"][0] == ("a.wav", size)

    at.file_uploader[0].set_value(("b.wav", audio_bytes, "audio/wav"))
    at.run()
    # Slot is overwritten in place, not accumulated.
    assert at.session_state["_duration"][0] == ("b.wav", size)
    duration_slots = [k for k in at.session_state.filtered_state if k == "_duration"]
    assert duration_slots == ["_duration"]


def test_run_renders_result_card(audio_bytes: bytes) -> None:
    """Clicking Run with mocked inference renders a result card. Patches the
    upstream MLX loader so no real model loads."""
    # spec= excludes the mlx_audio internals _supports_encoder_hoist probes
    # for, so the app takes the public generate() path against this fake.
    fake_model = MagicMock(spec=["generate"])
    fake_model.generate.return_value.text = "the quick brown fox (mocked)"
    with patch("mlx_audio.stt.utils.load_model", return_value=fake_model) as loader:
        at = _app().run()
        at.file_uploader[0].set_value(("sample_10s.wav", audio_bytes, "audio/wav"))
        at.toggle(key="use_segmentation").set_value(False)
        at.toggle(key="use_toxicity_check").set_value(False)
        at.run()
        at.button[0].click()
        at.run()
    assert not at.exception
    loader.assert_called()  # guard: the real speech model must never load
    assert at.subheader[0].value == "Transcribe (English)"
    assert "mocked" in at.text[0].value


def test_run_renders_multiple_result_cards(audio_bytes: bytes) -> None:
    """Transcribe + one translation drives the multi-card result grid (the
    N>1 _row_sizes -> st.columns loop in main()), which the single-task Run
    test does not exercise."""
    # spec= excludes the mlx_audio internals _supports_encoder_hoist probes
    # for, so the app takes the public generate() path against this fake.
    fake_model = MagicMock(spec=["generate"])

    # Key the fake off the prompt, not call order: this asserts the real
    # intent (each task's own prompt drives its own card) and stays correct
    # if task iteration order or inference count ever changes.
    def by_prompt(*_args: object, prompt: str = "", **_kwargs: object) -> MagicMock:
        return MagicMock(
            text="bonjour le monde" if "French" in prompt else "hello world"
        )

    fake_model.generate.side_effect = by_prompt
    with patch("mlx_audio.stt.utils.load_model", return_value=fake_model):
        at = _app().run()
        at.file_uploader[0].set_value(("sample_10s.wav", audio_bytes, "audio/wav"))
        at.pills[0].set_value(["Transcribe", "French"])
        at.toggle(key="use_segmentation").set_value(False)
        at.toggle(key="use_toxicity_check").set_value(False)
        at.run()
        at.button[0].click()
        at.run()
    assert not at.exception
    assert {s.value for s in at.subheader} == {"Transcribe (English)", "French"}
    # Each task's own inference lands on its own card.
    texts = " ".join(t.value for t in at.text)
    assert "hello world" in texts
    assert "bonjour le monde" in texts


def test_config_defines_no_custom_theme(app_config: dict) -> None:
    """No [theme] table, so the app gets Streamlit's built-in themes and the
    settings menu offers System / Light / Dark. See CLAUDE.md for why writing
    the defaults back in is not the same thing.

    The second half is the guard that outlives this config: should a custom
    theme ever be reintroduced, it has to define both sub-palettes or the
    appearance menu disappears entirely and the app locks to one mode
    (verified against 1.61.1 — a [theme] table carrying only primaryColor
    renders the menu with no appearance section at all)."""
    if "theme" in app_config:
        assert {"light", "dark"} <= app_config["theme"].keys()


def _option_paths(table: dict, valid: set[str]) -> Iterator[str]:
    """Flatten a parsed config.toml into the option paths it sets."""

    def walk(path: str, value: object) -> Iterator[str]:
        if path in valid:
            # A registered option, so stop rather than descend: a few options
            # take a table as their value (server.trustedUserHeaders) or a list
            # of them ([[theme.fontFaces]]), and those inner keys are free-form
            # rather than registry names.
            yield path
        elif isinstance(value, dict) and value:
            # An unregistered table with contents is a section, not a key.
            for key, child in value.items():
                yield from walk(f"{path}.{key}", child)
        else:
            # Scalar, list, or empty table under an unregistered path. The empty
            # case is what reports a typo'd section header with no keys of its
            # own, which would otherwise flatten to nothing and pass.
            yield path

    for key, value in table.items():
        yield from walk(key, value)


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        ({"server": {"maxUploadSize": 500}}, ["server.maxUploadSize"]),
        # An option whose value is a table is a leaf; its keys are free-form.
        (
            {"server": {"trustedUserHeaders": {"X-Forwarded-User": "email"}}},
            ["server.trustedUserHeaders"],
        ),
        # As is one whose value is an array of tables.
        ({"theme": {"fontFaces": [{"family": "X", "url": "u"}]}}, ["theme.fontFaces"]),
        # Typos still surface at every depth and every shape.
        ({"theme": {"typoColor": "#fff"}}, ["theme.typoColor"]),
        ({"thheme": {"fontFaces": [{"family": "X"}]}}, ["thheme.fontFaces"]),
        ({"browserr": {}}, ["browserr"]),
    ],
)
def test_option_paths_stops_at_registered_options(
    table: dict, expected: list[str]
) -> None:
    """The shipped config exercises one path of _option_paths, so the rest are
    pinned here: descending into an option's own value fails a valid config,
    and not reaching a typo under an unregistered prefix passes an invalid one.
    """
    assert list(_option_paths(table, set(config._config_options_template))) == expected


def test_config_has_no_invalid_options(app_config: dict) -> None:
    """Every key in config.toml is a registered Streamlit config option.
    Unknown keys are dropped silently rather than rejected, so a typo (or an
    option that only ever existed in a sibling table, as `base` does for
    [theme] but not [theme.light]) would otherwise go unnoticed."""
    valid = set(config._config_options_template)
    assert "theme.primaryColor" in valid  # registry is populated
    invalid = [path for path in _option_paths(app_config, valid) if path not in valid]
    assert invalid == [], f"invalid config options: {invalid}"
