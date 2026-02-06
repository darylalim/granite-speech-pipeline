from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from streamlit_app import (
    SUPPORTED_FORMATS,
    get_device,
    load_and_preprocess_audio,
)

AUDIO_DIR = Path(__file__).parent / "data" / "audio"


class TestGetDevice:
    @patch("streamlit_app.torch")
    def test_mps_preferred(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = True
        mock_torch.cuda.is_available.return_value = True
        assert get_device() == "mps"

    @patch("streamlit_app.torch")
    def test_cuda_fallback(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = True
        assert get_device() == "cuda"

    @patch("streamlit_app.torch")
    def test_cpu_fallback(self, mock_torch: MagicMock) -> None:
        mock_torch.backends.mps.is_available.return_value = False
        mock_torch.cuda.is_available.return_value = False
        assert get_device() == "cpu"


class TestSupportedFormats:
    def test_all_formats_present(self) -> None:
        expected = {"wav", "mp3", "m4a", "ogg", "flac", "webm", "aac"}
        assert set(SUPPORTED_FORMATS) == expected


class TestLoadAndPreprocessAudio:
    def _make_upload(self, path: Path) -> MagicMock:
        upload = MagicMock()
        upload.name = path.name
        upload.getvalue.return_value = path.read_bytes()
        return upload

    def test_loads_mp3(self) -> None:
        audio_file = self._make_upload(AUDIO_DIR / "sample_10s.mp3")
        wav, duration = load_and_preprocess_audio(audio_file)
        assert isinstance(wav, torch.Tensor)
        assert wav.shape[0] == 1
        assert duration > 0

    def test_resamples_to_16khz(self) -> None:
        audio_file = self._make_upload(AUDIO_DIR / "sample_10s.mp3")
        wav, _ = load_and_preprocess_audio(audio_file)
        assert wav.shape[1] > 0

    def test_invalid_audio_raises_runtime_error(self) -> None:
        upload = MagicMock()
        upload.name = "bad.wav"
        upload.getvalue.return_value = b"not audio data"
        with pytest.raises(RuntimeError, match="Failed to load audio file"):
            load_and_preprocess_audio(upload)
