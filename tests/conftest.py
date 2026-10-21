import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from whistlepipe.config import PROJECT_ROOT, load_detector_config, load_pipeline_config  # noqa: E402

CHANNEL_TONES_HZ = [11000.0, 13000.0, 17000.0, 19000.0, 23000.0, 29000.0]


@pytest.fixture(scope="session")
def base_cfg():
    return load_pipeline_config()


@pytest.fixture()
def cfg(base_cfg):
    return type(base_cfg)(copy.deepcopy(dict(base_cfg)))


@pytest.fixture(scope="session")
def det_cfg():
    return load_detector_config()


@pytest.fixture()
def fake_raw(tmp_path, cfg):
    """A tiny 6-channel 500 kHz FLAC (1 s) with a different tone on each channel, in a temp 'raw' dir."""
    raw = tmp_path / "raw"
    raw.mkdir()
    fs, n = 500000, 500000
    t = np.arange(n) / fs
    x = np.stack([0.2 * np.sin(2 * np.pi * f * t) for f in CHANNEL_TONES_HZ], axis=1)
    path = raw / "1705_20170101_000000_000.flac"
    sf.write(str(path), (x * 32767).astype(np.int16), fs, subtype="PCM_16", format="FLAC")
    cfg["paths"]["raw_dirs"] = [str(raw)]
    cfg["paths"]["synth_raw_dir"] = str(raw)
    cfg["paths"]["runs_dir"] = str(tmp_path / "runs")
    cfg["paths"]["dataset_dir"] = str(tmp_path / "preprocessed")
    return path


def real_first_file():
    d = PROJECT_ROOT / "encounter_191_audio"
    files = sorted(d.glob("*.flac"))
    return files[0] if files else None
