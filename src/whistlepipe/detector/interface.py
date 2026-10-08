"""S5 interface (the contract every whistle detector must satisfy).

This project's S5 is a NEWLY DESIGNED REPLACEMENT. The paper's S5 detector (its reference [31]) is
unavailable and is NOT reproduced here. The replacement is not claimed to be equivalent to it.

CONTRACT
--------
Input  : ``Spectrogram`` (S4 output, unchanged). ``spec.db`` has shape (T, 1025), float32, log-magnitude in dB,
         grid = 93.75 Hz per bin x 5.333 ms per frame (spec section 4). Row t is STFT frame t
         (samples [t*1024, t*1024+2048) of the 192 kHz signal); column k is frequency k*93.75 Hz.
Output : ``DetectionMap``.
         conf        (T, F) float32, finite, values in [0, 1].
         bin_lo/bin_hi inclusive S4 bin indices of the analysed band; conf[t, j] belongs to S4 node
                     (frame t, bin bin_lo + j). Nodes outside the band are implicitly 0.
         Semantics   per-node confidence that the node lies on a whistle contour (a ridge of a whistle's
                     fundamental OR of one of its harmonics; harmonic handling is done by S6). It is NOT a
                     patch/file label, and broadband clicks should score ~0.
         Thresholding is NOT done by S5: S6 applies ``decisions.s6.peak_threshold``.
Tiling : S5 may tile internally, but the output must not depend on the tiling (tested).
Boundary: at the first/last frame and at the band edges the detector sees zero-valued (floor-level) context.
Determinism: for identical input and checkpoint the output is identical (CPU run-to-run); CPU/GPU results agree
         to float tolerance (tested).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..dsp import Spectrogram


@dataclass
class DetectionMap:
    conf: np.ndarray
    bin_lo: int
    bin_hi: int
    n_bins_total: int
    detector_name: str
    detector_version: str
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return self.conf.shape[0]

    @property
    def n_band_bins(self) -> int:
        return self.conf.shape[1]

    def to_full(self) -> np.ndarray:
        """Embed in the full S4 grid (T, n_bins_total); nodes outside the band are 0."""
        out = np.zeros((self.n_frames, self.n_bins_total), np.float32)
        out[:, self.bin_lo:self.bin_hi + 1] = self.conf
        return out


class WhistleDetector(Protocol):
    name: str
    version: str

    def predict(self, spec: Spectrogram) -> DetectionMap: ...


def check_detection_map(dm: DetectionMap, spec: Spectrogram) -> None:
    """Raise AssertionError if the S5 contract is violated."""
    assert dm.conf.ndim == 2, "conf must be 2-D (T, F)"
    assert dm.conf.shape[0] == spec.n_frames, f"time axis {dm.conf.shape[0]} != S4 frames {spec.n_frames}"
    assert dm.conf.shape[1] == dm.bin_hi - dm.bin_lo + 1, "band width mismatch"
    assert 0 <= dm.bin_lo <= dm.bin_hi < spec.n_bins, "band outside the S4 grid"
    assert dm.conf.dtype == np.float32, f"dtype {dm.conf.dtype} != float32"
    assert np.isfinite(dm.conf).all(), "non-finite confidence"
    assert dm.conf.size == 0 or (dm.conf.min() >= 0.0 and dm.conf.max() <= 1.0), "confidence outside [0, 1]"
    assert dm.n_bins_total == spec.n_bins
