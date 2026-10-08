"""Diagnostic figures (debug mode only; never used by the pipeline logic)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_overview(path: Path, spec, dm, tr, cfg, title: str = "", seconds: float = 12.0) -> None:
    """Two excerpts (file start, and the busiest stretch by track count): S4, S5 confidence, tracks over S4."""
    hop_s = spec.hop_s
    n_win = int(seconds / hop_s)
    T = spec.n_frames
    starts = [0]
    if tr.tracks and T > 2 * n_win:
        cnt = np.zeros(T)
        for t in tr.tracks:
            cnt[t.frames[0]:t.frames[-1] + 1] += 1
        csum = np.convolve(cnt, np.ones(n_win), mode="valid")
        starts.append(int(np.argmax(csum)))
    k_lo, k_hi = dm.bin_lo, dm.bin_hi
    f_ext = [k_lo * spec.bin_hz / 1e3, (k_hi + 1) * spec.bin_hz / 1e3]
    fig, ax = plt.subplots(3, len(starts), figsize=(9 * len(starts), 9), squeeze=False)
    for j, a in enumerate(starts):
        b = min(T, a + n_win)
        t0, t1 = a * hop_s, b * hop_s
        sub = spec.db[a:b, k_lo:k_hi + 1]
        ext = [t0, t1, f_ext[0], f_ext[1]]
        ax[0, j].imshow(sub.T, origin="lower", aspect="auto", extent=ext, cmap="viridis",
                        vmin=np.percentile(sub, 5), vmax=np.percentile(sub, 99.8))
        ax[0, j].set_title(f"S4 log-magnitude (band)  [{t0:.0f}-{t1:.0f} s]")
        ax[1, j].imshow(dm.conf[a:b].T, origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=0, vmax=1)
        ax[1, j].set_title(f"S5 confidence ({dm.detector_version})")
        ax[2, j].imshow(sub.T, origin="lower", aspect="auto", extent=ext, cmap="gray",
                        vmin=np.percentile(sub, 5), vmax=np.percentile(sub, 99.8))
        for t in tr.tracks:
            if t.frames[-1] < a or t.frames[0] >= b:
                continue
            m = (t.frames >= a) & (t.frames < b)
            ax[2, j].plot(t.frames[m] * hop_s + 0.5 * hop_s, t.freq_hz[m] / 1e3, lw=1.6, color=plt.cm.tab20(t.id % 20))
        ax[2, j].set_title(f"S6 tracks over S4 ({len(tr.tracks)} kept in file, {len(tr.rejected)} rejected)")
        for r in range(3):
            ax[r, j].set_ylabel("kHz")
        ax[2, j].set_xlabel("time in file (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)
