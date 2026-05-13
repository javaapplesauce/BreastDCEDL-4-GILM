"""Per-volume percentile normalization preserves cross-channel intensity
ordering. Per-slice MinMax does not — that's the bug being fixed."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessing import _percentile_uint8_volume, fuse_rgb_slice


def _synthetic_volumes(seed: int = 0):
    """Three 3D volumes representing a tumor that enhances early then
    washes out. Tumor lives in a small central cube; outside is near-zero
    'air'."""
    rng = np.random.default_rng(seed)
    shape = (32, 32, 16)
    air_noise = 1e-6
    def make(intensity):
        v = rng.normal(0.0, air_noise, size=shape).astype(np.float32)
        v[10:20, 10:20, 6:10] = intensity + rng.normal(0, 0.05, size=(10, 10, 4))
        return v
    return make(0.2), make(1.0), make(0.6)


def test_percentile_preserves_channel_order_at_tumor():
    pre, early, late = _synthetic_volumes()
    r, g, b = _percentile_uint8_volume([pre, early, late], 1.0, 99.0)
    tum = (slice(10, 20), slice(10, 20), slice(6, 10))
    r_mu, g_mu, b_mu = r[tum].mean(), g[tum].mean(), b[tum].mean()
    assert g_mu > b_mu > r_mu, (
        f"channel order broken at tumor: R={r_mu:.1f} G={g_mu:.1f} B={b_mu:.1f}"
    )


def test_percentile_keeps_air_near_zero():
    pre, early, late = _synthetic_volumes()
    r, g, b = _percentile_uint8_volume([pre, early, late], 1.0, 99.0)
    air = (slice(0, 5), slice(0, 5), 0)
    assert r[air].mean() < 5
    assert g[air].mean() < 5
    assert b[air].mean() < 5


def test_per_slice_minmax_destroys_cross_channel_magnitude():
    """The bug being fixed. fuse_rgb_slice rescales each (pre, early, late)
    independently to [0, 255], so the cross-channel magnitude ordering
    that the model is supposed to learn from is gone: a slice where
    early=10*pre and a slice where early=pre produce the same per-channel
    means (~127 each)."""
    rng = np.random.default_rng(1)
    shape2d = (32, 32)
    base = rng.uniform(0.1, 0.5, size=shape2d).astype(np.float32)

    # Strongly-enhancing slice: green channel 10x brighter than red.
    enhancing = fuse_rgb_slice(base * 0.1, base * 1.0, base * 0.5)
    # Non-enhancing slice: all three channels identical.
    flat = fuse_rgb_slice(base, base, base)

    enh_means = enhancing.reshape(-1, 3).mean(0)
    flat_means = flat.reshape(-1, 3).mean(0)
    # Per-slice MinMax collapses both to roughly the same per-channel means.
    diff = np.abs(enh_means - flat_means).max()
    assert diff < 5, (
        f"per-slice MinMax should make enhancing and flat slices look "
        f"identical per-channel; got max channel diff = {diff:.1f}"
    )

    # And percentile normalization should NOT collapse them.
    pre, early, late = base * 0.1, base * 1.0, base * 0.5
    base_flat = base.copy()
    r1, g1, b1 = _percentile_uint8_volume(
        [pre[..., None], early[..., None], late[..., None]], 1.0, 99.0,
    )
    r2, g2, b2 = _percentile_uint8_volume(
        [base_flat[..., None], base_flat[..., None], base_flat[..., None]],
        1.0, 99.0,
    )
    enh_means_v = np.array([r1.mean(), g1.mean(), b1.mean()])
    flat_means_v = np.array([r2.mean(), g2.mean(), b2.mean()])
    diff_v = np.abs(enh_means_v - flat_means_v).max()
    assert diff_v > 30, (
        f"percentile normalization should preserve enhancement magnitude; "
        f"got max channel diff = {diff_v:.1f}"
    )


def test_percentile_handles_all_zero_volumes():
    z = np.zeros((4, 4, 2), dtype=np.float32)
    out = _percentile_uint8_volume([z, z, z], 1.0, 99.0)
    for o in out:
        assert o.dtype == np.uint8
        assert o.sum() == 0


def test_percentile_handles_degenerate_lo_hi():
    a = np.ones((4, 4, 2), dtype=np.float32) * 0.5
    out = _percentile_uint8_volume([a, a, a], 1.0, 99.0)
    for o in out:
        assert o.sum() == 0  # hi - lo < eps -> zero output
