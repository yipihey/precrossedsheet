"""The NGC 628 cutout and the small FITS reader."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faceon_disk.model import ccm_extinction  # noqa: E402
from faceon_disk.observe import (  # noqa: E402
    DATA_PATH,
    DECAM_EXTINCTION,
    foreground_factor,
    load_ngc628,
    read_fits_image,
    _star_mask,
    _radius_kpc,
)


class ExtinctionCurveTests(unittest.TestCase):
    def test_decam_z_is_redder_than_r(self):
        g, r, z = (ccm_extinction(w) for w in (480.0, 642.0, 917.0))
        self.assertGreater(g, r)
        self.assertGreater(r, z)
        self.assertGreater(z, 0.3)
        self.assertLess(z, 0.8)
        np.testing.assert_allclose((g, r, z), DECAM_EXTINCTION)

    def test_foreground_screen_dims_g_more_than_z(self):
        factor = foreground_factor(DECAM_EXTINCTION)
        self.assertGreater(factor[0], factor[2])
        self.assertTrue(np.all(factor > 1.0))


class FitsReaderTests(unittest.TestCase):
    def test_roundtrip_of_a_tiny_image(self):
        path = Path(__file__).resolve().parents[1] / "output" / "_tiny.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        _write_fits(path, image)
        header, got = read_fits_image(path)
        np.testing.assert_allclose(got, image.astype(np.float64))
        self.assertAlmostEqual(header["NAXIS1"], 4.0)
        path.unlink()


class StarMaskTests(unittest.TestCase):
    def test_spike_is_masked_and_the_disk_is_not(self):
        ny = nx = 80
        radius = _radius_kpc(ny, nx, 0.5)
        image = 0.25 * np.exp(-radius / 12.0)
        image[8, 70] += 3.0
        mask = _star_mask(image, radius)
        self.assertTrue(mask[8, 70])
        self.assertFalse(mask[40, 40])


class Ngc628Tests(unittest.TestCase):
    def test_cutout_is_centered_and_masked(self):
        self.assertTrue(DATA_PATH.exists())
        galaxy = load_ngc628()
        obs = galaxy.observation
        self.assertEqual(obs.bands, ("g", "r", "z"))
        self.assertEqual(obs.data.shape[0], 3)
        self.assertGreater(obs.extent, 12.0)
        self.assertLess(obs.extent, 20.0)
        self.assertGreater(int(obs.mask.sum()), 1000)
        self.assertFalse(obs.mask[obs.npix // 2, obs.npix // 2])
        center = galaxy.display[1, 180:220, 180:220].mean()
        corner = galaxy.display[1, :30, :30].mean()
        self.assertGreater(center, corner * 5)


def _write_fits(path: Path, image: np.ndarray) -> None:
    axes = image.shape[::-1]
    cards = [
        "SIMPLE  =                    T",
        f"BITPIX  =                  -32",
        f"NAXIS   =                    {image.ndim}",
    ]
    for i, n in enumerate(axes, start=1):
        cards.append(f"NAXIS{i:<3d}= {n:20d}")
    cards.append("END")
    header = "".join(card.ljust(80) for card in cards)
    header += " " * ((-len(header)) % 2880)
    payload = np.ascontiguousarray(image.astype(">f4")).tobytes()
    path.write_bytes(header.encode("ascii") + payload)


if __name__ == "__main__":
    unittest.main()
