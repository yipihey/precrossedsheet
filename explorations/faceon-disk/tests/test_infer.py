"""The face-on likelihood sees the scale-height ratio and not the absolute heights."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faceon_disk.infer import TRUTH, make_observation, predict  # noqa: E402
from faceon_disk.model import AttenuationGrid  # noqa: E402


class LikelihoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = AttenuationGrid(n_alpha=28, n_tau=36, n_z=500)
        cls.obs = make_observation(cls.grid, npix=28, extent=12.0, sigma=0.05, psf_kpc=0.35, seed=2)

    def test_rescaling_both_heights_keeps_the_faceon_image(self):
        theta = TRUTH.log_params()
        theta[3] += np.log(1.8)
        theta[4] += np.log(1.8)
        original = predict(TRUTH, self.grid, self.obs)
        scaled = predict(TRUTH.with_log_params(theta), self.grid, self.obs)
        np.testing.assert_allclose(original, scaled, rtol=1e-6, atol=1e-6)

    def test_wrong_dust_column_moves_the_image(self):
        theta = TRUTH.log_params()
        theta[2] = np.log(0.25)
        original = predict(TRUTH, self.grid, self.obs)
        faded = predict(TRUTH.with_log_params(theta), self.grid, self.obs)
        change = np.sqrt(np.mean((original - faded) ** 2))
        self.assertGreater(change, 0.03)
        center = original.shape[-1] // 2
        self.assertGreater(abs(original[1, center, center] - faded[1, center, center]), 0.08)


if __name__ == "__main__":
    unittest.main()
