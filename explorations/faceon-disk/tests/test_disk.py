"""Physics checks for the exponential star-plus-dust disk."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faceon_disk.model import (  # noqa: E402
    EXTINCTION,
    Disk,
    ccm_extinction,
    column_factors,
    faceon_image,
    render,
)


class ExtinctionTests(unittest.TestCase):
    def test_v_band_is_unity(self):
        self.assertAlmostEqual(ccm_extinction(549.5), 1.0, places=2)

    def test_bands_get_bluer_extinction(self):
        self.assertGreater(EXTINCTION[0], EXTINCTION[1])
        self.assertGreater(EXTINCTION[1], EXTINCTION[2])
        self.assertGreater(EXTINCTION[0], 1.0)
        self.assertLess(EXTINCTION[2], 1.0)


class ColumnTests(unittest.TestCase):
    def test_optically_thin_transmits_everything(self):
        factor = column_factors(np.array([0.0, 1e-8]), alpha=2.0)
        np.testing.assert_allclose(factor, 1.0, atol=1e-5)

    def test_screen_limit(self):
        # Dust much thinner than the stars: half the light is in front of the sheet.
        tau = np.array([0.5, 1.5, 3.0])
        factor = column_factors(tau, alpha=80.0, n_z=4000, steep=5.0)
        expected = 0.5 * (1.0 + np.exp(-tau))
        np.testing.assert_allclose(factor, expected, rtol=0.03, atol=0.01)

    def test_more_dust_or_thicker_dust_attenuates_more(self):
        thin_dust = column_factors(np.array([2.0]), alpha=8.0)[0]
        mixed = column_factors(np.array([2.0]), alpha=1.0)[0]
        heavier = column_factors(np.array([4.0]), alpha=1.0)[0]
        self.assertGreater(thin_dust, mixed)
        self.assertGreater(mixed, heavier)
        self.assertGreater(mixed, np.exp(-2.0))
        self.assertLess(mixed, 1.0)


class FaceOnTests(unittest.TestCase):
    def test_central_brightness_and_total_flux_without_dust(self):
        disk = Disk(h_R=3.0, I0=1.4, tau_V0=0.0, h_star=0.4, h_dust=0.2, spiral_amplitude=0.0)
        image = faceon_image(disk, npix=80, extent=18.0)
        # Pixel centers sit half a pixel off the origin, so compare with the exponential there.
        extent = 18.0
        offset = 0.5 / 80.0 * 2.0 * extent
        radius = np.hypot(offset, offset)
        np.testing.assert_allclose(image[:, 39, 39], disk.I0 * np.exp(-radius / disk.h_R), rtol=0.01)
        pixel_area = (36.0 / 80.0) ** 2
        total = image[1].sum() * pixel_area
        # ∫ I0 exp(-R/h) 2π R dR = 2π I0 h^2. The frame cuts the exp(-6) tail.
        self.assertAlmostEqual(total, 2.0 * np.pi * disk.I0 * disk.h_R**2, delta=0.04 * total)

    def test_faceon_independent_of_shared_height_rescaling(self):
        base = Disk(spiral_amplitude=0.35, tau_V0=2.0, h_star=0.30, h_dust=0.12)
        scaled = Disk(
            h_R=base.h_R,
            I0=base.I0,
            tau_V0=base.tau_V0,
            h_star=2.0 * base.h_star,
            h_dust=2.0 * base.h_dust,
            spiral_amplitude=base.spiral_amplitude,
            pitch_deg=base.pitch_deg,
        )
        a = faceon_image(base, npix=48, extent=12.0)
        b = faceon_image(scaled, npix=48, extent=12.0)
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_dust_reddens_the_center(self):
        dusty = Disk(spiral_amplitude=0.0, tau_V0=3.0, h_star=0.4, h_dust=0.15)
        clear = Disk(spiral_amplitude=0.0, tau_V0=0.0, h_star=0.4, h_dust=0.15, I0=dusty.I0, h_R=dusty.h_R)
        image = faceon_image(dusty, npix=32, extent=8.0)
        bare = faceon_image(clear, npix=32, extent=8.0)
        cy = cx = 16
        dusty_ratio = image[2, cy, cx] / image[0, cy, cx]
        bare_ratio = bare[2, cy, cx] / bare[0, cy, cx]
        self.assertGreater(dusty_ratio, bare_ratio * 1.2)
        self.assertLess(image[1, cy, cx], bare[1, cy, cx])


class RenderTests(unittest.TestCase):
    def test_faceon_ray_matches_column(self):
        disk = Disk(spiral_amplitude=0.3, tau_V0=1.8, h_star=0.36, h_dust=0.14)
        column = faceon_image(disk, npix=28, extent=10.0)
        marched = render(disk, 0.0, 0.0, npix_x=28, extent_x=10.0, steps=280)
        scale = np.max(column)
        self.assertLess(np.max(np.abs(marched - column)) / scale, 0.03)

    def test_height_rescaling_is_invisible_faceon_and_visible_inclined(self):
        base = Disk(spiral_amplitude=0.0, tau_V0=2.2, h_star=0.28, h_dust=0.10, h_R=3.0)
        thick = Disk(
            h_R=base.h_R,
            I0=base.I0,
            tau_V0=base.tau_V0,
            h_star=2.2 * base.h_star,
            h_dust=2.2 * base.h_dust,
            spiral_amplitude=0.0,
        )
        face_a = render(base, 0.0, npix_x=24, extent_x=10.0, steps=200)
        face_b = render(thick, 0.0, npix_x=24, extent_x=10.0, steps=200)
        np.testing.assert_allclose(face_a, face_b, rtol=0.03, atol=0.01 * base.I0)
        edge_a = render(base, 80.0, npix_x=36, extent_x=10.0, steps=260)
        edge_b = render(thick, 80.0, npix_x=36, extent_x=10.0, steps=260)
        rms = np.sqrt(np.mean((edge_a - edge_b) ** 2))
        self.assertGreater(rms, 0.02 * base.I0)

    def test_thin_disk_conserves_flux_when_inclined(self):
        disk = Disk(tau_V0=0.0, spiral_amplitude=0.0, h_star=0.35, h_dust=0.15, h_R=2.5, I0=1.0)
        extent = 16.0
        npix = 40
        area = (2.0 * extent / npix) ** 2
        face = render(disk, 0.0, npix_x=npix, extent_x=extent, steps=180)
        inclined = render(disk, 55.0, azimuth_deg=20.0, npix_x=npix, extent_x=extent, steps=180)
        self.assertAlmostEqual(face[0].sum() * area, inclined[0].sum() * area, delta=0.05 * face[0].sum() * area)

    def test_inclination_increases_extinction(self):
        disk = Disk(tau_V0=1.5, spiral_amplitude=0.0, h_star=0.4, h_dust=0.16)
        extent = 14.0
        npix = 32
        area = (2.0 * extent / npix) ** 2
        face = render(disk, 0.0, npix_x=npix, extent_x=extent, steps=200)[1].sum() * area
        edge = render(disk, 70.0, npix_x=npix, extent_x=extent, steps=200)[1].sum() * area
        self.assertLess(edge, 0.9 * face)


if __name__ == "__main__":
    unittest.main()
