"""The camera tips through a full turn instead of spinning in azimuth."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faceon_disk.model import Disk, camera_basis, faceon_image, render  # noqa: E402
from faceon_disk.movie import orbit_pose  # noqa: E402


class OrbitPoseTests(unittest.TestCase):
    def test_tilt_runs_through_a_full_turn_at_fixed_azimuth(self):
        self.assertEqual(orbit_pose(0.0), (0.0, 0.0))
        end_inc, end_az = orbit_pose(1.0)
        self.assertAlmostEqual(end_inc, 360.0, places=6)
        self.assertEqual(end_az, 0.0)
        times = np.linspace(0.0, 1.0, 721)
        inclinations = []
        azimuths = []
        for t in times:
            inc, az = orbit_pose(float(t))
            inclinations.append(inc)
            azimuths.append(az)
        inclinations = np.array(inclinations)
        azimuths = np.array(azimuths)
        self.assertTrue(np.allclose(azimuths, 0.0))
        self.assertGreaterEqual(np.min(np.diff(inclinations)), -1e-9)
        for angle in (90.0, 180.0, 270.0):
            self.assertTrue(np.any(inclinations >= angle))
        # The ends ease in, so a step near the middle is larger than one at the start.
        self.assertLess(inclinations[1] - inclinations[0], 0.05)
        self.assertGreater(np.max(np.diff(inclinations)), 0.4)

    def test_joins_have_no_step_in_pose(self):
        times = np.linspace(0.0, 1.0, 2001)
        poses = np.array([orbit_pose(float(t)) for t in times])
        self.assertLess(np.max(np.abs(np.diff(poses[:, 0]))), 0.5)
        self.assertTrue(np.allclose(poses[:, 1], 0.0))

    def test_tilt_through_the_far_side_does_not_flip(self):
        knot = np.array([1.5, 4.0, 0.3])
        angles = np.linspace(0.0, 360.0, 73)
        sky = []
        for inc in angles:
            e_x, e_y, _ = camera_basis(float(inc), 0.0)
            sky.append((float(knot @ e_x), float(knot @ e_y)))
        sky = np.array(sky)
        step = np.hypot(*np.diff(sky, axis=0).T)
        self.assertLess(np.max(step), 0.4)
        np.testing.assert_allclose(sky[0], sky[-1], atol=1e-8)
        # Edge-on, the in-plane offset is across the sky and the height is vertical.
        e_x, e_y, n_obs = camera_basis(90.0, 0.0)
        self.assertAlmostEqual(float(knot @ e_y), knot[2], places=6)
        self.assertGreater(abs(float(knot @ e_x)), 1.0)
        self.assertAlmostEqual(abs(n_obs[2]), 0.0, places=6)


class AzimuthVisibilityTests(unittest.TestCase):
    def test_axisymmetric_disk_does_not_show_the_orbit(self):
        disk = Disk(spiral_amplitude=0.0, tau_V0=1.2, h_star=0.4, h_dust=0.16)
        east = render(disk, 60.0, 0.0, npix_x=36, extent_x=10.0, steps=80)
        north = render(disk, 60.0, 90.0, npix_x=36, extent_x=10.0, steps=80)
        scale = max(float(np.max(east)), 1e-6)
        self.assertLess(np.max(np.abs(east - north)) / scale, 0.03)

    def test_offset_pattern_swings_around_the_disk(self):
        npix = 48
        extent = 10.0
        pattern = np.ones((npix, npix), dtype=np.float64)
        pattern[:, npix // 2 + 8 :] += 1.4
        disk = Disk(
            spiral_amplitude=0.0,
            tau_V0=0.0,
            I0=1.0,
            h_R=8.0,
            pattern_map=pattern,
            pattern_extent=extent,
        )
        face = faceon_image(disk, npix=npix, extent=extent, n_z=200)
        np.testing.assert_allclose(face[1], pattern, rtol=0.02, atol=0.02)
        views = [render(disk, 58.0, az, npix_x=40, extent_x=extent, steps=70)[1] for az in (0.0, 90.0, 180.0)]
        angles = [_bright_side(view) for view in views]
        # A full circuit moves the bright side by about half a turn between the first and third view.
        turn = (angles[2] - angles[0] + np.pi) % (2 * np.pi) - np.pi
        self.assertGreater(abs(turn), 0.6 * np.pi)
        self.assertGreater(np.sqrt(np.mean((views[0] - views[1]) ** 2)), 0.05)


def _bright_side(image: np.ndarray) -> float:
    weight = np.clip(image - np.median(image), 0.0, None)
    yy, xx = np.indices(image.shape)
    cy = np.sum(yy * weight) / np.sum(weight)
    cx = np.sum(xx * weight) / np.sum(weight)
    return float(np.arctan2(image.shape[0] / 2.0 - cy, cx - image.shape[1] / 2.0))


if __name__ == "__main__":
    unittest.main()
