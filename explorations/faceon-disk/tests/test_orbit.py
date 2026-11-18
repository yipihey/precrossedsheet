"""The camera circles a full turn instead of nodding in place."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faceon_disk.model import Disk, faceon_image, render  # noqa: E402
from faceon_disk.movie import ORBIT_INCLINATION, ORBIT_LIFTOFF, orbit_pose  # noqa: E402


class OrbitPoseTests(unittest.TestCase):
    def test_closed_full_turn_at_constant_inclination(self):
        self.assertEqual(orbit_pose(0.0), (0.0, 0.0))
        self.assertEqual(orbit_pose(1.0), (0.0, 360.0))
        cruise = np.linspace(ORBIT_LIFTOFF, 1.0 - ORBIT_LIFTOFF, 401)
        inclinations = []
        azimuths = []
        for t in cruise:
            inc, az = orbit_pose(float(t))
            inclinations.append(inc)
            azimuths.append(az)
        inclinations = np.array(inclinations)
        azimuths = np.array(azimuths)
        self.assertTrue(np.allclose(inclinations, ORBIT_INCLINATION))
        self.assertAlmostEqual(azimuths[0], 0.0, places=6)
        self.assertAlmostEqual(azimuths[-1], 360.0, places=6)
        self.assertGreaterEqual(np.min(np.diff(azimuths)), -1e-9)
        for quadrant in (90.0, 180.0, 270.0):
            self.assertTrue(np.any(azimuths >= quadrant))
        # Landing is face-on again, still at the end of the turn.
        late_inc, late_az = orbit_pose(0.97)
        self.assertLess(late_inc, 15.0)
        self.assertEqual(late_az, 360.0)

    def test_joins_have_no_step_in_pose(self):
        times = np.linspace(0.0, 1.0, 2001)
        poses = np.array([orbit_pose(float(t)) for t in times])
        self.assertLess(np.max(np.abs(np.diff(poses[:, 0]))), 0.5)
        # Azimuth is flat, then advances, then flat. No single step jumps by tens of degrees.
        self.assertLess(np.max(np.diff(poses[:, 1])), 2.0)


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
