"""Spherical clouds: the face-on column matches, and an inclined knot stays round."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faceon_disk.model import convolve_gaussian  # noqa: E402
from faceon_disk.splats import GaussianCloud, fit_cloud, prepare_observed, render_cloud, tau_from_image  # noqa: E402


def _axis_ratio(image: np.ndarray) -> float:
    """Major/minor second-moment ratio of a single blob. A circle is 1."""
    weights = np.clip(image, 0.0, None)
    total = float(weights.sum())
    yy, xx = np.indices(weights.shape)
    cx = float((weights * xx).sum() / total)
    cy = float((weights * yy).sum() / total)
    var_x = float((weights * (xx - cx) ** 2).sum() / total)
    var_y = float((weights * (yy - cy) ** 2).sum() / total)
    return max(var_x, var_y) / min(var_x, var_y)


class SplatTests(unittest.TestCase):
    def test_faceon_integral_equals_luminosity(self):
        cloud = GaussianCloud(
            star_mu=np.array([[0.0, 0.0, 0.0]]),
            star_sigma=np.array([0.40]),
            star_luminosity=np.array([[0.0, 2.5, 0.0]]),
            dust_mu=np.zeros((0, 3)),
            dust_sigma=np.zeros(0),
            dust_tau=np.zeros(0),
        )
        extent = 6.0
        npix = 96
        image = render_cloud(cloud, 0.0, 0.0, npix, extent, extinction=np.ones(3), n_slices=1)
        area = (2.0 * extent / npix) ** 2
        self.assertAlmostEqual(float(image[1].sum() * area), 2.5, delta=0.05)
        self.assertLess(float(image[0].sum()), 1e-6)
        self.assertLess(float(image[2].sum()), 1e-6)

    def test_foreground_dust_absorbs_and_background_dust_does_not(self):
        star = dict(
            star_mu=np.array([[0.0, 0.0, 0.0]]),
            star_sigma=np.array([0.30]),
            star_luminosity=np.array([[0.0, 1.0, 0.0]]),
        )
        # τ at the dust center is tau_column / (2πσ²) ≈ 1.5.
        tau_column = 1.5 * 2.0 * np.pi * 0.45**2
        bare = render_cloud(
            GaussianCloud(dust_mu=np.zeros((0, 3)), dust_sigma=np.zeros(0), dust_tau=np.zeros(0), **star),
            0.0,
            0.0,
            80,
            5.0,
            extinction=np.ones(3),
        )
        front = render_cloud(
            GaussianCloud(
                dust_mu=np.array([[0.0, 0.0, 2.0]]),
                dust_sigma=np.array([0.45]),
                dust_tau=np.array([tau_column]),
                **star,
            ),
            0.0,
            0.0,
            80,
            5.0,
            extinction=np.ones(3),
            n_slices=8,
        )
        behind = render_cloud(
            GaussianCloud(
                dust_mu=np.array([[0.0, 0.0, -2.0]]),
                dust_sigma=np.array([0.45]),
                dust_tau=np.array([tau_column]),
                **star,
            ),
            0.0,
            0.0,
            80,
            5.0,
            extinction=np.ones(3),
            n_slices=8,
        )
        center = (40, 40)
        bare_c = float(bare[1][center])
        front_c = float(front[1][center])
        behind_c = float(behind[1][center])
        self.assertLess(front_c, 0.45 * bare_c)
        self.assertAlmostEqual(behind_c, bare_c, delta=0.08 * bare_c)

    def test_inclined_knot_stays_round(self):
        cloud = GaussianCloud(
            star_mu=np.array([[0.2, -0.4, 0.0]]),
            star_sigma=np.array([0.45]),
            star_luminosity=np.array([[0.0, 1.0, 0.0]]),
            dust_mu=np.zeros((0, 3)),
            dust_sigma=np.zeros(0),
            dust_tau=np.zeros(0),
        )
        face = render_cloud(cloud, 0.0, 0.0, 80, 5.0, extinction=np.ones(3), n_slices=1)
        inclined = render_cloud(cloud, 75.0, 40.0, 80, 5.0, extinction=np.ones(3), n_slices=1)
        self.assertLess(_axis_ratio(face[1]), 1.15)
        self.assertLess(_axis_ratio(inclined[1]), 1.15)

    def test_fit_keeps_a_blob_on_its_side_of_the_image(self):
        image = np.zeros((3, 64, 64))
        image[:, 30, 48] = 1.0
        image[0] *= 0.7
        image[2] *= 1.4
        # Pixel is 0.125 kpc. The blob is a size the cloud scales can represent.
        image = np.stack([convolve_gaussian(image[band], 0.40 / 0.125) for band in range(3)])
        cloud = fit_cloud(
            image,
            extent=4.0,
            tau_v=np.zeros((64, 64)),
            extinction=np.ones(3),
            star_sigmas=(0.22, 0.50),
            dust_sigmas=(0.8,),
        )
        weights = cloud.star_luminosity[:, 1]
        self.assertLess(float(np.average(cloud.star_mu[:, 0], weights=weights)), -0.4)
        rendered = render_cloud(cloud, 0.0, 0.0, 64, 4.0, extinction=np.ones(3), n_slices=1)
        self.assertGreater(float(rendered[1, :, 32:].sum()), float(rendered[1, :, :32].sum()) * 2.0)
        correlation = np.corrcoef(image[1].ravel(), rendered[1].ravel())[0, 1]
        self.assertGreater(float(correlation), 0.95)
        self.assertLess(abs(float(rendered[1].sum() / image[1].sum()) - 1.0), 0.25)
        # Local color survives: the blob is redder in z than in g.
        red = rendered[2, 28:36, 44:52].sum()
        blue = rendered[0, 28:36, 44:52].sum()
        self.assertGreater(float(red), float(blue) * 1.3)

    def test_saturated_star_is_replaced_by_the_surrounding_disk(self):
        image = np.full((3, 40, 40), 0.25)
        image[0, 10, 30] = 5.0
        image[1, 10, 30] = 40.0
        image[2, 10, 30] = 20.0
        mask = np.zeros((40, 40), dtype=bool)
        mask[9:12, 29:32] = True
        filled = prepare_observed(image, mask)
        self.assertLess(float(filled[:, 10, 30].max()), 1.0)
        self.assertGreater(float(filled[1, 10, 30]), 0.05)

    def test_bright_knot_is_carried_by_one_small_cloud(self):
        extent = 4.0
        npix = 48
        image = np.full((3, npix, npix), 0.22)
        image[0] *= 0.75
        image[2] *= 1.25
        image[0, 20, 30] += 0.45
        image[1, 20, 30] += 1.7
        image[2, 20, 30] += 0.55
        cloud = fit_cloud(
            image,
            extent,
            tau_v=np.zeros((npix, npix)),
            extinction=np.ones(3),
            star_sigmas=(0.16, 0.36, 0.8),
            dust_sigmas=(0.9,),
        )
        pix = 2.0 * extent / npix
        sky_x = -extent + (30.5) / npix * 2.0 * extent
        sky_y = extent - (20.5) / npix * 2.0 * extent
        gx, gy = -sky_x, -sky_y
        dist = np.hypot(cloud.star_mu[:, 0] - gx, cloud.star_mu[:, 1] - gy)
        small = (cloud.star_sigma <= 0.13) & (dist < 0.30)
        self.assertEqual(int(small.sum()), 1)
        nearby_fine = (cloud.star_sigma < 0.25) & (dist < 0.22)
        self.assertLessEqual(int(nearby_fine.sum()), 2)
        excess_light = 1.7 * pix * pix
        self.assertGreater(float(cloud.star_luminosity[small, 1].sum()), 0.45 * excess_light)
        rendered = render_cloud(cloud, 0.0, 0.0, npix, extent, extinction=np.ones(3), n_slices=1)
        self.assertGreater(float(rendered[1, 20, 30]), 1.2)
        self.assertLess(float(rendered[1, 20, 30]), 2.6)

    def test_dust_follows_the_red_lane_and_the_blue_knot_stays_blue(self):
        extent = 6.0
        npix = 72
        yy, xx = np.mgrid[:npix, :npix]
        radius = np.hypot(yy - 36, xx - 36)
        disk = np.exp(-((radius / 16.0) ** 2))
        image = np.stack([0.85 * disk, disk, 1.15 * disk])
        lane = (np.abs(xx - 46) <= 1) & (radius < 22)
        image[0, lane] *= 0.40
        image[1, lane] *= 0.62
        image[2, lane] *= 0.85
        image[0, 28, 24] += 0.9
        image[1, 28, 24] += 0.45
        image[2, 28, 24] += 0.25
        extinction = np.array([1.18, 0.84, 0.46])
        tau = tau_from_image(image, extinction, extent)
        self.assertGreater(float(tau[36, 46]), float(tau[36, 26]) + 0.05)
        self.assertLess(float(tau[28, 24]), 0.05)
        cloud = fit_cloud(
            image,
            extent,
            tau_v=np.zeros((npix, npix)),
            extinction=extinction,
            star_sigmas=(0.18, 0.40),
            dust_sigmas=(0.16, 0.35),
            cluster_excess=5.0,
        )
        self.assertGreater(len(cloud.dust_sigma), 0)
        sky_x = -extent + (np.arange(npix) + 0.5) / npix * 2.0 * extent
        lane_x = sky_x[46]
        dust_x = -cloud.dust_mu[:, 0]
        self.assertGreater(float(np.average(np.exp(-((dust_x - lane_x) ** 2)))), 0.15)
        rendered = render_cloud(cloud, 0.0, 0.0, npix, extent, extinction=extinction, n_slices=1)
        blue = rendered[0, 28, 24] / max(rendered[1, 28, 24], 1e-4)
        red_lane = rendered[0, 36, 46] / max(rendered[1, 36, 46], 1e-4)
        self.assertGreater(blue, red_lane * 1.3)
