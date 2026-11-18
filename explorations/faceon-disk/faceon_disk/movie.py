"""Two tilts of one galaxy, then a path through other draws.

Each pass continues the opening tilt through a full 360° of inclination, with
the line of nodes held fixed. The disk goes edge-on, through the far side, and
back to the same face-on view. The synthetic movie walks exponential scale
heights. The observed movie walks spherical clouds through other line-of-sight
arrangements. Face-on frames of those arrangements agree; the inclined frames
show the depth.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .infer import TRUTH, blend_disks
from .model import Disk, convolve_gaussian, faceon_image, render, tonemap
from .splats import GaussianCloud, blend_depth, depth_route, render_cloud

WIDTH = 1280
HEIGHT = 800
BG = (8, 10, 16)
INK = (236, 238, 244)
MUTED = (158, 166, 180)
GOLD = (232, 190, 118)
PRIOR_COLOR = (88, 104, 128)
LINE = (42, 48, 62)


TILT_EASE = 0.08


def _tilt_fraction(t: float, edge: float = TILT_EASE) -> float:
    """Map t in [0, 1] onto a full turn. Speed is zero at the ends and constant between."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    speed = 1.0 / (1.0 - edge)

    def ramp(s: float) -> float:
        s = float(np.clip(s, 0.0, 1.0))
        return s**3 - 0.5 * s**4

    if t < edge:
        return speed * edge * ramp(t / edge)
    if t > 1.0 - edge:
        return 1.0 - speed * edge * ramp((1.0 - t) / edge)
    return speed * (t - 0.5 * edge)


def orbit_pose(t: float) -> tuple[float, float]:
    """Camera inclination and azimuth for one closed tilt. t is in [0, 1].

    Azimuth stays on the opening line of nodes. Inclination advances a full
    360°, so the disk tips through edge-on and returns to the same face-on view.
    """
    return 360.0 * _tilt_fraction(t), 0.0


def zoom_extents(inclination_deg: float) -> tuple[float, float]:
    cosine = np.cos(np.deg2rad(inclination_deg))
    extent_x = 8.0
    extent_y = 3.8 + (extent_x - 3.8) * cosine**2
    return extent_x, float(extent_y)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _paste_rgb(canvas: Image.Image, rgb: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    image = Image.fromarray(rgb, mode="RGB")
    image = image.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    canvas.paste(image, (x0, y0))


def _scatter(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    prior: np.ndarray,
    posterior: np.ndarray,
    current: tuple[float, float] | None,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, fill=(14, 16, 24), outline=LINE)
    pad_l, pad_r, pad_t, pad_b = 48, 16, 28, 36
    left, right = x0 + pad_l, x1 - pad_r
    top, bottom = y0 + pad_t, y1 - pad_b
    hx = (0.08, 1.35)
    hy = (0.03, 0.85)

    def xy(h_star: float, h_dust: float) -> tuple[float, float]:
        px = left + (np.log(h_star) - np.log(hx[0])) / (np.log(hx[1]) - np.log(hx[0])) * (right - left)
        py = bottom - (np.log(h_dust) - np.log(hy[0])) / (np.log(hy[1]) - np.log(hy[0])) * (bottom - top)
        return float(px), float(py)

    draw.text((x0 + 14, y0 + 8), "Scale heights", font=_font(16, bold=True), fill=INK)
    for cloud, color, radius in ((prior, PRIOR_COLOR, 2), (posterior, GOLD, 3)):
        for h_star, h_dust in cloud[:: max(len(cloud) // 280, 1)]:
            if not (hx[0] < h_star < hx[1] and hy[0] < h_dust < hy[1]):
                continue
            px, py = xy(h_star, h_dust)
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
    if current is not None:
        px, py = xy(*current)
        r = 7
        draw.ellipse((px - r, py - r, px + r, py + r), outline=(255, 255, 255), width=2)
    draw.text((left, y1 - 28), "h★", font=_font(14), fill=MUTED)
    draw.text((x0 + 8, top), "h dust", font=_font(14), fill=MUTED)


def _depth_scatter(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    prior: np.ndarray,
    current: np.ndarray | None,
) -> None:
    """σ against |z|. Coplanar clouds sit on the floor; the prior spreads above them."""
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=12, fill=(14, 16, 24), outline=LINE)
    pad_l, pad_r, pad_t, pad_b = 48, 16, 28, 36
    left, right = x0 + pad_l, x1 - pad_r
    top, bottom = y0 + pad_t, y1 - pad_b
    hx = (0.08, 4.5)
    hy = (0.008, 3.0)

    def xy(sigma: float, z_abs: float) -> tuple[float, float]:
        sigma = float(np.clip(sigma, hx[0] * 1.04, hx[1] / 1.04))
        z_abs = float(np.clip(z_abs, hy[0] * 1.2, hy[1] / 1.04))
        px = left + (np.log(sigma) - np.log(hx[0])) / (np.log(hx[1]) - np.log(hx[0])) * (right - left)
        py = bottom - (np.log(z_abs) - np.log(hy[0])) / (np.log(hy[1]) - np.log(hy[0])) * (bottom - top)
        return float(px), float(py)

    draw.text((x0 + 14, y0 + 8), "Cloud depth", font=_font(16, bold=True), fill=INK)
    if len(prior):
        step = max(len(prior) // 320, 1)
        for sigma, z_abs in prior[::step]:
            px, py = xy(float(sigma), float(z_abs))
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=PRIOR_COLOR)
    if current is not None and len(current):
        step = max(len(current) // 220, 1)
        for sigma, z_abs in current[::step]:
            px, py = xy(float(sigma), float(z_abs))
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=GOLD)
    draw.text((left, y1 - 28), "σ", font=_font(14), fill=MUTED)
    draw.text((x0 + 8, top), "|z|", font=_font(14), fill=MUTED)


def compose_frame(
    main_rgb: np.ndarray,
    zoom_rgb: np.ndarray,
    *,
    phase: str,
    detail: str,
    inclination: float,
    azimuth: float | None,
    disk: Disk | None,
    prior: np.ndarray,
    posterior: np.ndarray,
    heading: str,
    subhead: str,
    cloud_lines: list[str] | None = None,
    depth_prior: np.ndarray | None = None,
    depth_current: np.ndarray | None = None,
) -> np.ndarray:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    _paste_rgb(canvas, main_rgb, (24, 24, 776, 776))
    _paste_rgb(canvas, zoom_rgb, (796, 24, 1256, 220))
    draw.text((796, 232), heading, font=_font(22, bold=True), fill=INK)
    draw.text((796, 260), subhead, font=_font(14), fill=MUTED)
    draw.text((796, 286), phase, font=_font(20, bold=True), fill=GOLD)
    draw.text((796, 314), detail, font=_font(14), fill=MUTED)
    lines = [f"inclination   {inclination:5.1f}°"]
    if azimuth is None:
        lines.append("azimuth         —")
    else:
        lines.append(f"azimuth       {azimuth:5.0f}°")
    if cloud_lines is not None:
        lines += cloud_lines
    elif disk is None:
        lines += ["parameters follow", "once the tilt starts"]
    else:
        lines += [
            f"h_R          {disk.h_R:5.2f} kpc",
            f"τ_V(center)  {disk.tau_V0:5.2f}",
            f"h★           {disk.h_star:5.2f} kpc",
            f"h dust       {disk.h_dust:5.2f} kpc",
            f"h★ / h dust  {disk.h_star / disk.h_dust:5.2f}",
        ]
    y = 344
    for line in lines:
        draw.text((796, y), line, font=_font(17), fill=INK)
        y += 22
    if depth_prior is not None:
        _depth_scatter(draw, (796, 530, 1256, 776), depth_prior, depth_current)
    else:
        current = None if disk is None else (disk.h_star, disk.h_dust)
        _scatter(draw, (796, 530, 1256, 776), prior, posterior, current)
    return np.asarray(canvas)


def _zoom_box(inclination: float, extent: float) -> tuple[float, float]:
    extent_x, extent_y = zoom_extents(inclination)
    extent_x = min(extent_x, extent * 0.72)
    extent_y = min(extent_y, extent_x)
    return extent_x, float(extent_y)


def _render_pair(
    disk: Disk,
    inclination: float,
    azimuth: float,
    scale: float,
    steps: int,
    extent: float,
    extinction: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    main = render(
        disk,
        inclination,
        azimuth % 360.0,
        npix_x=320,
        extent_x=extent,
        steps=steps,
        extinction=extinction,
    )
    extent_x, extent_y = _zoom_box(inclination, extent)
    npix_y = max(96, int(round(220 * extent_y / extent_x)))
    zoom = render(
        disk,
        inclination,
        azimuth % 360.0,
        npix_x=220,
        npix_y=npix_y,
        extent_x=extent_x,
        extent_y=extent_y,
        steps=steps,
        extinction=extinction,
    )
    blur = 0.3
    return tonemap(convolve_gaussian(main, blur), scale=scale), tonemap(convolve_gaussian(zoom, blur), scale=scale)


def _render_cloud_pair(
    cloud: GaussianCloud,
    inclination: float,
    azimuth: float,
    scale: float,
    extent: float,
    extinction: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    az = azimuth % 360.0
    main = render_cloud(
        cloud,
        inclination,
        az,
        npix_x=400,
        extent_x=extent,
        extinction=extinction,
        n_slices=12,
    )
    extent_x, extent_y = _zoom_box(inclination, extent)
    npix_y = max(96, int(round(220 * extent_y / extent_x)))
    zoom = render_cloud(
        cloud,
        inclination,
        az,
        npix_x=220,
        npix_y=npix_y,
        extent_x=extent_x,
        extent_y=extent_y,
        extinction=extinction,
        n_slices=16,
    )
    return tonemap(main, scale=scale), tonemap(zoom, scale=scale)


def _cloud_caption(cloud: GaussianCloud) -> list[str]:
    median_sigma = float(np.median(cloud.star_sigma)) if len(cloud.star_sigma) else 0.0
    return [
        f"stars        {len(cloud.star_sigma):5d}",
        f"dust clouds  {len(cloud.dust_sigma):5d}",
        f"median σ     {median_sigma:5.2f} kpc",
        f"z rms        {cloud.z_rms:5.2f} kpc",
    ]


def _depth_prior_points(cloud: GaussianCloud, n: int = 2400, amplitude: float = 1.2, seed: int = 9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = np.concatenate([cloud.star_sigma, cloud.dust_sigma])
    if len(sigma) == 0:
        return np.zeros((0, 2))
    pick = rng.integers(0, len(sigma), size=n)
    z_abs = np.abs(amplitude * np.maximum(sigma[pick], 0.10) * rng.normal(size=n))
    return np.stack([sigma[pick], z_abs], axis=1)


def _depth_current_points(cloud: GaussianCloud) -> np.ndarray:
    sigma = np.concatenate([cloud.star_sigma, cloud.dust_sigma])
    z_abs = np.concatenate([np.abs(cloud.star_mu[:, 2]), np.abs(cloud.dust_mu[:, 2])])
    if len(sigma) == 0:
        return np.zeros((0, 2))
    return np.stack([sigma, z_abs], axis=1)


def _crop_observation(obs_rgb: np.ndarray, inclination: float, extent: float) -> np.ndarray:
    """Cut the inner sky out of the face-on observation for the side panel."""
    extent_x, extent_y = _zoom_box(inclination, extent)
    ny, nx = obs_rgb.shape[:2]
    # Sky Y is up in the image; row 0 is +extent.
    x0 = int(round((extent - extent_x) / (2 * extent) * nx))
    x1 = int(round((extent + extent_x) / (2 * extent) * nx))
    y0 = int(round((extent - extent_y) / (2 * extent) * ny))
    y1 = int(round((extent + extent_y) / (2 * extent) * ny))
    return obs_rgb[y0:y1, x0:x1]


def write_movie(
    path: Path,
    reference: Disk,
    path_disks: list[Disk],
    prior_heights: np.ndarray,
    posterior_heights: np.ndarray,
    scale: float,
    fps: int = 10,
    steps: int = 130,
    observed_linear: np.ndarray | None = None,
    extent: float = 12.0,
    extinction: np.ndarray | None = None,
    heading: str = "Synthetic disk",
    subhead: str = "Exponential star and dust atmospheres",
    observed_detail: str = "Synthetic g, r, i image",
    orbit_seconds: float = 8.0,
    cloud: GaussianCloud | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if observed_linear is None:
        observed = _observed_frame(scale, extent=extent)
    else:
        observed = tonemap(observed_linear, scale=scale)
    hold = int(1.6 * fps)
    fade = max(fps // 2, 1)
    spin = int(round(orbit_seconds * fps))
    frames = hold + fade + spin + spin
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    depth_draws = depth_route(cloud) if cloud is not None else None
    depth_prior = _depth_prior_points(cloud) if cloud is not None else None
    if cloud is None:
        face_main, face_zoom = _render_pair(reference, 0.0, 0.0, scale, steps, extent, extinction)
    else:
        face_main, face_zoom = _render_cloud_pair(cloud, 0.0, 0.0, scale, extent, extinction)
    print(f"Writing {frames} frames to {path}", flush=True)
    try:
        for index in range(frames):
            azimuth: float | None
            shown = cloud
            if index < hold:
                main_rgb = observed
                zoom_rgb = _crop_observation(observed, 0.0, extent)
                phase, detail, inclination, azimuth, disk = (
                    "Observed face-on",
                    observed_detail,
                    0.0,
                    None,
                    None,
                )
            elif index < hold + fade:
                u = (index - hold + 1) / fade
                main_rgb = _blend(observed, face_main, u)
                zoom_rgb = _blend(_crop_observation(observed, 0.0, extent), face_zoom, u)
                phase, detail, inclination, azimuth, disk = (
                    "Observed face-on",
                    "Fading to spherical clouds" if cloud is not None else "Fading to one posterior draw",
                    0.0,
                    0.0,
                    None if cloud is not None else reference,
                )
            elif index < hold + fade + spin:
                t = (index - (hold + fade)) / spin
                inclination, azimuth = orbit_pose(t)
                if cloud is None:
                    main_rgb, zoom_rgb = _render_pair(reference, inclination, azimuth, scale, steps, extent, extinction)
                    phase, detail = "First tilt", "Inclination through a full turn"
                    disk = reference
                else:
                    main_rgb, zoom_rgb = _render_cloud_pair(cloud, inclination, azimuth, scale, extent, extinction)
                    phase, detail = "First tilt", "Tilting through 360°"
                    disk = None
            else:
                t = (index - (hold + fade + spin)) / spin
                inclination, azimuth = orbit_pose(t)
                if cloud is None:
                    disk = blend_disks(path_disks, t)
                    main_rgb, zoom_rgb = _render_pair(disk, inclination, azimuth, scale, steps, extent, extinction)
                    phase, detail = "Second tilt", "Same arms, other scale heights"
                else:
                    assert depth_draws is not None
                    star_z, dust_z = blend_depth(depth_draws, t)
                    shown = cloud.with_z(star_z, dust_z)
                    main_rgb, zoom_rgb = _render_cloud_pair(shown, inclination, azimuth, scale, extent, extinction)
                    phase, detail = "Second tilt", "Same clouds, other depths"
                    disk = None
            frame = compose_frame(
                main_rgb,
                zoom_rgb,
                phase=phase,
                detail=detail,
                inclination=inclination,
                azimuth=azimuth,
                disk=disk,
                prior=prior_heights,
                posterior=posterior_heights,
                heading=heading,
                subhead=subhead,
                cloud_lines=None if shown is None else _cloud_caption(shown),
                depth_prior=depth_prior,
                depth_current=None if shown is None else _depth_current_points(shown),
            )
            process.stdin.write(frame.tobytes())
            if index % fps == 0:
                az_text = "—" if azimuth is None else f"{azimuth:.0f}°"
                print(
                    f"  frame {index + 1}/{frames}  {phase}  i={inclination:.0f}°  az={az_text}",
                    flush=True,
                )
    finally:
        process.stdin.close()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited with {code}")


def _blend(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    if a.shape != b.shape:
        b_img = Image.fromarray(b, mode="RGB").resize((a.shape[1], a.shape[0]), Image.Resampling.LANCZOS)
        b = np.asarray(b_img)
    mixed = (1.0 - u) * a.astype(np.float32) + u * b.astype(np.float32)
    return np.clip(mixed + 0.5, 0, 255).astype(np.uint8)


def _observed_frame(scale: float, extent: float = 12.0, npix: int = 340) -> np.ndarray:
    """High-resolution version of the synthetic observation, for the opening frames.

    Same galaxy, PSF, and noise amplitude as the fit. Rendered on the movie grid
    so the opening is the galaxy rather than a blown-up coarse pixel grid.
    """
    clean = faceon_image(TRUTH, npix=npix, extent=extent, n_z=1200)
    psf = 0.42 / (2.0 * extent / npix)
    clean = convolve_gaussian(clean, psf)
    noisy = clean + np.random.default_rng(1).normal(0.0, 0.05, size=clean.shape)
    return tonemap(noisy, scale=scale)
