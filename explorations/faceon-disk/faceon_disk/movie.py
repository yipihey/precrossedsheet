"""Two orbits of one posterior disk, then a path through other draws.

Each orbit lifts off the face-on view, circles a full 360° of azimuth at fixed
inclination, and sets back down. An axisymmetric disk would look the same at
every azimuth; the arms carried from the image are what make the circuit visible.
The second orbit moves the scale heights through posterior realizations.
Face-on frames of those realizations agree; the inclined frames show the heights.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .infer import TRUTH, blend_disks
from .model import Disk, convolve_gaussian, faceon_image, render, tonemap

WIDTH = 1280
HEIGHT = 800
BG = (8, 10, 16)
INK = (236, 238, 244)
MUTED = (158, 166, 180)
GOLD = (232, 190, 118)
PRIOR_COLOR = (88, 104, 128)
LINE = (42, 48, 62)


ORBIT_INCLINATION = 58.0
ORBIT_LIFTOFF = 0.14


def _smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def orbit_pose(
    t: float,
    i_cruise: float = ORBIT_INCLINATION,
    liftoff: float = ORBIT_LIFTOFF,
) -> tuple[float, float]:
    """Camera inclination and azimuth for one closed orbit. t is in [0, 1].

    The view starts and ends face-on at azimuth 0. Between the liftoff and the
    landing the inclination is constant and the azimuth advances exactly 360°,
    with zero angular speed at both ends of that cruise so the joins do not jerk.
    """
    if t <= 0.0:
        return 0.0, 0.0
    if t >= 1.0:
        return 0.0, 360.0
    if t < liftoff:
        return i_cruise * _smoothstep(t / liftoff), 0.0
    if t > 1.0 - liftoff:
        return i_cruise * _smoothstep((1.0 - t) / liftoff), 360.0
    u = (t - liftoff) / (1.0 - 2.0 * liftoff)
    return i_cruise, 360.0 * _smoothstep(u)


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
    if disk is None:
        lines += ["parameters follow", "once the orbit starts"]
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
    face_main, face_zoom = _render_pair(reference, 0.0, 0.0, scale, steps, extent, extinction)
    print(f"Writing {frames} frames to {path}", flush=True)
    try:
        for index in range(frames):
            azimuth: float | None
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
                    "Fading to one posterior draw",
                    0.0,
                    0.0,
                    reference,
                )
            elif index < hold + fade + spin:
                t = (index - (hold + fade)) / spin
                inclination, azimuth = orbit_pose(t)
                main_rgb, zoom_rgb = _render_pair(reference, inclination, azimuth, scale, steps, extent, extinction)
                phase, detail = "First orbit", "Arms from the image, one full turn"
                disk = reference
            else:
                t = (index - (hold + fade + spin)) / spin
                inclination, azimuth = orbit_pose(t)
                disk = blend_disks(path_disks, t)
                main_rgb, zoom_rgb = _render_pair(disk, inclination, azimuth, scale, steps, extent, extinction)
                phase, detail = "Second orbit", "Same arms, other scale heights"
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
