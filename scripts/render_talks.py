#!/usr/bin/env python3
"""Render standalone Quarto Reveal.js slide decks under talks/*/index.qmd into _site/talks/*/
and copy shared React laboratory components and assets into _site/talks/."""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TALKS_DIR = ROOT / "talks"
SITE_DIR = ROOT / "_site"
SITE_TALKS_DIR = SITE_DIR / "talks"


def sync_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    if os.environ.get("RENDERING_SUB_TALK") == "1":
        return

    SITE_TALKS_DIR.mkdir(parents=True, exist_ok=True)
    sync_tree(TALKS_DIR / "shared", SITE_TALKS_DIR / "shared")
    sync_tree(TALKS_DIR / "assets", SITE_TALKS_DIR / "assets")

    if (ROOT / "chapters" / "figures").exists():
        sync_tree(ROOT / "chapters" / "figures", SITE_DIR / "chapters" / "figures")

    if (TALKS_DIR / "README.md").exists():
        shutil.copy2(TALKS_DIR / "README.md", SITE_TALKS_DIR / "README.md")

    env = dict(os.environ)
    env["RENDERING_SUB_TALK"] = "1"

    for deck_qmd in sorted(TALKS_DIR.glob("*/index.qmd")):
        deck_dir = deck_qmd.parent
        build_dir = deck_dir / "_build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
        out_dir = SITE_TALKS_DIR / deck_dir.name
        print(f"[render_talks] Rendering Reveal.js deck: {deck_qmd.relative_to(ROOT)} -> {out_dir.relative_to(ROOT)}")
        subprocess.run(
            ["quarto", "render", "index.qmd"],
            cwd=str(deck_dir),
            env=env,
            check=True,
        )
        sync_tree(build_dir, out_dir)
        if build_dir.exists():
            shutil.rmtree(build_dir)
        for md_file in deck_dir.glob("*.md"):
            shutil.copy2(md_file, out_dir / md_file.name)

    for track_readme in sorted(TALKS_DIR.glob("*/README.md")):
        track_out = SITE_TALKS_DIR / track_readme.parent.name
        track_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(track_readme, track_out / "README.md")

    print("[render_talks] All presentations synced to _site/talks/")


if __name__ == "__main__":
    main()
