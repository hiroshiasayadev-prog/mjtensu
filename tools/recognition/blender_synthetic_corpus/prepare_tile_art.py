from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import cairosvg
from PIL import Image


UPSTREAM_URL = "https://github.com/FluffyStuff/riichi-mahjong-tiles"
HONOR_FILES = {
    "east": "Ton.svg",
    "south": "Nan.svg",
    "west": "Shaa.svg",
    "north": "Pei.svg",
    "white": "Haku.svg",
    "green": "Hatsu.svg",
    "red": "Chun.svg",
}


def tile_files() -> dict[str, str]:
    files: dict[str, str] = {}
    for suit, prefix in (("m", "Man"), ("p", "Pin"), ("s", "Sou")):
        for rank in range(1, 10):
            files[f"{rank}{suit}"] = f"{prefix}{rank}.svg"
        files[f"red5{suit}"] = f"{prefix}5-Dora.svg"
    files.update(HONOR_FILES)
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rasterize(svg_path: Path, width: int, height: int) -> Image.Image:
    payload = cairosvg.svg2png(
        url=str(svg_path),
        output_width=width,
        output_height=height,
    )
    return Image.open(io.BytesIO(payload)).convert("RGBA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rasterize canonical FluffyStuff Regular face-art SVGs as transparent decals."
    )
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upstream_root = args.upstream_root.resolve()
    regular = upstream_root / "Regular"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    front_path = regular / "Front.svg"
    back_path = regular / "Back.svg"
    license_path = upstream_root / "LICENSE.md"
    for required in (front_path, back_path, license_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    manifest: dict[str, object] = {
        "schema": "mjtensu.blender-synthetic-tile-art/v1",
        "canonical_upstream_url": UPSTREAM_URL,
        "upstream_root": str(upstream_root),
        "variant": "Regular",
        "front_svg_sha256": sha256(front_path),
        "back_svg_sha256": sha256(back_path),
        "license_sha256": sha256(license_path),
        "raster_size": [args.width, args.height],
        "extraction": (
            "Canonical upstream Regular/ keeps the physical Front/Back artwork separate from "
            "the tile-face SVGs. Rasterize the face SVGs directly on transparent backgrounds; "
            "do not use the frontend-precomposed copies under external/agari. Haku.svg is "
            "intentionally transparent, so white-dragon appearance comes from the 3D ivory body."
        ),
        "tiles": {},
    }

    for code, file_name in tile_files().items():
        source_path = regular / file_name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        rgba = rasterize(source_path, args.width, args.height)
        out_path = output / f"{code}.png"
        rgba.save(out_path, optimize=True)

        alpha = rgba.getchannel("A")
        histogram = alpha.histogram()
        nonzero = sum(histogram[1:])
        manifest["tiles"][code] = {
            "source_file": file_name,
            "source_sha256": sha256(source_path),
            "artwork_png": out_path.name,
            "artwork_sha256": sha256(out_path),
            "alpha_fraction": nonzero / (args.width * args.height),
        }

    white = manifest["tiles"]["white"]
    if float(white["alpha_fraction"]) > 0.001:
        raise RuntimeError(
            "Canonical Haku.svg unexpectedly contains visible face artwork: "
            f"alpha_fraction={white['alpha_fraction']}"
        )

    manifest_path = output / "asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "ok",
        "manifest": str(manifest_path),
        "tiles": len(tile_files()),
        "white_alpha_fraction": white["alpha_fraction"],
    }, indent=2))


if __name__ == "__main__":
    main()
