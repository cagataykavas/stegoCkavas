"""Command-line interface and zero-dataset demonstration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np

from .stego_algorithms import payload_capacity_bytes, process_image


TEXT_ALGORITHMS = ("lsb", "dct", "dwt")


def project_description() -> Dict[str, Any]:
    return {
        "project": "Stego Lab",
        "scope": "Reproducible image steganography and steganalysis experiments.",
        "embedding": {
            "lsb": "Seeded pixel-channel permutation with a delimited UTF-8 payload.",
            "dct": "Quantization-index modulation in 8x8 luminance coefficients.",
            "dwt": "Quantization-index modulation in Haar detail bands.",
            "augmentation_only": ["pvd", "dft", "svd"],
        },
        "integrity": "Transform payloads contain a versioned header and CRC32.",
        "steganalysis": {
            "features": ["raw", "dct", "residual_hist", "residual_cooc"],
            "classical_models": ["random_forest", "logistic_regression", "svm", "xgboost", "lightgbm"],
            "deep_model": "Fixed high-pass filters, truncation, and a compact CNN.",
            "explanations": ["input-gradient saliency", "residual maps", "occlusion heatmaps"],
        },
        "security_boundary": (
            "A permutation seed and CRC provide reproducibility and integrity checks, not encryption or secrecy."
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _psnr(first: np.ndarray, second: np.ndarray) -> float:
    error = float(np.mean((first.astype(np.float32) - second.astype(np.float32)) ** 2))
    if error == 0.0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(error))


def _synthetic_cover(size: int, seed: int) -> np.ndarray:
    """Create a deterministic textured cover so the demo needs no downloaded data."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    rng = np.random.default_rng(seed)
    texture = rng.normal(0.0, 4.0, size=(size, size)).astype(np.float32)
    blue = 80.0 + 60.0 * np.sin(x / 29.0) + 0.18 * y + texture
    green = 95.0 + 45.0 * np.cos(y / 37.0) + 0.14 * x + texture
    red = 110.0 + 35.0 * np.sin((x + y) / 41.0) + 0.08 * x + texture
    return np.clip(np.dstack([blue, green, red]), 0, 255).astype(np.uint8)


def run_demo(output_dir: Path, message: str, seed: int, size: int = 512) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cover_path = output_dir / "cover.png"
    cover = _synthetic_cover(size=size, seed=seed)
    if not cv2.imwrite(str(cover_path), cover):
        raise IOError(f"Could not write {cover_path}")

    results: Dict[str, Any] = {}
    for index, algorithm in enumerate(TEXT_ALGORITHMS):
        algorithm_seed = seed + index
        stego_path = output_dir / f"stego_{algorithm}.png"
        process_image(
            str(cover_path),
            algorithm,
            action="embed",
            secret_message=message,
            seed=algorithm_seed,
            out_path=str(stego_path),
        )
        recovered = process_image(
            str(stego_path),
            algorithm,
            action="extract",
            seed=algorithm_seed,
        )
        stego = cv2.imread(str(stego_path), cv2.IMREAD_COLOR)
        if stego is None:
            raise IOError(f"Could not read generated image {stego_path}")
        results[algorithm] = {
            "round_trip": recovered == message,
            "recovered_message": recovered,
            "capacity_bytes": payload_capacity_bytes(size, size, algorithm),
            "psnr_db": round(_psnr(cover, stego), 3),
            "output": stego_path.name,
            "sha256": _sha256(stego_path),
        }

    report = {
        "status": "ok" if all(item["round_trip"] for item in results.values()) else "failed",
        "seed": seed,
        "cover": {"path": cover_path.name, "size": [size, size], "sha256": _sha256(cover_path)},
        "message_utf8_bytes": len(message.encode("utf-8")),
        "algorithms": results,
        "security_note": project_description()["security_boundary"],
    }
    report_path = output_dir / "demo_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if report["status"] != "ok":
        raise RuntimeError(f"At least one round trip failed; inspect {report_path}")
    return report


def _read_key(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image steganography and steganalysis lab.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed = subparsers.add_parser("embed", help="Hide a UTF-8 message in a PNG-compatible image.")
    embed.add_argument("--input", type=Path, required=True)
    embed.add_argument("--output", type=Path, required=True)
    embed.add_argument("--algorithm", choices=TEXT_ALGORITHMS, default="lsb")
    embed.add_argument("--message", required=True)
    embed.add_argument("--seed", type=int, default=42)
    embed.add_argument("--key-output", type=Path)

    extract = subparsers.add_parser("extract", help="Recover a UTF-8 message.")
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--algorithm", choices=TEXT_ALGORITHMS, default="lsb")
    extract.add_argument("--seed", type=int, default=42)
    extract.add_argument("--key", type=Path)

    demo = subparsers.add_parser("demo", help="Run LSB, DCT, and DWT round trips without a dataset.")
    demo.add_argument("--output-dir", type=Path, default=Path("demo_output"))
    demo.add_argument("--message", default="reproducible stego demo")
    demo.add_argument("--seed", type=int, default=1337)
    demo.add_argument("--size", type=int, default=512)

    subparsers.add_parser("describe", help="Print the project contract as JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "describe":
        print(json.dumps(project_description(), indent=2, sort_keys=True))
        return 0
    if args.command == "demo":
        print(json.dumps(run_demo(args.output_dir, args.message, args.seed, args.size), indent=2, sort_keys=True))
        return 0
    if args.command == "embed":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        process_image(
            str(args.input),
            args.algorithm,
            action="embed",
            secret_message=args.message,
            seed=args.seed,
            out_path=str(args.output),
        )
        key_path = args.key_output or args.output.with_suffix(args.output.suffix + ".key.json")
        key = {"algorithm": args.algorithm, "seed": args.seed, "format": 1}
        key_path.write_text(json.dumps(key, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"status": "ok", "output": str(args.output), "key": str(key_path)}, indent=2))
        return 0

    algorithm = args.algorithm
    seed = args.seed
    if args.key:
        key = _read_key(args.key)
        algorithm = str(key["algorithm"])
        seed = int(key["seed"])
    message = process_image(str(args.input), algorithm, action="extract", seed=seed)
    print(message)
    return 0
