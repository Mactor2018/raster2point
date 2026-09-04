from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np

from .core import alignment_report, load_raster, transfer
from .io import load_points, save_points


def main() -> None:
    parser = argparse.ArgumentParser(prog="raster2point")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("transfer", "rgb", "mask"):
        command = sub.add_parser(name)
        command.add_argument("raster")
        command.add_argument("points")
        command.add_argument("output")
        command.add_argument("--fill", type=float, default=np.nan if name != "mask" else -1)
        command.add_argument("--require-crs-match", action="store_true")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("raster")
    inspect.add_argument("points")
    batch = sub.add_parser("batch")
    batch.add_argument("manifest", help="CSV with raster,points,output columns")
    batch.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "inspect":
        points, _ = load_points(args.points)
        raster = load_raster(args.raster)
        print(json.dumps(alignment_report(raster, points).__dict__, default=str, indent=2))
        return
    if args.command == "batch":
        failures = 0
        with open(args.manifest, newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    if Path(row["output"]).exists() and not args.overwrite:
                        continue
                    _run(row["raster"], row["points"], row["output"], np.nan)
                except Exception as exc:  # report all samples, then fail at the end
                    failures += 1
                    print(f"ERROR {row.get('output', '?')}: {exc}")
        if failures:
            raise SystemExit(f"{failures} batch item(s) failed")
        return
    _run(args.raster, args.points, args.output, args.fill, args.require_crs_match)


def _run(raster_path: str, points_path: str, output: str, fill: float, require_crs_match: bool = False) -> None:
    points, metadata = load_points(points_path)
    result = transfer(points, load_raster(raster_path), fill_value=fill, require_crs_match=require_crs_match)
    save_points(output, np.column_stack((points, result.values)), metadata)
    print(f"wrote {output}: {int(result.valid.sum())}/{len(points)} points sampled")
