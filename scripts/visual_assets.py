#!/usr/bin/env python3
"""Manage generated visual assets for the Jupyter Book."""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "img" / "generated" / "visual-assets.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def save_manifest(data: dict) -> None:
    MANIFEST.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def asset_map(data: dict) -> dict[str, dict]:
    return {asset["id"]: asset for asset in data["assets"]}


def selected_assets(data: dict, asset_id: str | None = None) -> list[dict]:
    assets = sorted(data["assets"], key=lambda item: item["order"])
    if asset_id is None:
        return assets
    matches = [asset for asset in assets if asset["id"] == asset_id]
    if not matches:
        raise SystemExit(f"Unknown asset id: {asset_id}")
    return matches


def rel_path(path_value: str) -> Path:
    return ROOT / path_value


def split_source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines if lines else [""]


def read_png_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", header[16:24])


def generator_prompt(data: dict, asset: dict) -> str:
    style = data["style"]
    defaults = style.get("output_defaults", {})
    constraints = [
        f"Exact final canvas: {asset['recommended_size']} px.",
        f"Aspect ratio: {asset['aspect_ratio']}.",
        f"File format: {defaults.get('file_format', 'PNG')}.",
        f"Color space: {defaults.get('color_space', 'sRGB')}.",
        f"Background: {defaults.get('background', 'opaque off-white')}.",
        defaults.get("safe_margin", "Keep a clear safe margin."),
        defaults.get("text_size", "Use only readable labels."),
    ]
    return "\n".join(
        [
            asset["prompt"],
            "",
            "Shared style:",
            style["base_prompt"],
            "",
            "Output requirements:",
            *constraints,
            "",
            "Avoid:",
            style["negative_prompt"],
        ]
    )


def status_command(args: argparse.Namespace) -> int:
    data = load_manifest()
    print("Order  Status     File     Book     Size       Asset")
    print("-----  ---------  -------  -------  ---------  -------------------------------")
    for asset in selected_assets(data, args.asset):
        exists = rel_path(asset["target_path"]).exists()
        file_state = "exists" if exists else "missing"
        book_state = "in-book" if asset.get("used_by") else "unused"
        print(
            f"{asset['order']:>5}  {asset['status']:<9}  "
            f"{file_state:<7}  {book_state:<7}  "
            f"{asset['recommended_size']:<9}  {asset['id']}"
        )

    pending = [
        asset for asset in selected_assets(data, args.asset)
        if asset["status"] == "pending"
    ]
    if pending:
        next_asset = pending[0]
        print(f"\nNext pending: {next_asset['order']:02d} {next_asset['id']}")
        print(f"Target path: {next_asset['target_path']}")

    if args.show_prompts:
        for asset in selected_assets(data, args.asset):
            print(f"\n## {asset['order']:02d} {asset['id']}")
            print(f"Target: {asset['target_path']}")
            print(f"Size: {asset['recommended_size']} px ({asset['aspect_ratio']})")
            print(f"Alt text: {asset['alt_text']}")
            print("\nCopy/paste prompt:")
            print("```text")
            print(generator_prompt(data, asset))
            print("```")
            if "negative_prompt" in data["style"]:
                print("\nNegative prompt field, if the generator has one:")
                print("```text")
                print(data["style"]["negative_prompt"])
                print("```")

    return 0


def validate_command(args: argparse.Namespace) -> int:
    data = load_manifest()
    valid_statuses = set(data["production_status_values"])
    errors: list[str] = []
    warnings: list[str] = []

    for asset in selected_assets(data, args.asset):
        status = asset["status"]
        target = rel_path(asset["target_path"])
        size = asset.get("recommended_size", "")
        if status not in valid_statuses:
            errors.append(f"{asset['id']}: invalid status {status!r}")
        if not re.fullmatch(r"\d+x\d+", size):
            errors.append(f"{asset['id']}: invalid recommended_size {size!r}")
        if size and size not in asset.get("prompt", ""):
            warnings.append(
                f"{asset['id']}: prompt does not mention recommended_size {size}"
            )
        if status in {"generated", "approved"} and not target.exists():
            errors.append(
                f"{asset['id']}: status is {status!r}, but {asset['target_path']} is missing"
            )
        if target.exists() and target.suffix.lower() == ".png" and re.fullmatch(r"\d+x\d+", size):
            actual_size = read_png_size(target)
            expected_width, expected_height = (int(part) for part in size.split("x"))
            if actual_size and actual_size != (expected_width, expected_height):
                errors.append(
                    f"{asset['id']}: expected {size}px, but file is "
                    f"{actual_size[0]}x{actual_size[1]}px"
                )
        if status == "pending" and target.exists():
            warnings.append(
                f"{asset['id']}: file exists, but status is still pending"
            )
        for placement in asset.get("used_by", []):
            source = ROOT / placement["file"]
            if not source.exists():
                errors.append(f"{asset['id']}: placement source missing: {placement['file']}")

    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)

    if errors:
        return 1

    print("Visual asset manifest is valid.")
    return 0


def replace_text_file(path: Path, current_ref: str, target_ref: str) -> str:
    text = path.read_text(encoding="utf-8")
    if current_ref in text:
        path.write_text(text.replace(current_ref, target_ref), encoding="utf-8")
        return "changed"
    if target_ref in text:
        return "already"
    return "missing-reference"


def replace_ipynb_text(path: Path, current_ref: str, target_ref: str) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    already = False
    for cell in data.get("cells", []):
        source = "".join(cell.get("source", []))
        if target_ref in source:
            already = True
        if current_ref in source:
            cell["source"] = split_source(source.replace(current_ref, target_ref))
            changed = True
    if changed:
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return "changed"
    return "already" if already else "missing-reference"


def replace_ipynb_markdown_cell(path: Path, cell_index: int, target_markdown: str) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    idx = cell_index - 1
    if idx < 0 or idx >= len(cells):
        return "missing-cell"
    desired = target_markdown.rstrip() + "\n"
    current = "".join(cells[idx].get("source", []))
    if current == desired:
        return "already"
    cells[idx]["cell_type"] = "markdown"
    cells[idx]["source"] = [desired]
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return "changed"


def markdown_insert_status(path: Path, target_markdown: str) -> str:
    if path.suffix == ".ipynb":
        data = json.loads(path.read_text(encoding="utf-8"))
        for cell in data.get("cells", []):
            if target_markdown in "".join(cell.get("source", [])):
                return "already"
        return "missing-reference"

    text = path.read_text(encoding="utf-8")
    return "already" if target_markdown in text else "missing-reference"


def apply_placement(placement: dict) -> tuple[str, str]:
    path = ROOT / placement["file"]
    mode = placement["mode"]
    if mode == "text_replace":
        if path.suffix == ".ipynb":
            result = replace_ipynb_text(path, placement["current_ref"], placement["target_ref"])
        else:
            result = replace_text_file(path, placement["current_ref"], placement["target_ref"])
    elif mode == "replace_markdown_cell":
        result = replace_ipynb_markdown_cell(
            path,
            int(placement["cell_index"]),
            placement["target_markdown"],
        )
    elif mode == "markdown_insert":
        result = markdown_insert_status(path, placement["target_markdown"])
    else:
        result = f"unsupported-mode:{mode}"
    return placement["file"], result


def apply_command(args: argparse.Namespace) -> int:
    data = load_manifest()
    if not args.all and not args.asset and not args.existing:
        raise SystemExit("Use --asset ASSET_ID, --all, or --existing.")

    if args.existing:
        assets = [
            asset for asset in selected_assets(data)
            if rel_path(asset["target_path"]).exists()
        ]
    else:
        assets = selected_assets(data, args.asset) if not args.all else selected_assets(data)
    failures: list[str] = []
    for asset in assets:
        target = rel_path(asset["target_path"])
        if asset.get("used_by") and not target.exists() and not args.allow_missing:
            failures.append(
                f"{asset['id']}: target image missing: {asset['target_path']}"
            )
            continue
        if not asset.get("used_by"):
            print(f"{asset['id']}: no book placements to apply")
            continue
        for placement in asset["used_by"]:
            file_name, result = apply_placement(placement)
            print(f"{asset['id']}: {file_name}: {result}")

    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    return 1 if failures else 0


def sync_command(args: argparse.Namespace) -> int:
    data = load_manifest()
    changed_status = False
    generated_assets = [
        asset for asset in selected_assets(data)
        if rel_path(asset["target_path"]).exists()
    ]

    if not generated_assets:
        print("No generated image files found under img/generated/.")
        return 0

    failures: list[str] = []
    for asset in generated_assets:
        if asset["status"] == "pending":
            asset["status"] = "generated"
            changed_status = True
            print(f"{asset['id']}: marked generated")
        if not asset.get("used_by"):
            print(f"{asset['id']}: no book placements to apply")
            continue
        for placement in asset["used_by"]:
            file_name, result = apply_placement(placement)
            print(f"{asset['id']}: {file_name}: {result}")

    if changed_status:
        save_manifest(data)

    for failure in failures:
        print(f"ERROR: {failure}", file=sys.stderr)
    return 1 if failures else 0


def mark_command(args: argparse.Namespace) -> int:
    data = load_manifest()
    valid_statuses = set(data["production_status_values"])
    if args.status not in valid_statuses:
        raise SystemExit(f"Invalid status {args.status!r}. Use one of: {', '.join(sorted(valid_statuses))}")
    assets = asset_map(data)
    if args.asset not in assets:
        raise SystemExit(f"Unknown asset id: {args.asset}")
    asset = assets[args.asset]
    asset["status"] = args.status
    if args.note:
        asset.setdefault("production_notes", []).append(args.note)
    save_manifest(data)
    print(f"{args.asset}: marked {args.status}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="Show generation progress")
    status_parser.add_argument("--asset", help="Limit output to one asset id")
    status_parser.add_argument("--show-prompts", action="store_true", help="Print full prompts")
    status_parser.set_defaults(func=status_command)

    validate_parser = subparsers.add_parser("validate", help="Validate manifest and file state")
    validate_parser.add_argument("--asset", help="Limit validation to one asset id")
    validate_parser.set_defaults(func=validate_command)

    apply_parser = subparsers.add_parser("apply", help="Point book references at generated assets")
    apply_group = apply_parser.add_mutually_exclusive_group(required=True)
    apply_group.add_argument("--asset", help="Apply one asset id")
    apply_group.add_argument("--all", action="store_true", help="Apply all generated asset placements")
    apply_group.add_argument(
        "--existing",
        action="store_true",
        help="Apply only assets whose target image file already exists",
    )
    apply_parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Apply references even when the target image file does not exist",
    )
    apply_parser.set_defaults(func=apply_command)

    sync_parser = subparsers.add_parser(
        "sync",
        help="Mark existing generated files and apply their book placements",
    )
    sync_parser.set_defaults(func=sync_command)

    mark_parser = subparsers.add_parser("mark", help="Update one asset status")
    mark_parser.add_argument("asset", help="Asset id")
    mark_parser.add_argument("status", help="New status")
    mark_parser.add_argument("--note", help="Optional production note")
    mark_parser.set_defaults(func=mark_command)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
