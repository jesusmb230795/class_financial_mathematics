"""Install the repository's JupyterLab defaults in an isolated local profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_SETTINGS = REPO_ROOT / "jupyter" / "lab-settings"
TARGET_SETTINGS = REPO_ROOT / ".jupyter-config" / "lab" / "user-settings"
SEED_WORKSPACES = REPO_ROOT / "jupyter" / "workspaces"
TARGET_WORKSPACES = REPO_ROOT / ".jupyter-config" / "lab" / "workspaces"


def copy_tree(source: Path, target: Path, *, force: bool) -> int:
    copied = 0
    for source_file in sorted(source.rglob("*")):
        if not source_file.is_file():
            continue
        destination = target / source_file.relative_to(source)
        if destination.exists() and not force:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)
        copied += 1
    return copied


def install_workspaces(*, force: bool) -> int:
    installed = 0
    TARGET_WORKSPACES.mkdir(parents=True, exist_ok=True)
    destinations = set()
    for source_file in sorted(SEED_WORKSPACES.glob("*.jupyterlab-workspace")):
        workspace = json.loads(source_file.read_text(encoding="utf-8"))
        workspace_id = workspace["metadata"]["id"]
        workspace_hash = hashlib.sha256(workspace_id.encode()).hexdigest()[:4]
        workspace_slug = workspace_id.replace("/", "-")
        destination = TARGET_WORKSPACES / (
            f"{workspace_slug}-{workspace_hash}.jupyterlab-workspace"
        )
        destinations.add(destination)
        if destination.exists() and not force:
            continue
        destination.write_text(
            json.dumps(workspace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        installed += 1
    if force:
        for stale_workspace in TARGET_WORKSPACES.glob("*.jupyterlab-workspace"):
            if stale_workspace not in destinations:
                stale_workspace.unlink()
    return installed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Restore tracked defaults, replacing local JupyterLab settings.",
    )
    args = parser.parse_args()

    settings = copy_tree(SEED_SETTINGS, TARGET_SETTINGS, force=args.force)
    workspaces = install_workspaces(force=args.force)
    print(
        f"JupyterLab profile ready: {settings} setting file(s), "
        f"{workspaces} workspace(s) installed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
