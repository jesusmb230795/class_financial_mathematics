"""Check local links in a generated Jupyter Book HTML directory."""

from __future__ import annotations

import argparse
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


IGNORED_SCHEMES = {"http", "https", "mailto", "tel", "javascript", "data"}
LINK_ATTRIBUTES = {"href", "src"}


class LinkParser(HTMLParser):
    """Collect element ids and local link-like attributes from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if not value:
                continue
            if name == "id" or name == "name":
                self.ids.add(value)
            if name in LINK_ATTRIBUTES:
                self.links.append((name, value))


def parse_html(path: Path) -> LinkParser:
    parser = LinkParser()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    return parser


def is_ignored(raw_link: str) -> bool:
    parsed = urlsplit(raw_link)
    return parsed.scheme in IGNORED_SCHEMES or raw_link.startswith("//")


def resolve_target(root: Path, current_file: Path, raw_link: str) -> tuple[Path, str] | None:
    if not raw_link or raw_link == "#":
        return None
    if is_ignored(raw_link):
        return None

    parsed = urlsplit(raw_link)
    if parsed.scheme or parsed.netloc:
        return None

    raw_path = unquote(parsed.path)
    fragment = unquote(parsed.fragment)
    if not raw_path:
        return current_file, fragment

    target = (current_file.parent / raw_path).resolve()
    if raw_path.endswith("/"):
        target = target / "index.html"
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target, fragment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html_dir", type=Path, help="Generated _build/html directory")
    args = parser.parse_args()

    root = args.html_dir.resolve()
    if not root.exists():
        raise SystemExit(f"{root} does not exist. Build the book before checking links.")

    html_files = [
        path
        for path in root.rglob("*.html")
        if "_static" not in path.relative_to(root).parts
    ]
    parsed_files = {path: parse_html(path) for path in html_files}
    ids_by_file = {path: parsed.ids for path, parsed in parsed_files.items()}

    errors: dict[Path, list[str]] = defaultdict(list)
    for html_file, parsed in parsed_files.items():
        for attr_name, raw_link in parsed.links:
            target_info = resolve_target(root, html_file, raw_link)
            if target_info is None:
                continue

            target, fragment = target_info
            if not target.exists():
                errors[html_file].append(f"missing {attr_name} target: {raw_link}")
                continue

            if fragment and target.suffix == ".html":
                target_ids = ids_by_file.get(target)
                if target_ids is None:
                    target_ids = parse_html(target).ids
                    ids_by_file[target] = target_ids
                if fragment not in target_ids:
                    errors[html_file].append(f"missing anchor in {raw_link}: #{fragment}")

    if errors:
        print("Book link check failed:")
        for html_file in sorted(errors):
            relative = html_file.relative_to(root)
            for error in errors[html_file]:
                print(f"  {relative}: {error}")
        return 1

    print(f"Checked {len(html_files)} HTML files under {root}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
