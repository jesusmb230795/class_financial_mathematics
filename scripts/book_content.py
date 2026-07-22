"""Validate the book content contract and stage incoming editorial text."""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = Path("content/plan.json")
DEBT_PATH = Path("content/debt.json")
INCOMING_TEMPLATE = Path("content/templates/incoming-text.md")
INCOMING_DIRECTORY = Path("content/incoming")
ROADMAP_PATH = Path("chapters/course-roadmap.md")
GLOSSARY_PATH = Path("chapters/glossary.md")
BIBLIOGRAPHY_PATH = Path("references.bib")
EXPECTED_MODULE_IDS = [f"{number:02d}" for number in range(11)]
REQUIRED_INCOMING_FIELDS = (
    "title",
    "module",
    "content_type",
    "source_language",
    "status",
    "target",
    "citations_status",
    "rights_status",
)
REQUIRED_INCOMING_SECTIONS = (
    "Purpose",
    "Learning objectives",
    "Draft text",
    "Sources to verify",
    "Integration notes",
)
TEXT_SUFFIXES = {".ipynb", ".json", ".md", ".py", ".toml", ".yaml", ".yml"}
SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
CONTENT_TYPE_PATTERN = re.compile(r"[a-z][a-z0-9-]*\Z")
DEBT_REQUIRED_STRING_FIELDS = (
    "id",
    "title",
    "category",
    "status",
    "priority",
    "observed_symptom",
    "potential_impact",
    "engineering_risk",
    "concrete_failure_mode",
    "owner",
    "recommended_next_action",
)
DEBT_REQUIRED_LIST_FIELDS = (
    "evidence",
    "affected_workflows_or_users",
    "dependencies",
    "validation_criteria",
)
DEBT_SCORE_FIELDS = ("impact", "urgency", "risk", "confidence", "effort")
DEBT_PRIORITIES = {"P0", "P1", "P2", "P3"}
FORBIDDEN_CHECKPOINT_HEADING = "## Checkpoint exercise"
LEGACY_NOTEBOOK_PATH_PATTERNS = (
    re.compile(r"notebooks/class/"),
    re.compile(r"""["']notebooks["']\s*/\s*["']class["']"""),
)
PUBLISHED_LESSON_MARKERS = {
    "learning objectives": ("## Learning objectives",),
    "prerequisites": ("## Prerequisites",),
    "handoff": ("## Handoff", "## Next step", "## Next steps"),
}


class BookContentError(ValueError):
    """Raised when a command cannot safely use the content contract."""


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_values(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BookContentError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BookContentError(
            f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def load_plan(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Load the canonical content plan or raise a concise contract error."""
    path = root / PLAN_PATH
    data = _load_json(path)
    if not isinstance(data, dict):
        raise BookContentError(f"{PLAN_PATH}: top-level value must be an object")
    return data


def load_debt_register(
    plan: dict[str, Any],
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Load the debt register selected by the canonical content plan."""
    book = plan.get("book")
    raw_path = book.get("debt_register") if isinstance(book, dict) else None
    if not _is_nonempty_string(raw_path):
        raw_path = DEBT_PATH.as_posix()
    data = _load_json(root / str(raw_path))
    if not isinstance(data, dict):
        raise BookContentError(f"{raw_path}: top-level value must be an object")
    return data


def _relative_path_error(value: object, label: str) -> str | None:
    if not _is_nonempty_string(value):
        return f"{label}: expected a non-empty relative path"
    raw = str(value)
    if "\\" in raw:
        return f"{label}: use POSIX separators in repository paths: {raw!r}"
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts:
        return f"{label}: path must stay inside the repository: {raw!r}"
    return None


def _validate_string_list(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, list):
        return [f"{label}: expected a list"]
    if not value and not allow_empty:
        errors.append(f"{label}: list must not be empty")
    for index, item in enumerate(value):
        if not _is_nonempty_string(item):
            errors.append(f"{label}[{index}]: expected a non-empty string")
    strings = [item for item in value if isinstance(item, str)]
    if len(strings) != len(set(strings)):
        errors.append(f"{label}: duplicate values are not allowed")
    return errors


def _validate_plan_schema(plan: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    if plan.get("schema_version") != 1:
        errors.append(f"{PLAN_PATH}: schema_version must be 1")

    book = plan.get("book")
    if not isinstance(book, dict):
        errors.append(f"{PLAN_PATH}: book must be an object")
        book = {}
    for field in (
        "title",
        "publication_language",
        "production_toc",
        "migration_toc",
        "notebook_manifest",
        "debt_register",
    ):
        if not _is_nonempty_string(book.get(field)):
            errors.append(f"{PLAN_PATH}: book.{field} must be a non-empty string")
    errors.extend(
        _validate_string_list(
            book.get("accepted_source_languages"),
            f"{PLAN_PATH}: book.accepted_source_languages",
        )
    )
    languages = book.get("accepted_source_languages")
    if (
        isinstance(languages, list)
        and _is_nonempty_string(book.get("publication_language"))
        and book["publication_language"] not in languages
    ):
        errors.append(
            f"{PLAN_PATH}: book.publication_language must be accepted as a source language"
        )

    allowed_values = plan.get("allowed_values")
    if not isinstance(allowed_values, dict):
        errors.append(f"{PLAN_PATH}: allowed_values must be an object")
        allowed_values = {}
    for field in ("publication_status", "completion_status"):
        errors.extend(
            _validate_string_list(
                allowed_values.get(field),
                f"{PLAN_PATH}: allowed_values.{field}",
            )
        )

    modules = plan.get("modules")
    if not isinstance(modules, list):
        return [*errors, f"{PLAN_PATH}: modules must be a list"]

    module_ids: list[str] = []
    publication_values = set(_string_values(allowed_values.get("publication_status")))
    completion_values = set(_string_values(allowed_values.get("completion_status")))
    required_fields = (
        "id",
        "title",
        "publication_status",
        "completion_status",
        "roadmap_status",
        "overview",
        "target_overview",
        "existing_sources",
        "gaps",
        "next_text_target",
    )

    for index, module in enumerate(modules):
        label = f"{PLAN_PATH}: modules[{index}]"
        if not isinstance(module, dict):
            errors.append(f"{label}: expected an object")
            continue
        for field in required_fields:
            if field not in module:
                errors.append(f"{label}: missing required field {field!r}")

        module_id = module.get("id")
        if not isinstance(module_id, str) or not re.fullmatch(r"\d{2}", module_id):
            errors.append(f"{label}.id: expected a two-digit string")
        else:
            module_ids.append(module_id)

        for field in ("title", "roadmap_status", "next_text_target"):
            if not _is_nonempty_string(module.get(field)):
                errors.append(f"{label}.{field}: expected a non-empty string")

        publication_status = module.get("publication_status")
        if publication_status not in publication_values:
            errors.append(
                f"{label}.publication_status: expected one of {sorted(publication_values)}"
            )
        completion_status = module.get("completion_status")
        if completion_status not in completion_values:
            errors.append(f"{label}.completion_status: expected one of {sorted(completion_values)}")

        overview = module.get("overview")
        if overview is not None:
            path_error = _relative_path_error(overview, f"{label}.overview")
            if path_error:
                errors.append(path_error)
            elif not str(overview).startswith("chapters/") or not str(overview).endswith(".md"):
                errors.append(f"{label}.overview: expected a Markdown path under chapters/")
            elif not (root / str(overview)).is_file():
                errors.append(f"{label}.overview: source does not exist: {overview}")
        elif publication_status == "published":
            errors.append(f"{label}.overview: published modules require an overview")

        target_overview = module.get("target_overview")
        path_error = _relative_path_error(target_overview, f"{label}.target_overview")
        if path_error:
            errors.append(path_error)
        elif not str(target_overview).startswith("chapters/") or not str(target_overview).endswith(
            ".md"
        ):
            errors.append(f"{label}.target_overview: expected a Markdown path under chapters/")

        sources = module.get("existing_sources")
        errors.extend(
            _validate_string_list(
                sources,
                f"{label}.existing_sources",
                allow_empty=True,
            )
        )
        if isinstance(sources, list):
            for source_index, pattern in enumerate(sources):
                source_label = f"{label}.existing_sources[{source_index}]"
                path_error = _relative_path_error(pattern, source_label)
                if path_error:
                    errors.append(path_error)
                    continue
                matches = [path for path in root.glob(str(pattern)) if path.is_file()]
                if not matches:
                    kind = "glob" if glob.has_magic(str(pattern)) else "path"
                    errors.append(f"{source_label}: {kind} matches no files: {pattern}")

        errors.extend(
            _validate_string_list(
                module.get("gaps"),
                f"{label}.gaps",
                allow_empty=True,
            )
        )

    if module_ids != EXPECTED_MODULE_IDS:
        errors.append(
            f"{PLAN_PATH}: module ids must be unique and ordered 00 through 10; found {module_ids}"
        )

    for field in ("production_toc", "migration_toc", "notebook_manifest", "debt_register"):
        value = book.get(field)
        path_error = _relative_path_error(value, f"{PLAN_PATH}: book.{field}")
        if path_error:
            errors.append(path_error)
        elif not (root / str(value)).is_file():
            errors.append(f"{PLAN_PATH}: book.{field} does not exist: {value}")
    return errors


def _validate_debt_register(plan: dict[str, Any], root: Path) -> list[str]:
    """Validate the durable debt register used to sequence book completion."""
    book = plan.get("book", {})
    raw_path = book.get("debt_register") if isinstance(book, dict) else None
    if not _is_nonempty_string(raw_path):
        return []
    path = root / str(raw_path)
    if not path.is_file():
        return []
    try:
        data = _load_json(path)
    except BookContentError as exc:
        return [str(exc)]
    label = _display_path(path, root)
    if not isinstance(data, dict):
        return [f"{label}: top-level value must be an object"]

    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append(f"{label}: schema_version must be 1")
    items = data.get("items")
    if not isinstance(items, list):
        return [*errors, f"{label}: items must be a list"]

    identifiers: list[str] = []
    for index, item in enumerate(items):
        item_label = f"{label}: items[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{item_label}: expected an object")
            continue
        for field in DEBT_REQUIRED_STRING_FIELDS:
            if not _is_nonempty_string(item.get(field)):
                errors.append(f"{item_label}.{field}: expected a non-empty string")
        identifier = item.get("id")
        if isinstance(identifier, str):
            identifiers.append(identifier)
            if not re.fullmatch(r"BOOK-\d{3}", identifier):
                errors.append(f"{item_label}.id: expected BOOK-NNN")
        if item.get("status") not in {"open", "accepted", "resolved"}:
            errors.append(f"{item_label}.status: expected open, accepted, or resolved")
        if item.get("priority") not in DEBT_PRIORITIES:
            errors.append(
                f"{item_label}.priority: expected one of {sorted(DEBT_PRIORITIES)}"
            )
        for field in DEBT_REQUIRED_LIST_FIELDS:
            errors.extend(_validate_string_list(item.get(field), f"{item_label}.{field}"))

        prioritization = item.get("prioritization")
        if not isinstance(prioritization, dict):
            errors.append(f"{item_label}.prioritization: expected an object")
            continue
        for field in DEBT_SCORE_FIELDS:
            value = prioritization.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
                errors.append(f"{item_label}.prioritization.{field}: expected an integer 1-5")

    if len(identifiers) != len(set(identifiers)):
        errors.append(f"{label}: debt item ids must be unique")
    return errors


def _yaml_scalar(value: str) -> str:
    value = re.split(r"\s+#", value.strip(), maxsplit=1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        if value[0] == '"':
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return value[1:-1]
            return decoded if isinstance(decoded, str) else str(decoded)
        return value[1:-1].replace("''", "'")
    return value


def _normalize_content_path(raw_path: str) -> str:
    path = PurePosixPath(raw_path.removeprefix("./"))
    if path.suffix in {".ipynb", ".md"}:
        path = path.with_suffix("")
    return path.as_posix()


def _toc_pages(text: str, *, production: bool) -> list[str]:
    pages: list[str] = []
    if production:
        root_match = re.search(r"(?m)^root:\s*(?P<path>[^#\n]+)", text)
        if root_match:
            pages.append(_yaml_scalar(root_match.group("path")))
    pages.extend(
        _yaml_scalar(match.group("path"))
        for match in re.finditer(
            r"(?m)^\s*-\s+file:\s*(?P<path>[^#\n]+)",
            text,
        )
    )
    return pages


def _toc_module_labels(text: str, *, production: bool) -> set[str]:
    key = "caption" if production else "title"
    labels = {
        _yaml_scalar(match.group("value"))
        for match in re.finditer(
            rf"(?m)^\s*(?:-\s+)?{key}:\s*(?P<value>Module\s+\d+\s+-\s+[^#\n]+)",
            text,
        )
    }
    return labels


def _resolve_toc_path(root: Path, raw_path: str) -> list[Path]:
    path = root / raw_path
    if PurePosixPath(raw_path).suffix in {".ipynb", ".md"}:
        return [path] if path.is_file() else []
    return [
        candidate
        for candidate in (
            path,
            root / f"{raw_path}.md",
            root / f"{raw_path}.ipynb",
        )
        if candidate.is_file()
    ]


def _first_sequence_difference(left: list[str], right: list[str]) -> str:
    for index, (left_item, right_item) in enumerate(
        zip_longest(left, right, fillvalue="<missing>"),
        start=1,
    ):
        if left_item != right_item:
            return (
                f"first difference at position {index}: "
                f"production={left_item!r}, migration={right_item!r}"
            )
    return "unknown difference"


def _validate_tocs_and_notebooks(
    plan: dict[str, Any],
    root: Path,
) -> tuple[list[str], list[str], list[str]]:
    errors: list[str] = []
    book = plan.get("book", {})
    if not isinstance(book, dict):
        return errors, [], []

    production_path = root / str(book.get("production_toc", ""))
    migration_path = root / str(book.get("migration_toc", ""))
    if not production_path.is_file() or not migration_path.is_file():
        return errors, [], []

    production_text = production_path.read_text(encoding="utf-8")
    migration_text = migration_path.read_text(encoding="utf-8")
    production_raw = _toc_pages(production_text, production=True)
    migration_raw = _toc_pages(migration_text, production=False)
    production_pages = [_normalize_content_path(path) for path in production_raw]
    migration_pages = [_normalize_content_path(path) for path in migration_raw]

    for label, raw_pages, normalized_pages in (
        ("production TOC", production_raw, production_pages),
        ("migration TOC", migration_raw, migration_pages),
    ):
        if len(normalized_pages) != len(set(normalized_pages)):
            errors.append(f"{label}: duplicate page entries are not allowed")
        for raw_page in raw_pages:
            matches = _resolve_toc_path(root, raw_page)
            if not matches:
                errors.append(f"{label}: page does not exist: {raw_page}")
            elif len(matches) > 1:
                errors.append(f"{label}: extensionless page is ambiguous: {raw_page}")

    if production_pages != migration_pages:
        errors.append(
            "TOC page parity/order mismatch: "
            + _first_sequence_difference(production_pages, migration_pages)
        )

    production_labels = _toc_module_labels(production_text, production=True)
    migration_labels = _toc_module_labels(migration_text, production=False)
    modules = plan.get("modules", [])
    if isinstance(modules, list):
        expected_labels = {
            f"Module {int(str(module.get('id', '')))} - {module.get('title', '')}"
            for module in modules
            if isinstance(module, dict)
            and module.get("publication_status") == "published"
            and str(module.get("id", "")).isdigit()
        }
        for label, actual_labels in (
            ("production TOC", production_labels),
            ("migration TOC", migration_labels),
        ):
            unexpected = sorted(actual_labels - expected_labels)
            if unexpected:
                errors.append(
                    f"{label}: module sections are not marked published in "
                    f"content/plan.json: {unexpected}"
                )
        for module in modules:
            if not isinstance(module, dict) or module.get("publication_status") != "published":
                continue
            module_id = str(module.get("id", ""))
            title = str(module.get("title", ""))
            expected_label = f"Module {int(module_id)} - {title}" if module_id.isdigit() else ""
            if expected_label not in production_labels:
                errors.append(
                    f"published module {module_id}: missing production TOC section "
                    f"{expected_label!r}"
                )
            if expected_label not in migration_labels:
                errors.append(
                    f"published module {module_id}: missing migration TOC section "
                    f"{expected_label!r}"
                )
            overview = module.get("overview")
            if isinstance(overview, str):
                normalized_overview = _normalize_content_path(overview)
                if normalized_overview not in production_pages:
                    errors.append(
                        f"published module {module_id}: overview missing from production TOC: "
                        f"{overview}"
                    )
                if normalized_overview not in migration_pages:
                    errors.append(
                        f"published module {module_id}: overview missing from migration TOC: "
                        f"{overview}"
                    )

    manifest_path = root / str(book.get("notebook_manifest", ""))
    if not manifest_path.is_file():
        return errors, production_pages, migration_pages
    try:
        manifest = _load_json(manifest_path)
    except BookContentError as exc:
        errors.append(str(exc))
        return errors, production_pages, migration_pages
    if not isinstance(manifest, dict) or not isinstance(manifest.get("published"), list):
        errors.append(f"{_display_path(manifest_path, root)}: published must be a list")
        return errors, production_pages, migration_pages

    published_raw = manifest["published"]
    if any(not _is_nonempty_string(path) for path in published_raw):
        errors.append(
            f"{_display_path(manifest_path, root)}: published paths must be non-empty strings"
        )
        return errors, production_pages, migration_pages
    published_paths = [_normalize_content_path(path) for path in published_raw]
    if len(published_paths) != len(set(published_paths)):
        errors.append(f"{_display_path(manifest_path, root)}: duplicate published paths")

    for raw_path, normalized_path in zip(published_raw, published_paths, strict=True):
        path_error = _relative_path_error(raw_path, "notebook manifest published path")
        if path_error:
            errors.append(path_error)
            continue
        if not (root / raw_path).is_file():
            errors.append(f"notebook manifest published path does not exist: {raw_path}")
        if normalized_path not in production_pages:
            errors.append(f"published notebook missing from production TOC: {raw_path}")
        if normalized_path not in migration_pages:
            errors.append(f"published notebook missing from migration TOC: {raw_path}")

    published_set = set(published_paths)
    for label, pages in (
        ("production TOC", production_pages),
        ("migration TOC", migration_pages),
    ):
        for page in pages:
            if page.startswith("notebooks/") and page not in published_set:
                errors.append(f"{label}: notebook is not published in the manifest: {page}")

    return errors, production_pages, migration_pages


def _roadmap_rows(text: str) -> list[tuple[str, str, str]]:
    lines = text.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.fullmatch(
                r"\|\s*Module\s*\|\s*Title\s*\|\s*Publication status\s*\|",
                line,
                flags=re.IGNORECASE,
            )
        ),
        None,
    )
    if header_index is None:
        return []

    rows: list[tuple[str, str, str]] = []
    for line in lines[header_index + 1 :]:
        if not line.lstrip().startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            continue
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        rows.append((cells[0], cells[1], cells[2]))
    return rows


def _validate_roadmap(plan: dict[str, Any], root: Path) -> list[str]:
    path = root / ROADMAP_PATH
    if not path.is_file():
        return [f"{ROADMAP_PATH}: missing roadmap source"]
    rows = _roadmap_rows(path.read_text(encoding="utf-8"))
    if not rows:
        return [f"{ROADMAP_PATH}: missing Module/Title/Publication status table"]

    modules = (
        [module for module in plan.get("modules", []) if isinstance(module, dict)]
        if isinstance(plan.get("modules"), list)
        else []
    )
    errors: list[str] = []
    normalized_row_ids = [f"{int(row[0]):02d}" if row[0].isdigit() else row[0] for row in rows]
    expected_ids = [str(module.get("id", "")) for module in modules]
    if normalized_row_ids != expected_ids:
        errors.append(
            f"{ROADMAP_PATH}: table module order must match content/plan.json; "
            f"found {normalized_row_ids}"
        )

    for index, module in enumerate(modules):
        if index >= len(rows):
            break
        row_id, row_title, row_status = rows[index]
        module_id = str(module.get("id", ""))
        normalized_id = f"{int(row_id):02d}" if row_id.isdigit() else row_id
        if normalized_id != module_id:
            continue
        if row_title != module.get("title"):
            errors.append(
                f"{ROADMAP_PATH}: module {module_id} title differs from content/plan.json"
            )
        if row_status != module.get("roadmap_status"):
            errors.append(
                f"{ROADMAP_PATH}: module {module_id} publication status differs from "
                "content/plan.json"
            )
    return errors


def _validate_glossary_module_names(plan: dict[str, Any], root: Path) -> list[str]:
    """Keep glossary ownership labels aligned with canonical module titles."""
    path = root / GLOSSARY_PATH
    if not path.is_file():
        return []
    modules = plan.get("modules")
    module_records = modules if isinstance(modules, list) else []
    allowed_titles = {
        str(module.get("title"))
        for module in module_records
        if isinstance(module, dict) and _is_nonempty_string(module.get("title"))
    }
    errors: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] in {"Term", "---"}:
            continue
        module_title = cells[2]
        if module_title not in allowed_titles:
            errors.append(
                f"{GLOSSARY_PATH}:{line_number}: glossary module {module_title!r} "
                f"does not match a canonical title in {PLAN_PATH}"
            )
    return errors


def _validate_citation_keys(root: Path) -> list[str]:
    """Require every MyST citation key in canonical content to resolve."""
    bibliography = root / BIBLIOGRAPHY_PATH
    if not bibliography.is_file():
        return [f"{BIBLIOGRAPHY_PATH}: missing bibliography"]
    known_keys = {
        match.group("key").strip()
        for match in re.finditer(
            r"(?m)^@\w+\{\s*(?P<key>[^,\s]+)\s*,",
            bibliography.read_text(encoding="utf-8"),
        )
    }
    errors: list[str] = []
    citation_pattern = re.compile(r"\{cite(?::[a-z]+)?\}`(?P<keys>[^`]+)`")
    for path in _canonical_source_files(root):
        if path.suffix not in {".md", ".py"}:
            continue
        text = path.read_text(encoding="utf-8")
        for match in citation_pattern.finditer(text):
            for key in re.split(r"[\s,]+", match.group("keys").strip()):
                if key and key not in known_keys:
                    errors.append(
                        f"{_display_path(path, root)}: citation key is missing from "
                        f"{BIBLIOGRAPHY_PATH}: {key}"
                    )
    return errors


def _top_level_yaml_value(text: str, key: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*(?P<value>[^#\n]+)", text)
    return _yaml_scalar(match.group("value")) if match else None


def _section_yaml_value(text: str, section: str, key: str) -> str | None:
    section_match = re.search(
        rf"(?ms)^{re.escape(section)}:\s*\n(?P<body>(?:^[ \t]+.*\n?)*)",
        text,
    )
    if not section_match:
        return None
    value_match = re.search(
        rf"(?m)^[ \t]+{re.escape(key)}:\s*(?P<value>[^#\n]+)",
        section_match.group("body"),
    )
    return _yaml_scalar(value_match.group("value")) if value_match else None


def _first_markdown_title(text: str) -> str | None:
    match = re.search(r"(?m)^#\s+(?P<title>.+?)\s*$", text)
    return match.group("title").strip() if match else None


def _validate_book_title(plan: dict[str, Any], root: Path) -> list[str]:
    book = plan.get("book", {})
    expected = book.get("title") if isinstance(book, dict) else None
    if not _is_nonempty_string(expected):
        return []
    errors: list[str] = []
    for relative_path in (Path("_config.yml"), Path("_config.outputs.yml")):
        path = root / relative_path
        if not path.is_file():
            errors.append(f"{relative_path}: missing book configuration")
            continue
        actual = _top_level_yaml_value(path.read_text(encoding="utf-8"), "title")
        if actual != expected:
            errors.append(
                f"{relative_path}: title {actual!r} does not match content plan {expected!r}"
            )

    migration_path = root / str(book.get("migration_toc", ""))
    if migration_path.is_file():
        migration_text = migration_path.read_text(encoding="utf-8")
        for section in ("project", "site"):
            actual = _section_yaml_value(migration_text, section, "title")
            if actual != expected:
                errors.append(
                    f"{_display_path(migration_path, root)}: {section}.title {actual!r} "
                    f"does not match content plan {expected!r}"
                )

    intro = root / "intro.md"
    if not intro.is_file():
        errors.append("intro.md: missing book root")
    else:
        actual = _first_markdown_title(intro.read_text(encoding="utf-8"))
        if actual != expected:
            errors.append(f"intro.md: title {actual!r} does not match content plan {expected!r}")
    return errors


def _parse_frontmatter(text: str, path: Path) -> tuple[dict[str, str], str, list[str]]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text, [f"{path}: frontmatter must start on the first line"]
    try:
        closing_index = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration:
        return {}, text, [f"{path}: frontmatter is missing its closing delimiter"]

    metadata: dict[str, str] = {}
    errors: list[str] = []
    for line_number, line in enumerate(lines[1:closing_index], start=2):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_-]*):\s*(.*)", line)
        if not match:
            errors.append(f"{path}:{line_number}: expected a scalar frontmatter field")
            continue
        key, raw_value = match.groups()
        if key in metadata:
            errors.append(f"{path}:{line_number}: duplicate frontmatter field {key!r}")
            continue
        metadata[key] = _yaml_scalar(raw_value)
    body = "\n".join(lines[closing_index + 1 :])
    return metadata, body, errors


def _validate_incoming_file(
    path: Path,
    plan: dict[str, Any],
    root: Path,
) -> list[str]:
    relative_path = Path(_display_path(path, root))
    metadata, body, errors = _parse_frontmatter(
        path.read_text(encoding="utf-8"),
        relative_path,
    )
    for field in REQUIRED_INCOMING_FIELDS:
        if not _is_nonempty_string(metadata.get(field)):
            errors.append(f"{relative_path}: missing non-empty frontmatter field {field!r}")

    raw_modules = plan.get("modules")
    modules = (
        {str(module.get("id")): module for module in raw_modules if isinstance(module, dict)}
        if isinstance(raw_modules, list)
        else {}
    )
    module_id = metadata.get("module", "")
    if module_id not in modules:
        errors.append(f"{relative_path}: unknown module {module_id!r}")

    book = plan.get("book")
    languages = _string_values(
        book.get("accepted_source_languages") if isinstance(book, dict) else None
    )
    if metadata.get("source_language") not in languages:
        errors.append(f"{relative_path}: source_language must be one of {sorted(languages)}")
    content_type = metadata.get("content_type", "")
    if content_type and not CONTENT_TYPE_PATTERN.fullmatch(content_type):
        errors.append(f"{relative_path}: content_type must be a lowercase kebab-case value")
    if metadata.get("status") and metadata.get("status") != "received":
        errors.append(f"{relative_path}: incoming status must be 'received'")
    if metadata.get("citations_status") and metadata.get("citations_status") != "needs-review":
        errors.append(f"{relative_path}: incoming citations_status must be 'needs-review'")
    if metadata.get("rights_status") and metadata.get("rights_status") != "needs-review":
        errors.append(f"{relative_path}: incoming rights_status must be 'needs-review'")
    target_error = _relative_path_error(metadata.get("target"), f"{relative_path}: target")
    if target_error:
        errors.append(target_error)
    elif PurePosixPath(metadata["target"]).suffix not in {".ipynb", ".md"}:
        errors.append(f"{relative_path}: target must be a Markdown or notebook path")

    title = _first_markdown_title(body)
    if title != metadata.get("title"):
        errors.append(
            f"{relative_path}: level-one title {title!r} must match frontmatter title "
            f"{metadata.get('title')!r}"
        )

    sections = [
        match.group("title").strip() for match in re.finditer(r"(?m)^##\s+(?P<title>.+?)\s*$", body)
    ]
    positions: list[int] = []
    for required in REQUIRED_INCOMING_SECTIONS:
        count = sections.count(required)
        if count != 1:
            errors.append(
                f"{relative_path}: required section {required!r} must appear exactly once"
            )
        elif required in sections:
            positions.append(sections.index(required))
    if len(positions) == len(REQUIRED_INCOMING_SECTIONS) and positions != sorted(positions):
        errors.append(f"{relative_path}: required sections are out of order")
    return errors


def _validate_incoming(plan: dict[str, Any], root: Path) -> tuple[list[str], int]:
    directory = root / INCOMING_DIRECTORY
    if not directory.is_dir():
        return [f"{INCOMING_DIRECTORY}: missing incoming content directory"], 0
    files = [
        path for path in sorted(directory.rglob("*.md")) if path.name.casefold() != "readme.md"
    ]
    errors: list[str] = []
    for path in files:
        errors.extend(_validate_incoming_file(path, plan, root))
    return errors, len(files)


def _canonical_source_files(root: Path) -> list[Path]:
    fixed = (
        "CONTRIBUTING.md",
        "Makefile",
        "README.md",
        "START_HERE.md",
        "_config.outputs.yml",
        "_config.yml",
        "_toc.yml",
        "myst.yml",
        "notebooks/manifest.json",
        "pyproject.toml",
    )
    files = {root / relative_path for relative_path in fixed if (root / relative_path).is_file()}
    for directory_name in ("chapters", "content", "notebooks/course"):
        directory = root / directory_name
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            if INCOMING_DIRECTORY in path.relative_to(root).parents:
                continue
            files.add(path)
    return sorted(files)


def _validate_no_legacy_notebook_path(root: Path) -> list[str]:
    errors: list[str] = []
    for path in _canonical_source_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in LEGACY_NOTEBOOK_PATH_PATTERNS):
            errors.append(
                f"{_display_path(path, root)}: canonical source references the legacy "
                "notebooks/class/ path"
            )
    return errors


def _validate_manifest_source_coverage(plan: dict[str, Any], root: Path) -> list[str]:
    """Require every manifest entry to have one canonical module owner."""
    book = plan.get("book")
    raw_manifest_path = book.get("notebook_manifest") if isinstance(book, dict) else None
    if not _is_nonempty_string(raw_manifest_path):
        return []
    manifest_path = root / str(raw_manifest_path)
    if not manifest_path.is_file():
        return []
    try:
        manifest = _load_json(manifest_path)
    except BookContentError as exc:
        return [str(exc)]
    if not isinstance(manifest, dict):
        return []

    source_owners: dict[str, set[str]] = {}
    modules = plan.get("modules")
    if isinstance(modules, list):
        for module in modules:
            if not isinstance(module, dict):
                continue
            module_id = str(module.get("id", ""))
            sources = module.get("existing_sources")
            if not isinstance(sources, list):
                continue
            for pattern in sources:
                if not isinstance(pattern, str):
                    continue
                for match in root.glob(pattern):
                    if match.is_file():
                        relative = _display_path(match, root)
                        source_owners.setdefault(relative, set()).add(module_id)

    errors: list[str] = []
    seen_sections: dict[str, str] = {}
    for section in ("published", "labs", "legacy"):
        values = manifest.get(section)
        if not isinstance(values, list):
            errors.append(
                f"{_display_path(manifest_path, root)}: {section} must be a list"
            )
            continue
        for index, raw_path in enumerate(values):
            label = (
                f"{_display_path(manifest_path, root)}: "
                f"{section}[{index}]"
            )
            if not _is_nonempty_string(raw_path):
                errors.append(f"{label}: expected a non-empty path")
                continue
            path_error = _relative_path_error(raw_path, label)
            if path_error:
                errors.append(path_error)
                continue
            if not (root / str(raw_path)).is_file():
                errors.append(f"{label}: source does not exist: {raw_path}")

            previous_section = seen_sections.get(str(raw_path))
            if previous_section is not None:
                errors.append(
                    f"{label}: source is already listed under {previous_section}: "
                    f"{raw_path}"
                )
            else:
                seen_sections[str(raw_path)] = section

            owners = sorted(source_owners.get(str(raw_path), set()))
            if not owners:
                errors.append(
                    f"{label}: source has no module owner in "
                    f"{PLAN_PATH}: {raw_path}"
                )
            elif len(owners) > 1:
                errors.append(
                    f"{label}: source has multiple module owners {owners}: {raw_path}"
                )
    return errors


def _canonical_notebook_source(root: Path, raw_path: str) -> Path:
    """Return the editable source selected by the repository authoring policy."""
    path = root / raw_path
    if path.suffix == ".ipynb":
        percent_source = path.with_suffix(".py")
        if percent_source.is_file():
            return percent_source
    return path


def _validate_lesson_semantics(plan: dict[str, Any], root: Path) -> list[str]:
    """Check promotion markers that structural notebook validation cannot prove."""
    book = plan.get("book")
    raw_manifest_path = book.get("notebook_manifest") if isinstance(book, dict) else None
    if not _is_nonempty_string(raw_manifest_path):
        return []
    manifest_path = root / str(raw_manifest_path)
    if not manifest_path.is_file():
        return []
    try:
        manifest = _load_json(manifest_path)
    except BookContentError as exc:
        return [str(exc)]
    if not isinstance(manifest, dict):
        return []

    modules_by_id = {
        str(module.get("id")): module
        for module in plan.get("modules", [])
        if isinstance(module, dict)
    }
    errors: list[str] = []
    for section in ("published", "labs", "legacy"):
        values = manifest.get(section)
        if not isinstance(values, list):
            continue
        for raw_path in values:
            if not isinstance(raw_path, str):
                continue
            source = _canonical_notebook_source(root, raw_path)
            if not source.is_file() or source.suffix not in {".md", ".py"}:
                continue
            text = source.read_text(encoding="utf-8")
            label = _display_path(source, root)
            if FORBIDDEN_CHECKPOINT_HEADING.casefold() in text.casefold():
                errors.append(f"{label}: remove the checkpoint exercise section")
            if section == "published":
                folded = text.casefold()
                for marker_name, alternatives in PUBLISHED_LESSON_MARKERS.items():
                    if not any(alternative.casefold() in folded for alternative in alternatives):
                        errors.append(
                            f"{label}: published lesson is missing a "
                            f"{marker_name} section"
                        )
                if "{cite}" not in text:
                    errors.append(
                        f"{label}: published lesson needs at least one traceable citation"
                    )
                lesson_match = re.match(r"(?P<module>\d+)\.", source.name)
                if lesson_match:
                    module_id = f"{int(lesson_match.group('module')):02d}"
                    module = modules_by_id.get(module_id)
                    title = module.get("title") if isinstance(module, dict) else None
                    if _is_nonempty_string(title):
                        expected_module_label = f"Module: {title}"
                        if expected_module_label not in text:
                            errors.append(
                                f"{label}: published lesson must declare canonical label "
                                f"{expected_module_label!r}"
                            )
            elif section == "labs" and "{cite}" not in text:
                errors.append(
                    f"{label}: source-only lab needs at least one traceable citation "
                    "before promotion"
                )
    return errors


def validate_repository(root: Path = REPO_ROOT) -> tuple[list[str], int]:
    """Return all content contract errors and the number of staged drafts."""
    try:
        plan = load_plan(root)
    except BookContentError as exc:
        return [str(exc)], 0

    errors = _validate_plan_schema(plan, root)
    errors.extend(_validate_debt_register(plan, root))
    errors.extend(_validate_manifest_source_coverage(plan, root))
    errors.extend(_validate_lesson_semantics(plan, root))
    errors.extend(_validate_tocs_and_notebooks(plan, root)[0])
    errors.extend(_validate_roadmap(plan, root))
    errors.extend(_validate_glossary_module_names(plan, root))
    errors.extend(_validate_citation_keys(root))
    errors.extend(_validate_book_title(plan, root))
    incoming_errors, incoming_count = _validate_incoming(plan, root)
    errors.extend(incoming_errors)
    errors.extend(_validate_no_legacy_notebook_path(root))
    return errors, incoming_count


def format_status(plan: dict[str, Any]) -> str:
    """Render a compact, deterministic module status report."""
    raw_modules = plan.get("modules")
    modules = (
        [module for module in raw_modules if isinstance(module, dict)]
        if isinstance(raw_modules, list)
        else []
    )
    headers = ("ID", "Publication", "Completion", "Title")
    rows = [
        (
            str(module.get("id", "")),
            str(module.get("publication_status", "")),
            str(module.get("completion_status", "")),
            str(module.get("title", "")),
        )
        for module in modules
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)) for row in rows
    )
    lines.append("")
    lines.append("Next text targets:")
    lines.extend(
        f"{module.get('id', '')}: {module.get('next_text_target', '')}" for module in modules
    )
    return "\n".join(lines)


def format_debt_status(debt: dict[str, Any]) -> str:
    """Render the open debt queue in explicit priority order."""
    raw_items = debt.get("items")
    items = (
        [
            item
            for item in raw_items
            if isinstance(item, dict) and item.get("status") == "open"
        ]
        if isinstance(raw_items, list)
        else []
    )
    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    items.sort(
        key=lambda item: (
            priority_rank.get(str(item.get("priority")), 99),
            str(item.get("id", "")),
        )
    )
    noun = "item" if len(items) == 1 else "items"
    lines = [f"Tracked open debt: {len(items)} {noun}"]
    lines.extend(
        f"{item.get('priority', '')}  {item.get('id', '')}: {item.get('title', '')}"
        for item in items
    )
    return "\n".join(lines)


def _normalize_module_id(raw_module: str) -> str:
    if not re.fullmatch(r"\d{1,2}", raw_module):
        raise BookContentError("module must be a one- or two-digit number")
    return f"{int(raw_module):02d}"


def _escape_template_scalar(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def create_incoming(
    *,
    root: Path = REPO_ROOT,
    module: str,
    slug: str,
    title: str,
    content_type: str,
    language: str,
    target: str | None = None,
    body_file: Path | None = None,
) -> Path:
    """Create one staged intake file without overwriting existing work."""
    plan = load_plan(root)
    schema_errors = _validate_plan_schema(plan, root)
    if schema_errors:
        raise BookContentError(
            "cannot create incoming content from an invalid plan: " + schema_errors[0]
        )

    module_id = _normalize_module_id(module)
    modules = {
        str(item.get("id")): item for item in plan.get("modules", []) if isinstance(item, dict)
    }
    if module_id not in modules:
        raise BookContentError(f"unknown module: {module_id}")
    if not SLUG_PATTERN.fullmatch(slug):
        raise BookContentError("slug must be lowercase kebab-case")
    if not _is_nonempty_string(title):
        raise BookContentError("title must not be empty")
    if "\n" in title or "\r" in title:
        raise BookContentError("title must fit on one line")
    if not CONTENT_TYPE_PATTERN.fullmatch(content_type):
        raise BookContentError("type must be lowercase kebab-case")

    accepted_languages = plan.get("book", {}).get("accepted_source_languages", [])
    if language not in accepted_languages:
        raise BookContentError(f"language must be one of: {', '.join(sorted(accepted_languages))}")

    resolved_target = target or modules[module_id].get("target_overview")
    target_error = _relative_path_error(resolved_target, "target")
    if target_error:
        raise BookContentError(target_error)
    if PurePosixPath(str(resolved_target)).suffix not in {".ipynb", ".md"}:
        raise BookContentError("target must be a Markdown or notebook path")

    template_path = root / INCOMING_TEMPLATE
    if not template_path.is_file():
        raise BookContentError(f"missing incoming template: {INCOMING_TEMPLATE}")
    template = template_path.read_text(encoding="utf-8")

    if body_file is None:
        draft_text = "<!-- Paste or write the received source text here. -->"
    else:
        try:
            draft_text = body_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError as exc:
            raise BookContentError(f"body file does not exist: {body_file}") from exc
        if not body_file.is_file():
            raise BookContentError(f"body file is not a regular file: {body_file}")
        if not draft_text:
            raise BookContentError(f"body file is empty: {body_file}")

    yaml_title = 'title: "{{ title }}"'
    if yaml_title not in template:
        raise BookContentError(
            f"{INCOMING_TEMPLATE}: missing required title frontmatter placeholder"
        )
    rendered = template.replace(
        yaml_title,
        f'title: "{_escape_template_scalar(title.strip())}"',
        1,
    )
    replacements = {
        "{{ title }}": title.strip(),
        "{{ module }}": module_id,
        "{{ content_type }}": content_type,
        "{{ source_language }}": language,
        "{{ target }}": _escape_template_scalar(str(resolved_target)),
        "{{ draft_text }}": draft_text,
    }
    for placeholder, value in replacements.items():
        if placeholder not in rendered:
            raise BookContentError(
                f"{INCOMING_TEMPLATE}: missing required placeholder {placeholder}"
            )
        rendered = rendered.replace(placeholder, value)

    destination_directory = root / INCOMING_DIRECTORY
    if not destination_directory.is_dir():
        raise BookContentError(f"missing incoming directory: {INCOMING_DIRECTORY}")
    destination = destination_directory / f"{module_id}-{slug}.md"
    try:
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(rendered.rstrip() + "\n")
    except FileExistsError as exc:
        raise BookContentError(f"refusing to overwrite existing file: {destination}") from exc
    return destination


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("validate", help="Validate the complete book content contract")
    subparsers.add_parser("status", help="Show module readiness and next text targets")

    new_parser = subparsers.add_parser("new", help="Create a staged incoming text file")
    new_parser.add_argument("--module", required=True, help="Module id, such as 04")
    new_parser.add_argument("--slug", required=True, help="Lowercase kebab-case file slug")
    new_parser.add_argument("--title", required=True, help="Human-readable content title")
    new_parser.add_argument("--type", dest="content_type", required=True)
    new_parser.add_argument("--language", required=True, help="Source language code")
    new_parser.add_argument(
        "--target",
        help="Planned integration path; defaults to the module target overview",
    )
    new_parser.add_argument(
        "--body-file",
        type=Path,
        help="Optional UTF-8 file whose text is inserted into the draft section",
    )
    return parser


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    """Run the book content CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    active_root = root or REPO_ROOT
    try:
        if args.command == "validate":
            errors, incoming_count = validate_repository(active_root)
            if errors:
                print("Book content validation failed:")
                for error in errors:
                    print(f"  - {error}")
                return 1
            module_count = len(load_plan(active_root).get("modules", []))
            print(
                "Book content contract is valid: "
                f"{module_count} modules, {incoming_count} incoming drafts"
            )
            return 0
        if args.command == "status":
            plan = load_plan(active_root)
            print(format_status(plan))
            print()
            print(format_debt_status(load_debt_register(plan, active_root)))
            return 0
        destination = create_incoming(
            root=active_root,
            module=args.module,
            slug=args.slug,
            title=args.title,
            content_type=args.content_type,
            language=args.language,
            target=args.target,
            body_file=args.body_file,
        )
        print(f"Created {_display_path(destination, active_root)}")
        return 0
    except BookContentError as exc:
        print(f"Book content error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
