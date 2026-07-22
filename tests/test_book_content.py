from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.book_content import (
    BookContentError,
    create_incoming,
    format_debt_status,
    format_status,
    load_plan,
    main,
    validate_repository,
)


MODULE_TITLES = [
    "Setup and Python Ecosystem",
    "Markets, Instruments, and Data",
    "Quantitative Methods and Financial Time Series",
    "Economics, Macro, and Currency",
    "Financial Statement Analysis and Financial Modeling",
    "Corporate Issuers and Equity Valuation",
    "Fixed Income, Credit, and Term Structure",
    "Derivatives and Risk Management",
    "Alternative Investments",
    "Portfolio Management, Asset Allocation, and Performance",
    "Advanced Pathways and Capstone",
]
BOOK_TITLE = "Quantitative Financial Mathematics with Python"


def write_text(root: Path, relative_path: str, text: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(root: Path, relative_path: str, value: object) -> None:
    write_text(root, relative_path, json.dumps(value, indent=2) + "\n")


def module_record(number: int) -> dict[str, object]:
    module_id = f"{number:02d}"
    published = number == 0
    overview = "chapters/00-python-ecosystem.md" if published else None
    return {
        "id": module_id,
        "title": MODULE_TITLES[number],
        "publication_status": "published" if published else "roadmap",
        "completion_status": "substantial" if published else "not-started",
        "roadmap_status": "Published; operational baseline" if published else "Roadmap",
        "overview": overview,
        "target_overview": overview or f"chapters/{module_id}-target.md",
        "existing_sources": ["notebooks/course/0.*"] if published else [],
        "gaps": ["Keep the module contract explicit."],
        "next_text_target": f"Next useful text for module {module_id}.",
    }


def make_repository(root: Path) -> dict[str, object]:
    modules = [module_record(number) for number in range(11)]
    plan = {
        "schema_version": 1,
        "book": {
            "title": BOOK_TITLE,
            "publication_language": "en",
            "accepted_source_languages": ["en", "es"],
            "production_toc": "_toc.yml",
            "migration_toc": "myst.yml",
            "notebook_manifest": "notebooks/manifest.json",
            "debt_register": "content/debt.json",
        },
        "allowed_values": {
            "publication_status": ["published", "source-only", "roadmap"],
            "completion_status": [
                "substantial",
                "partial",
                "source-only",
                "not-started",
            ],
        },
        "modules": modules,
    }
    write_json(root, "content/plan.json", plan)
    write_json(
        root,
        "content/debt.json",
        {
            "schema_version": 1,
            "items": [
                {
                    "id": "BOOK-001",
                    "title": "Complete a test module",
                    "category": "coverage",
                    "status": "open",
                    "priority": "P1",
                    "evidence": ["content/plan.json"],
                    "affected_workflows_or_users": ["authors"],
                    "observed_symptom": "The module is incomplete.",
                    "potential_impact": "The book remains incomplete.",
                    "engineering_risk": "New sources can drift.",
                    "concrete_failure_mode": "A draft is promoted too early.",
                    "dependencies": ["both TOCs"],
                    "owner": "book editors",
                    "recommended_next_action": "Receive and review the missing text.",
                    "validation_criteria": ["The module passes the publication gate."],
                    "prioritization": {
                        "impact": 5,
                        "urgency": 4,
                        "risk": 3,
                        "confidence": 5,
                        "effort": 4,
                    },
                }
            ],
        },
    )
    write_text(
        root,
        "content/templates/incoming-text.md",
        """---
title: "{{ title }}"
module: "{{ module }}"
content_type: "{{ content_type }}"
source_language: "{{ source_language }}"
status: "received"
target: "{{ target }}"
citations_status: "needs-review"
rights_status: "needs-review"
---

# {{ title }}

## Purpose

Purpose text.

## Learning objectives

- Objective.

## Draft text

{{ draft_text }}

## Sources to verify

- Source.

## Integration notes

- Note.
""",
    )
    write_text(root, "content/incoming/README.md", "# Incoming text\n")
    write_text(root, "content/README.md", "# Book content workflow\n")
    write_text(root, "README.md", "# Repository\n")
    write_text(root, "START_HERE.md", "# Start here\n")
    write_text(root, "Makefile", "check:\n\t@true\n")
    write_text(root, "pyproject.toml", '[project]\nname = "test-book"\n')
    write_text(
        root,
        "references.bib",
        """@book{source2026,
  title = {Source},
  author = {Editor, Example},
  year = {2026},
  publisher = {Example Press}
}
""",
    )
    write_text(root, "_config.yml", f"title: {BOOK_TITLE}\n")
    write_text(root, "_config.outputs.yml", f"title: {BOOK_TITLE}\n")
    write_text(root, "intro.md", f"# {BOOK_TITLE}\n")
    write_text(root, "chapters/00-python-ecosystem.md", "# Setup and Python Ecosystem\n")
    roadmap_rows = "\n".join(
        f"| {int(module['id'])} | {module['title']} | {module['roadmap_status']} |"
        for module in modules
    )
    write_text(
        root,
        "chapters/course-roadmap.md",
        f"""# Course Roadmap

| Module | Title | Publication status |
| --- | --- | --- |
{roadmap_rows}
""",
    )
    write_text(
        root,
        "notebooks/course/0.1.reproducible_stack.md",
        """# Reproducible Stack

Module: Setup and Python Ecosystem

## Lesson summary

Summary {cite}`source2026`.

## Learning objectives

- Objective.

## Prerequisites

- Repository setup.

## Practice

Verify one observable result from this lesson.

## Handoff

Continue to the next lesson.
""",
    )
    write_json(
        root,
        "notebooks/manifest.json",
        {
            "published": ["notebooks/course/0.1.reproducible_stack.md"],
            "labs": [],
            "legacy": [],
        },
    )
    write_text(
        root,
        "_toc.yml",
        """format: jb-book
root: intro

parts:
  - caption: Course Path
    chapters:
      - file: chapters/course-roadmap
  - caption: Module 0 - Setup and Python Ecosystem
    chapters:
      - file: chapters/00-python-ecosystem
      - file: notebooks/course/0.1.reproducible_stack
""",
    )
    write_text(
        root,
        "myst.yml",
        f"""version: 1
project:
  title: {BOOK_TITLE}
  toc:
    - file: intro.md
    - title: Course Path
      children:
        - file: chapters/course-roadmap.md
    - title: Module 0 - Setup and Python Ecosystem
      children:
        - file: chapters/00-python-ecosystem.md
        - file: notebooks/course/0.1.reproducible_stack.md
site:
  title: {BOOK_TITLE}
""",
    )
    return plan


def error_text(root: Path) -> str:
    errors, _ = validate_repository(root)
    return "\n".join(errors)


def test_validate_repository_accepts_a_coherent_contract(tmp_path: Path) -> None:
    make_repository(tmp_path)

    errors, incoming_count = validate_repository(tmp_path)

    assert errors == []
    assert incoming_count == 0
    assert main(["validate"], root=tmp_path) == 0


def test_validate_detects_toc_order_manifest_and_module_drift(tmp_path: Path) -> None:
    make_repository(tmp_path)
    myst = (tmp_path / "myst.yml").read_text(encoding="utf-8")
    myst = myst.replace(
        """        - file: chapters/00-python-ecosystem.md
        - file: notebooks/course/0.1.reproducible_stack.md""",
        """        - file: notebooks/course/0.1.reproducible_stack.md
        - file: chapters/00-python-ecosystem.md""",
    ).replace(
        "Module 0 - Setup and Python Ecosystem",
        "Module 0 - Wrong title",
    )
    write_text(tmp_path, "myst.yml", myst)

    errors = error_text(tmp_path)

    assert "TOC page parity/order mismatch" in errors
    assert "missing migration TOC section" in errors
    assert "overview missing from migration TOC" not in errors


def test_validate_rejects_a_toc_module_not_marked_published(tmp_path: Path) -> None:
    make_repository(tmp_path)
    toc = (tmp_path / "_toc.yml").read_text(encoding="utf-8")
    myst = (tmp_path / "myst.yml").read_text(encoding="utf-8")
    write_text(
        tmp_path,
        "_toc.yml",
        toc
        + """
  - caption: Module 4 - Financial Statement Analysis and Financial Modeling
    chapters: []
""",
    )
    write_text(
        tmp_path,
        "myst.yml",
        myst
        + """
    - title: Module 4 - Financial Statement Analysis and Financial Modeling
      children: []
""",
    )

    errors = error_text(tmp_path)

    assert "module sections are not marked published" in errors


def test_validate_detects_schema_glob_roadmap_and_legacy_path_errors(
    tmp_path: Path,
) -> None:
    plan = make_repository(tmp_path)
    modules = plan["modules"]
    assert isinstance(modules, list)
    first_module = modules[0]
    assert isinstance(first_module, dict)
    first_module["existing_sources"] = ["notebooks/course/missing-*"]
    write_json(tmp_path, "content/plan.json", plan)
    roadmap = (tmp_path / "chapters/course-roadmap.md").read_text(encoding="utf-8")
    write_text(
        tmp_path,
        "chapters/course-roadmap.md",
        roadmap.replace("Published; operational baseline", "Published"),
    )
    write_text(tmp_path, "README.md", "Old path: notebooks/class/example.md\n")

    errors = error_text(tmp_path)

    assert "glob matches no files" in errors
    assert "module 00 publication status differs" in errors
    assert "canonical source references the legacy notebooks/class/ path" in errors


def test_validate_detects_constructed_legacy_notebook_path(tmp_path: Path) -> None:
    make_repository(tmp_path)
    write_text(
        tmp_path,
        "notebooks/course/0.2.environment_validation.py",
        'lesson_path = project_root / "notebooks" / "class"\n',
    )

    errors = error_text(tmp_path)

    assert "canonical source references the legacy notebooks/class/ path" in errors


def test_validate_requires_one_module_owner_for_every_manifest_source(
    tmp_path: Path,
) -> None:
    plan = make_repository(tmp_path)
    write_text(tmp_path, "notebooks/legacy/unassigned.ipynb", "{}\n")
    manifest = json.loads(
        (tmp_path / "notebooks/manifest.json").read_text(encoding="utf-8")
    )
    manifest["legacy"] = ["notebooks/legacy/unassigned.ipynb"]
    write_json(tmp_path, "notebooks/manifest.json", manifest)

    errors = error_text(tmp_path)

    assert "source has no module owner" in errors

    modules = plan["modules"]
    assert isinstance(modules, list)
    module_nine = modules[9]
    assert isinstance(module_nine, dict)
    module_nine["existing_sources"] = ["notebooks/legacy/unassigned.*"]
    write_json(tmp_path, "content/plan.json", plan)

    assert "source has no module owner" not in error_text(tmp_path)


def test_validate_accepts_p0_debt_priority(tmp_path: Path) -> None:
    make_repository(tmp_path)
    debt = json.loads((tmp_path / "content/debt.json").read_text(encoding="utf-8"))
    debt["items"][0]["priority"] = "P0"
    write_json(tmp_path, "content/debt.json", debt)

    assert "priority" not in error_text(tmp_path)


def test_debt_status_orders_open_p0_before_lower_priorities() -> None:
    debt = {
        "items": [
            {"id": "BOOK-002", "title": "Second", "status": "open", "priority": "P2"},
            {"id": "BOOK-001", "title": "First", "status": "open", "priority": "P0"},
            {"id": "BOOK-003", "title": "Third", "status": "open", "priority": "P3"},
        ]
    }

    status = format_debt_status(debt)

    assert status.splitlines()[1:] == [
        "P0  BOOK-001: First",
        "P2  BOOK-002: Second",
        "P3  BOOK-003: Third",
    ]


def test_validate_rejects_generic_or_incomplete_published_assessment(
    tmp_path: Path,
) -> None:
    make_repository(tmp_path)
    lesson = tmp_path / "notebooks/course/0.1.reproducible_stack.md"
    text = lesson.read_text(encoding="utf-8")
    lesson.write_text(
        text.replace("## Prerequisites", "## Prior knowledge").replace(
            "Verify one observable result from this lesson.",
            "Reproduce one result that demonstrates this objective.",
        ),
        encoding="utf-8",
    )

    errors = error_text(tmp_path)

    assert "replace the generic checkpoint" in errors
    assert "published lesson is missing a prerequisites section" in errors


def test_validate_requires_citations_in_published_lessons(tmp_path: Path) -> None:
    make_repository(tmp_path)
    lesson = tmp_path / "notebooks/course/0.1.reproducible_stack.md"
    lesson.write_text(
        lesson.read_text(encoding="utf-8").replace(" {cite}`source2026`", ""),
        encoding="utf-8",
    )

    assert "published lesson needs at least one traceable citation" in error_text(
        tmp_path
    )


def test_validate_rejects_stale_published_module_label(tmp_path: Path) -> None:
    make_repository(tmp_path)
    lesson = tmp_path / "notebooks/course/0.1.reproducible_stack.md"
    lesson.write_text(
        lesson.read_text(encoding="utf-8").replace(
            "Module: Setup and Python Ecosystem",
            "Module: Legacy Setup",
        ),
        encoding="utf-8",
    )

    assert "published lesson must declare canonical label" in error_text(tmp_path)


def test_validate_requires_citations_in_source_only_labs(tmp_path: Path) -> None:
    plan = make_repository(tmp_path)
    modules = plan["modules"]
    assert isinstance(modules, list)
    module_six = modules[6]
    assert isinstance(module_six, dict)
    module_six["existing_sources"] = ["notebooks/labs/6.1.*"]
    write_json(tmp_path, "content/plan.json", plan)
    write_text(tmp_path, "notebooks/labs/6.1.curve.py", "# %%\nvalue = 1\n")
    write_text(tmp_path, "notebooks/labs/6.1.curve.ipynb", "{}\n")
    manifest = json.loads(
        (tmp_path / "notebooks/manifest.json").read_text(encoding="utf-8")
    )
    manifest["labs"] = ["notebooks/labs/6.1.curve.ipynb"]
    write_json(tmp_path, "notebooks/manifest.json", manifest)

    errors = error_text(tmp_path)

    assert "source-only lab needs at least one traceable citation" in errors

    source = tmp_path / "notebooks/labs/6.1.curve.py"
    source.write_text(
        "# %% [markdown]\n# Method {cite}`source2026`\n",
        encoding="utf-8",
    )

    assert "source-only lab needs at least one traceable citation" not in error_text(
        tmp_path
    )


def test_validate_rejects_legacy_glossary_module_names(tmp_path: Path) -> None:
    make_repository(tmp_path)
    write_text(
        tmp_path,
        "chapters/glossary.md",
        """# Glossary

| Term | Definition | Main module |
| --- | --- | --- |
| Beta | Market sensitivity. | Portfolio theory |
""",
    )

    errors = error_text(tmp_path)

    assert "does not match a canonical title" in errors


def test_validate_requires_citation_keys_to_resolve(tmp_path: Path) -> None:
    make_repository(tmp_path)
    lesson = tmp_path / "notebooks/course/0.1.reproducible_stack.md"
    lesson.write_text(
        lesson.read_text(encoding="utf-8") + "\nClaim {cite}`missing-source`.\n",
        encoding="utf-8",
    )

    errors = error_text(tmp_path)

    assert "citation key is missing from references.bib: missing-source" in errors

    write_text(
        tmp_path,
        "references.bib",
        (
            (tmp_path / "references.bib").read_text(encoding="utf-8")
            + "@book{missing-source,\n  title = {Source},\n  year = {2026}\n}\n"
        ),
    )

    assert "citation key is missing" not in error_text(tmp_path)


def test_validate_reports_malformed_schema_without_crashing(tmp_path: Path) -> None:
    plan = make_repository(tmp_path)
    allowed_values = plan["allowed_values"]
    assert isinstance(allowed_values, dict)
    allowed_values["publication_status"] = None
    plan["modules"] = "not-a-module-list"
    write_json(tmp_path, "content/plan.json", plan)

    errors, incoming_count = validate_repository(tmp_path)
    combined = "\n".join(errors)

    assert incoming_count == 0
    assert "allowed_values.publication_status: expected a list" in combined
    assert "modules must be a list" in combined


def test_validate_rejects_incomplete_or_invalid_debt_records(tmp_path: Path) -> None:
    make_repository(tmp_path)
    debt = json.loads((tmp_path / "content/debt.json").read_text(encoding="utf-8"))
    debt["items"][0]["id"] = "invalid"
    debt["items"][0]["priority"] = "urgent"
    debt["items"][0]["prioritization"]["impact"] = 6
    debt["items"][0]["validation_criteria"] = []
    write_json(tmp_path, "content/debt.json", debt)

    errors = error_text(tmp_path)

    assert "expected BOOK-NNN" in errors
    assert "expected one of ['P0', 'P1', 'P2', 'P3']" in errors
    assert "prioritization.impact: expected an integer 1-5" in errors
    assert "validation_criteria: list must not be empty" in errors


def test_validate_checks_incoming_frontmatter_and_required_section_order(
    tmp_path: Path,
) -> None:
    make_repository(tmp_path)
    write_text(
        tmp_path,
        "content/incoming/04-invalid.md",
        """---
title: "Draft"
module: "99"
content_type: "Lesson Draft"
source_language: "fr"
status: "draft"
target: "../outside.md"
citations_status: "done"
rights_status: "unknown"
---

# Different title

## Draft text

Text.

## Purpose

Purpose.
""",
    )

    errors, incoming_count = validate_repository(tmp_path)
    combined = "\n".join(errors)

    assert incoming_count == 1
    assert "unknown module '99'" in combined
    assert "source_language must be one of" in combined
    assert "content_type must be a lowercase kebab-case value" in combined
    assert "incoming status must be 'received'" in combined
    assert "path must stay inside the repository" in combined
    assert "level-one title 'Different title' must match" in combined
    assert "required section 'Learning objectives' must appear exactly once" in combined


def test_create_incoming_renders_body_and_refuses_overwrite(tmp_path: Path) -> None:
    make_repository(tmp_path)
    body_file = tmp_path / "received.txt"
    body_file.write_text("Texto fuente en español.\n", encoding="utf-8")

    destination = create_incoming(
        root=tmp_path,
        module="4",
        slug="financial-statement-architecture",
        title='Financial "Statement" Architecture',
        content_type="lesson",
        language="es",
        body_file=body_file,
    )

    assert destination.name == "04-financial-statement-architecture.md"
    rendered = destination.read_text(encoding="utf-8")
    assert 'title: "Financial \\"Statement\\" Architecture"' in rendered
    assert 'module: "04"' in rendered
    assert 'target: "chapters/04-target.md"' in rendered
    assert "Texto fuente en español." in rendered
    errors, incoming_count = validate_repository(tmp_path)
    assert errors == []
    assert incoming_count == 1

    with pytest.raises(BookContentError, match="refusing to overwrite"):
        create_incoming(
            root=tmp_path,
            module="04",
            slug="financial-statement-architecture",
            title="Duplicate",
            content_type="lesson",
            language="es",
        )


def test_cli_status_and_new_are_testable_with_an_explicit_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    make_repository(tmp_path)

    assert main(["status"], root=tmp_path) == 0
    status_output = capsys.readouterr().out
    assert "Publication" in status_output
    assert "04" in status_output
    assert "Next text targets:" in status_output
    assert "Tracked open debt: 1 item" in status_output
    assert "P1  BOOK-001: Complete a test module" in status_output
    assert "Financial Statement Analysis and Financial Modeling" in format_status(
        load_plan(tmp_path)
    )

    assert (
        main(
            [
                "new",
                "--module",
                "04",
                "--slug",
                "accounting-identities",
                "--title",
                "Accounting Identities",
                "--type",
                "lesson",
                "--language",
                "en",
                "--target",
                "chapters/04-accounting-identities.md",
            ],
            root=tmp_path,
        )
        == 0
    )
    assert (tmp_path / "content/incoming/04-accounting-identities.md").is_file()
