#!/usr/bin/env python3
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_catalog() -> dict:
    catalog_path = ROOT / "code" / "catalog.json"
    if not catalog_path.exists():
        return {}
    try:
        data = json.loads(catalog_path.read_text())
    except json.JSONDecodeError:
        return {}
    return {entry.get("id") or entry.get("name"): entry for entry in data if isinstance(entry, dict)}


def git_remote(path: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(path), "config", "--get", "remote.origin.url"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except subprocess.CalledProcessError:
        return ""


def collect_code_section() -> list[dict]:
    catalog = load_catalog()
    code_root = ROOT / "code"
    entries = []
    for folder in sorted(code_root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        meta = catalog.get(folder.name, {})
        title, snippet, status = parse_summary(folder / "summary.md")
        status = load_status(folder, status)
        tags = parse_tags(folder / "tags.md")
        name = title or meta.get("name") or folder.name
        source = meta.get("source") or git_remote(folder)
        entries.append(
            {
                "title": name,
                "slug": folder.name,
                "source": source,
                "snippet": snippet,
                "tags": tags,
                "status": status,
            }
        )
    return entries


CHECKBOX_RE = re.compile(r"^[#>*-]?\s*\[([ xX])\]\s*(.*)$")


def extract_status(lines: list[str]) -> str:
    for line in lines:
        match = CHECKBOX_RE.match(line.strip())
        if match:
            return "[x]" if match.group(1).lower() == "x" else "[ ]"
    return ""


def parse_summary(summary_path: Path) -> tuple[str, str, str]:
    if not summary_path.exists():
        return summary_path.parent.name, "", ""

    lines = [line.rstrip() for line in summary_path.read_text().splitlines()]
    status = extract_status(lines)

    title = summary_path.parent.name
    snippet_lines: list[str] = []
    seen_title = False
    for line in lines:
        text = line.strip()
        if not text:
            if seen_title and snippet_lines:
                break
            continue
        if CHECKBOX_RE.match(text):
            continue
        if text.startswith("#"):
            text = text.lstrip("#").strip()
        if not seen_title:
            title = text or title
            seen_title = True
            continue
        snippet_lines.append(text)

    snippet = " ".join(snippet_lines).strip()
    return title, snippet, status


def load_status(folder: Path, fallback: str = "") -> str:
    status_path = folder / "status.md"
    if status_path.exists():
        status = extract_status(status_path.read_text().splitlines())
        if status:
            return status
    return fallback


def parse_tags(tags_path: Path) -> list[str]:
    if not tags_path.exists():
        return []
    return [line.strip() for line in tags_path.read_text().splitlines() if line.strip()]


def collect_papers_section() -> list[dict]:
    papers_root = ROOT / "papers"
    papers = []
    for folder in sorted(papers_root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        title, snippet, status = parse_summary(folder / "summary.md")
        status = load_status(folder, status)
        tags = parse_tags(folder / "tags.md")
        papers.append(
            {
                "title": title,
                "slug": folder.name,
                "snippet": snippet,
                "tags": tags,
                "status": status,
            }
        )
    return papers


def collect_datasets_section() -> list[dict]:
    datasets_root = ROOT / "datasets"
    datasets = []
    for folder in sorted(datasets_root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        title, snippet, status = parse_summary(folder / "summary.md")
        status = load_status(folder, status)
        tags = parse_tags(folder / "tags.md")
        datasets.append(
            {
                "title": title,
                "slug": folder.name,
                "snippet": snippet,
                "tags": tags,
                "status": status,
            }
        )
    return datasets


def collect_text_blocks(base: Path) -> list[str]:
    blocks = []
    if not base.exists():
        return blocks
    for path in sorted(base.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        title = path.stem.replace("-", " ").title()
        rel_path = path.relative_to(ROOT)
        blocks.append(f"{title} ({rel_path})")
    return blocks


def build_readme() -> str:
    code_entries = collect_code_section()
    papers = collect_papers_section()
    datasets = collect_datasets_section()
    ideas = collect_text_blocks(ROOT / "ideas")
    evaluations = collect_text_blocks(ROOT / "evaluation")
    models = collect_text_blocks(ROOT / "models")

    lines: list[str] = []
    lines.append("# Handwashing Research Hub")
    lines.append("")
    lines.append(
        "Aggregated code, papers, datasets, models, and experiment ideas for automated handwashing assessment."
    )
    lines.append(f"_Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M %Z')}_")
    lines.append("")
    lines.append("## Structure")
    lines.append("- `code/`: cloned codebases and pipelines")
    lines.append("- `papers/`: papers with `summary.md`, `tags.md`, and `paper.pdf`")
    lines.append("- `datasets/`: storage location for raw/processed datasets (gitignored)")
    lines.append("- `models/`: exported weights and model cards")
    lines.append("- `evaluation/`: benchmarks and result artifacts")
    lines.append("- `ideas/`: future experiment notes and design sketches")
    lines.append("")

    lines.append("## Codebases")
    if code_entries:
        for entry in code_entries:
            tag_text = ", ".join(entry["tags"]) if entry["tags"] else "no tags"
            source_text = f" — source: {entry['source']}" if entry["source"] else ""
            snippet = f": {entry['snippet']}" if entry["snippet"] else ""
            status = entry["status"] or "[ ]"
            lines.append(
                f"- {status} **{entry['title']}** (`code/{entry['slug']}`) — tags: {tag_text}{source_text}{snippet}"
            )
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Papers")
    if papers:
        for paper in papers:
            tag_text = ", ".join(paper["tags"]) if paper["tags"] else "no tags"
            snippet = f": {paper['snippet']}" if paper["snippet"] else ""
            status = paper["status"] or "[ ]"
            lines.append(
                f"- {status} **{paper['title']}** (`papers/{paper['slug']}`) — tags: {tag_text}{snippet}"
            )
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Datasets")
    if datasets:
        for ds in datasets:
            tag_text = ", ".join(ds["tags"]) if ds["tags"] else "no tags"
            snippet = f": {ds['snippet']}" if ds["snippet"] else ""
            status = ds["status"] or "[ ]"
            lines.append(
                f"- {status} **{ds['title']}** (`datasets/{ds['slug']}`) — tags: {tag_text}{snippet}"
            )
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Models")
    if models:
        for entry in models:
            lines.append(f"- {entry}")
    else:
        lines.append("- Staging area for trained weights and model cards.")
    lines.append("")

    lines.append("## Evaluation")
    if evaluations:
        for entry in evaluations:
            lines.append(f"- {entry}")
    else:
        lines.append("- Add benchmark summaries or result notebooks here.")
    lines.append("")

    lines.append("## Ideas")
    if ideas:
        for entry in ideas:
            lines.append(f"- {entry}")
    else:
        lines.append("- Track hypotheses and method ideas in this folder.")
    lines.append("")

    todo_path = ROOT / "ideas" / "todo.md"
    if todo_path.exists():
        lines.append("## TODOs")
        lines.extend(todo_path.read_text().splitlines())
        lines.append("")

    lines.append("## Automation")
    lines.append("- `scripts/build_readme.py` regenerates this README from folder metadata.")
    lines.append("- `.github/workflows/build-readme.yml` runs the generator on each push and commits changes.")
    lines.append("- `.github/workflows/pages.yml` builds GitHub Pages from the generated docs.")
    lines.append("")
    lines.append("To add new assets, drop them in the appropriate folder with minimal metadata; the automation will refresh this page.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    readme_path = ROOT / "README.md"
    content = build_readme()
    readme_path.write_text(content)
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    docs_path = docs_dir / "index.md"
    docs_path.write_text("---\nlayout: default\n---\n\n" + content)


if __name__ == "__main__":
    main()
