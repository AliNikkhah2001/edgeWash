#!/usr/bin/env python3
import json
import subprocess
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


def collect_code_section() -> list[tuple[str, str, str]]:
    catalog = load_catalog()
    code_root = ROOT / "code"
    entries = []
    for folder in sorted(code_root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        meta = catalog.get(folder.name, {})
        name = meta.get("name") or folder.name
        source = meta.get("source") or git_remote(folder)
        entries.append((name, folder.name, source))
    return entries


def parse_summary(summary_path: Path) -> tuple[str, str]:
    if not summary_path.exists():
        return summary_path.parent.name, ""
    lines = [line.rstrip() for line in summary_path.read_text().splitlines()]
    non_empty = [line for line in lines if line.strip()]
    title = non_empty[0].lstrip("# ").strip() if non_empty else summary_path.parent.name

    snippet_lines = []
    seen_title = False
    for line in lines:
        text = line.strip()
        if not text:
            if seen_title and snippet_lines:
                break
            continue
        if not seen_title:
            seen_title = True
            continue
        snippet_lines.append(text)

    snippet = " ".join(snippet_lines).strip()
    return title, snippet


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
        title, snippet = parse_summary(folder / "summary.md")
        tags = parse_tags(folder / "tags.md")
        papers.append(
            {
                "title": title,
                "slug": folder.name,
                "snippet": snippet,
                "tags": tags,
            }
        )
    return papers


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
    ideas = collect_text_blocks(ROOT / "ideas")
    evaluations = collect_text_blocks(ROOT / "evaluation")
    models = collect_text_blocks(ROOT / "models")

    lines: list[str] = []
    lines.append("# Handwashing Research Hub")
    lines.append("")
    lines.append(
        "Aggregated code, papers, datasets, models, and experiment ideas for automated handwashing assessment."
    )
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
        for name, folder, source in code_entries:
            source_text = f" — source: {source}" if source else ""
            lines.append(f"- **{name}** (`code/{folder}`){source_text}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Papers")
    if papers:
        for paper in papers:
            tag_text = ", ".join(paper["tags"]) if paper["tags"] else "no tags"
            snippet = f": {paper['snippet']}" if paper["snippet"] else ""
            lines.append(f"- **{paper['title']}** (`papers/{paper['slug']}`) — tags: {tag_text}{snippet}")
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

    lines.append("## Automation")
    lines.append("- `scripts/build_readme.py` regenerates this README from folder metadata.")
    lines.append("- `.github/workflows/build-readme.yml` runs the generator on each push and commits changes.")
    lines.append("")
    lines.append("To add new assets, drop them in the appropriate folder with minimal metadata; the automation will refresh this page.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    readme_path = ROOT / "README.md"
    readme_path.write_text(build_readme())


if __name__ == "__main__":
    main()
