from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


NOTEBOOK_DIR = Path(__file__).resolve().parent
PROJECT2_ROOT = NOTEBOOK_DIR.parent
REPORT_DIR = PROJECT2_ROOT / "reports"
TRACKED_SUMMARY_PATH = NOTEBOOK_DIR / "ACCEPTANCE.md"


def execute_notebook(path: Path, *, timeout: int) -> dict[str, object]:
    notebook = nbformat.read(path, as_version=4)
    started = time.perf_counter()
    status = "passed"
    error = ""
    try:
        client = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name="project1-agent",
            resources={"metadata": {"path": str(NOTEBOOK_DIR)}},
            allow_errors=False,
        )
        client.execute()
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    elapsed = round(time.perf_counter() - started, 2)
    nbformat.write(notebook, path)
    return {
        "notebook": path.name,
        "status": status,
        "seconds": elapsed,
        "error": error,
    }


def write_summary(rows: list[dict[str, object]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "notebook_acceptance_summary.json"
    md_path = REPORT_DIR / "notebook_acceptance_summary.md"
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    passed = sum(row["status"] == "passed" for row in rows)
    lines = [
        "# Notebook 验收汇总",
        "",
        f"- 总数：{len(rows)}",
        f"- 通过：{passed}",
        f"- 失败：{len(rows) - passed}",
        "",
        "| Notebook | 状态 | 秒数 | 错误 |",
        "| --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['notebook']} | {row['status']} | {row['seconds']} | "
            f"{str(row['error']).replace('|', '/')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    TRACKED_SUMMARY_PATH.write_text(
        "\n".join(lines)
        + "\n\n> 该文件由 `execute_notebooks.py` 生成；完整执行后应为 16/16。\n",
        encoding="utf-8",
    )
    print(md_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="[0-9][0-9]_*.ipynb")
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    paths = sorted(NOTEBOOK_DIR.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No notebooks matched: {args.pattern}")
    rows = []
    for path in paths:
        print(f"Executing {path.name}...")
        row = execute_notebook(path, timeout=args.timeout)
        rows.append(row)
        print(f"  {row['status']} in {row['seconds']}s")
    write_summary(rows)
    if any(row["status"] != "passed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
