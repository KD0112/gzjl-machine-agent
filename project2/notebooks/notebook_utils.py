from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from IPython.display import Markdown, display


@dataclass
class CommandResult:
    command: str
    returncode: int
    output: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def show_markdown(text: str) -> None:
    display(Markdown(text))


def show_table(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    display(frame)
    return frame


def check(name: str, condition: bool, detail: str = "") -> dict[str, Any]:
    status = "PASS" if condition else "FAIL"
    row = {"检查项": name, "状态": status, "说明": detail}
    print(f"[{status}] {name}" + (f" | {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"{name} failed: {detail}")
    return row


def check_equal(name: str, actual: Any, expected: Any) -> dict[str, Any]:
    return check(name, actual == expected, f"actual={actual!r}, expected={expected!r}")


def run_command(
    args: Sequence[str | Path],
    *,
    cwd: Path,
    timeout: int = 180,
    env: Mapping[str, str] | None = None,
    expected_code: int = 0,
) -> CommandResult:
    command = [str(item) for item in args]
    process_env = os.environ.copy()
    if env:
        process_env.update({str(key): str(value) for key, value in env.items()})
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=process_env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout.strip()
    print(f"$ {' '.join(command)}")
    print(output)
    if completed.returncode != expected_code:
        raise AssertionError(
            f"command returned {completed.returncode}, expected {expected_code}"
        )
    return CommandResult(
        command=" ".join(command),
        returncode=completed.returncode,
        output=output,
    )


def run_unittest(
    modules: Sequence[str],
    *,
    project2_root: Path,
    quiet: bool = False,
) -> CommandResult:
    args = [sys.executable, "-m", "unittest", *modules]
    args.append("-q" if quiet else "-v")
    return run_command(args, cwd=project2_root)


def source_excerpt(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    start_index = max(0, start - 1)
    end_index = min(len(lines), end)
    excerpt = "\n".join(
        f"{index + 1:>4}: {lines[index]}"
        for index in range(start_index, end_index)
    )
    print(excerpt)
    return excerpt


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def masked_environment(names: Sequence[str]) -> list[dict[str, str]]:
    rows = []
    for name in names:
        value = os.getenv(name, "")
        rows.append(
            {
                "变量": name,
                "状态": "已配置" if value else "未配置",
                "显示值": "***" if value else "",
            }
        )
    return rows


def file_inventory(paths: Sequence[Path], root: Path) -> list[dict[str, Any]]:
    return [
        {
            "文件": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            "存在": path.exists(),
            "大小": path.stat().st_size if path.exists() and path.is_file() else None,
        }
        for path in paths
    ]
