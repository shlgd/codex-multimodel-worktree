#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDE_GLOBS = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.der",
    "*.crt",
    "*.cer",
    ".npmrc",
    ".netrc",
    "*id_rsa*",
    "*id_ed25519*",
    "*secret*",
    "*secrets*",
    "*credential*",
    "*credentials*",
    "*.tfstate",
    "*.tfstate.*",
]


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        check=check,
        text=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _git(repo_root: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=repo_root, check=check)


def _repo_root(start: Path) -> Path:
    cp = _run(["git", "rev-parse", "--show-toplevel"], cwd=start)
    return Path(cp.stdout.strip())


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _iso(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _format_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours:
        return f"{hours}h{mins:02d}m{secs:02d}s"
    if mins:
        return f"{mins}m{secs:02d}s"
    return f"{secs}s"


def _slugify(s: str) -> str:
    s = s.strip()
    if not s:
        return "model"
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:80] or "model"


def _matches_any_glob(path: str, globs: list[str]) -> bool:
    p = Path(path)
    for g in globs:
        if p.match(g):
            return True
    return False


def _is_probably_binary(blob: bytes) -> bool:
    if b"\x00" in blob:
        return True
    return False


@dataclasses.dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    total_tokens: int | None = None


@dataclasses.dataclass(frozen=True)
class TokenCountInfo:
    total: TokenUsage | None = None
    last: TokenUsage | None = None
    model_context_window: int | None = None


def _to_int_or_none(v: object) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None
    return None


def _parse_token_usage_obj(obj: object) -> TokenUsage | None:
    if not isinstance(obj, dict):
        return None
    d: dict[str, object] = obj
    return TokenUsage(
        input_tokens=_to_int_or_none(d.get("input_tokens")),
        cached_input_tokens=_to_int_or_none(d.get("cached_input_tokens")),
        output_tokens=_to_int_or_none(d.get("output_tokens")),
        reasoning_output_tokens=_to_int_or_none(d.get("reasoning_output_tokens")),
        total_tokens=_to_int_or_none(d.get("total_tokens")),
    )


def _extract_token_count_info_from_codex_jsonl(jsonl_path: Path) -> TokenCountInfo | None:
    """
    `codex exec --json` prints a JSONL event stream to stdout.

    We look for the most recent `token_count` event, which contains:
      payload.info.total_token_usage
      payload.info.last_token_usage
      payload.info.model_context_window
    """
    if not jsonl_path.exists():
        return None

    last_info: TokenCountInfo | None = None
    try:
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                payload = obj.get("payload")
                if not isinstance(payload, dict):
                    continue
                if payload.get("type") != "token_count":
                    continue
                info = payload.get("info")
                if not isinstance(info, dict):
                    continue
                total = _parse_token_usage_obj(info.get("total_token_usage"))
                last = _parse_token_usage_obj(info.get("last_token_usage"))
                last_info = TokenCountInfo(
                    total=total,
                    last=last,
                    model_context_window=_to_int_or_none(info.get("model_context_window")),
                )
    except OSError:
        return None

    return last_info


def _format_token_summary(info: TokenCountInfo) -> str:
    parts: list[str] = []
    if info.total:
        if info.total.total_tokens is not None:
            parts.append(f"total={info.total.total_tokens}")
        if info.total.input_tokens is not None:
            parts.append(f"in={info.total.input_tokens}")
        if info.total.output_tokens is not None:
            parts.append(f"out={info.total.output_tokens}")
        if info.total.cached_input_tokens is not None:
            parts.append(f"cached_in={info.total.cached_input_tokens}")
    if info.last and info.last.input_tokens is not None:
        parts.append(f"last_in={info.last.input_tokens}")
    if info.model_context_window is not None:
        parts.append(f"ctx={info.model_context_window}")
    return ", ".join(parts) if parts else "unavailable"


def _token_summary_or_none(info: TokenCountInfo | None) -> str | None:
    if not info:
        return None
    s = _format_token_summary(info)
    return None if s == "unavailable" else s


@dataclasses.dataclass(frozen=True)
class RunConfig:
    repo_root: Path
    base_ref: str
    base_sha: str
    prompt: str
    prompt_hash8: str
    run_id: str
    out_dir: Path
    worktrees_dir: Path
    keep_worktrees: bool
    max_file_bytes: int
    exclude_globs: list[str]
    codex_approval_policy: str
    codex_sandbox: str
    codex_timeout_secs: int
    codex_config_kvs: list[str]
    jobs: int
    verify_cmd: str | None
    include_codex_logs: bool


@dataclasses.dataclass(frozen=True)
class ModelJob:
    model_id: str
    model_slug: str
    worktree_path: Path
    output_file: Path
    codex_config_kvs: list[str]
    run_label: str


def _build_run_id(base_sha: str, prompt_hash8: str) -> str:
    # Keep run ids short and readable while remaining reasonably unique.
    # We include the prompt hash rather than the base sha because the base sha is already
    # recorded inside `input.txt` and per-model reports.
    ts = _now_utc().strftime("%Y%m%d-%H%M%SZ")
    return f"{ts}-{prompt_hash8}"


def _tracked_files(repo_root: Path) -> list[str]:
    cp = _git(repo_root, ["ls-files", "-z"])
    raw = cp.stdout.encode("utf-8", "replace")
    # stdout is text; safe to split on \0 after re-encoding
    parts = raw.split(b"\x00")
    files: list[str] = []
    for part in parts:
        if not part:
            continue
        files.append(part.decode("utf-8", "replace"))
    return files


def _write_base_snapshot(cfg: RunConfig, *, input_file: Path) -> None:
    files = _tracked_files(cfg.repo_root)
    input_file.parent.mkdir(parents=True, exist_ok=True)

    with input_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Codex multi-model worktree run\n")
        f.write(f"run_id: {cfg.run_id}\n")
        f.write(f"repo_root: {cfg.repo_root}\n")
        f.write(f"base_ref: {cfg.base_ref}\n")
        f.write(f"base_sha: {cfg.base_sha}\n")
        f.write(f"created_at_utc: {_iso(_now_utc())}\n")
        f.write(f"models: <see model reports>\n")
        f.write(f"prompt_sha256_8: {cfg.prompt_hash8}\n")
        f.write("\n## Prompt\n")
        f.write(cfg.prompt.rstrip() + "\n")
        f.write("\n## Base snapshot (tracked files)\n")
        f.write(
            "Format:\n"
            "<<<FILE path sha256=<hex> bytes=<n> text=<true|false> included=<true|false> reason=<...>>>\n"
            "<content if included>\n"
            "<<<END FILE>>>\n\n"
        )

        for rel in files:
            if _matches_any_glob(rel, cfg.exclude_globs):
                f.write(
                    f"<<<FILE {rel} sha256=SKIPPED bytes=? text=? included=false reason=excluded_by_glob>>>\n"
                )
                f.write("<<<END FILE>>>\n\n")
                continue

            abs_path = cfg.repo_root / rel
            try:
                data = abs_path.read_bytes()
            except FileNotFoundError:
                f.write(
                    f"<<<FILE {rel} sha256=SKIPPED bytes=? text=? included=false reason=missing>>>\n"
                )
                f.write("<<<END FILE>>>\n\n")
                continue

            sha = _sha256_hex(data)
            size = len(data)
            is_bin = _is_probably_binary(data)
            if is_bin:
                f.write(
                    f"<<<FILE {rel} sha256={sha} bytes={size} text=false included=false reason=binary>>>\n"
                )
                f.write("<<<END FILE>>>\n\n")
                continue

            if size > cfg.max_file_bytes:
                f.write(
                    f"<<<FILE {rel} sha256={sha} bytes={size} text=true included=false reason=too_large>>>\n"
                )
                f.write("<<<END FILE>>>\n\n")
                continue

            text = data.decode("utf-8", errors="replace")
            f.write(f"<<<FILE {rel} sha256={sha} bytes={size} text=true included=true reason=ok>>>\n")
            f.write(text.rstrip() + "\n")
            f.write("<<<END FILE>>>\n\n")


def _make_prompt_wrapper(original_prompt: str) -> str:
    return (
        "You are running in an isolated git worktree.\n"
        "Rules:\n"
        "- Implement the user's request by editing files in this repository.\n"
        "- Do not create commits, branches, or tags.\n"
        "- Do not modify files under .codex-multimodel/ or .codex-worktrees/ (if present).\n"
        "- Avoid adding large/binary assets.\n"
        "- Finish with a concise summary of what you changed.\n"
        "\n"
        "User prompt:\n"
        + original_prompt.strip()
        + "\n"
    )


def _create_worktree(cfg: RunConfig, job: ModelJob) -> None:
    if job.worktree_path.exists():
        raise RuntimeError(f"Worktree path already exists: {job.worktree_path}")
    _git(
        cfg.repo_root,
        ["worktree", "add", "--detach", str(job.worktree_path), cfg.base_sha],
        check=True,
    )


def _toml_literal(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # Keep it simple; TOML accepts decimals.
        return repr(value)
    if isinstance(value, str):
        # json.dumps gives us a properly quoted string (TOML compatible for strings).
        return json.dumps(value)
    if isinstance(value, list):
        # Support simple arrays (strings/numbers/bools).
        return "[" + ", ".join(_toml_literal(v) for v in value) + "]"
    raise ValueError(f"Unsupported config value type: {type(value).__name__}")


def _parse_model_config_json(raw: str) -> dict[str, list[str]]:
    """
    Input format:
      {
        "model-id": {"key": "value", "other": 123},
        "another-model": {"flag": true}
      }

    Output:
      { "model-id": ["key=<toml>", "other=<toml>"], ... }
    """
    if not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("--model-config-json must be a JSON object")

    def _flatten(prefix: str, value: object, *, model_id: str, out_items: list[tuple[str, object]]) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(k, str) or not k.strip():
                    raise ValueError(f"config key for {model_id} must be a non-empty string")
                next_prefix = f"{prefix}.{k}" if prefix else k
                _flatten(next_prefix, v, model_id=model_id, out_items=out_items)
            return
        out_items.append((prefix, value))

    out: dict[str, list[str]] = {}
    for model_id, cfg in data.items():
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model id keys in --model-config-json must be non-empty strings")
        if not isinstance(cfg, dict):
            raise ValueError(f"config for {model_id} must be a JSON object")
        flat_items: list[tuple[str, object]] = []
        _flatten("", cfg, model_id=model_id, out_items=flat_items)
        kvs: list[str] = [f"{k}={_toml_literal(v)}" for k, v in flat_items]
        out[model_id] = kvs
    return out


ALLOWED_REASONING_EFFORTS: set[str] = {"low", "medium", "high", "xhigh"}


@dataclasses.dataclass(frozen=True)
class ModelRunSpec:
    model_id: str
    reasoning_effort: str | None


def _split_models_args(models_csv_or_list: list[str]) -> list[str]:
    models: list[str] = []
    for item in models_csv_or_list:
        parts = [p.strip() for p in item.split(",")] if "," in item else [item.strip()]
        for p in parts:
            if p:
                models.append(p)
    return models


def _parse_models_spec(spec: str) -> list[ModelRunSpec]:
    """
    Parse user-friendly model lists like:
      "gpt-5.2 high, gpt-5.2-codex xhigh, o4-mini"

    Accepted per-item forms:
      - "<model>"
      - "<model> <effort>"
      - "<model>:<effort>" / "<model>@<effort>" / "<model>=<effort>"
      - "<model>(<effort>)"
    """
    spec = spec.strip()
    if not spec:
        return []

    runs: list[ModelRunSpec] = []
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue

        model_id: str | None = None
        effort: str | None = None

        m = re.match(r"^(?P<model>.+?)\\((?P<effort>[^)]+)\\)$", item)
        if m:
            model_id = m.group("model").strip()
            effort = m.group("effort").strip()
        elif (" " not in item) and any(sep in item for sep in (":", "@", "=")):
            for sep in (":", "@", "="):
                if sep in item:
                    left, right = item.split(sep, 1)
                    model_id = left.strip()
                    effort = right.strip()
                    break
        else:
            parts = item.split()
            if len(parts) == 1:
                model_id = parts[0].strip()
                effort = None
            elif len(parts) == 2:
                model_id = parts[0].strip()
                effort = parts[1].strip()
            else:
                raise ValueError(
                    f"Invalid models-spec item: {item!r}. Expected '<model>' or '<model> <effort>'."
                )

        if not model_id:
            raise ValueError(f"Invalid models-spec item: {item!r}. Model id is empty.")

        if effort is not None:
            effort = effort.strip()
            if not effort:
                raise ValueError(f"Invalid models-spec item: {item!r}. Effort is empty.")
            if effort not in ALLOWED_REASONING_EFFORTS:
                allowed = ", ".join(sorted(ALLOWED_REASONING_EFFORTS))
                raise ValueError(
                    f"Invalid reasoning effort {effort!r} for model {model_id!r}. Allowed: {allowed}."
                )

        runs.append(ModelRunSpec(model_id=model_id, reasoning_effort=effort))

    return runs


def _kvs_from_effort(effort: str | None) -> list[str]:
    if not effort:
        return []
    return [f"reasoning.effort={_toml_literal(effort)}"]


def _build_unique_job_slugs(
    *,
    runs: list[ModelRunSpec],
    model_cfg_map: dict[str, list[str]],
) -> list[tuple[ModelRunSpec, str, list[str], str]]:
    """
    Returns tuples:
      (run_spec, unique_slug, per_run_kvs, run_label)
    """
    slugs: list[str] = []
    labels: list[str] = []
    kvs_list: list[list[str]] = []

    for r in runs:
        base = _slugify(r.model_id)

        label_effort = r.reasoning_effort
        if label_effort is None:
            for kv in model_cfg_map.get(r.model_id, []):
                if kv.startswith("reasoning.effort="):
                    tail = kv.split("=", 1)[1].strip()
                    label_effort = tail.strip("\"'")
                    break

        slug = base
        label = r.model_id
        if label_effort:
            slug = f"{base}__{_slugify(label_effort)}"
            label = f"{r.model_id} ({label_effort})"

        per_run_kvs = list(model_cfg_map.get(r.model_id, [])) + _kvs_from_effort(r.reasoning_effort)

        slugs.append(slug)
        labels.append(label)
        kvs_list.append(per_run_kvs)

    seen: dict[str, int] = {}
    unique_slugs: list[str] = []
    unique_labels: list[str] = []
    for slug, label in zip(slugs, labels, strict=True):
        n = seen.get(slug, 0) + 1
        seen[slug] = n
        if n == 1:
            unique_slugs.append(slug)
            unique_labels.append(label)
        else:
            unique_slugs.append(f"{slug}__{n}")
            unique_labels.append(f"{label} #{n}")

    out: list[tuple[ModelRunSpec, str, list[str], str]] = []
    for r, slug, kvs, label in zip(runs, unique_slugs, kvs_list, unique_labels, strict=True):
        out.append((r, slug, kvs, label))
    return out


async def _run_one_model(
    cfg: RunConfig,
    job: ModelJob,
) -> tuple[int, int, TokenCountInfo | None, str]:
    """
    Returns: (exit_code, duration_seconds, token_info, combined_log_text)
    """
    started = _now_utc()

    if not job.worktree_path.exists():
        raise RuntimeError(f"Worktree path missing (expected pre-created): {job.worktree_path}")

    codex_last = job.worktree_path / ".codex_last_message.txt"
    codex_jsonl = job.worktree_path / ".codex_exec.jsonl"
    prompt_wrapped = _make_prompt_wrapper(cfg.prompt)
    codex_cmd = [
        "codex",
        "-a",
        cfg.codex_approval_policy,
        "exec",
        "-s",
        cfg.codex_sandbox,
        "--color",
        "never",
        "--json",
        "-C",
        str(job.worktree_path),
        "-m",
        job.model_id,
        "-o",
        str(codex_last),
        *sum((["-c", kv] for kv in (cfg.codex_config_kvs + job.codex_config_kvs)), []),
        prompt_wrapped,
    ]

    stderr_b: bytes = b""
    with codex_jsonl.open("wb") as stdout_f:
        proc = await asyncio.create_subprocess_exec(
            *codex_cmd,
            stdout=stdout_f,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _stdout_ignored, stderr_b = await asyncio.wait_for(
                proc.communicate(), timeout=cfg.codex_timeout_secs
            )
        except asyncio.TimeoutError:
            proc.kill()
            _stdout_ignored, stderr_b = await proc.communicate()
            exit_code = 124
        else:
            exit_code = proc.returncode or 0

    ended = _now_utc()
    duration_s = int((ended - started).total_seconds())

    codex_stderr = stderr_b.decode("utf-8", errors="replace")
    last_message = ""
    if codex_last.exists():
        last_message = codex_last.read_text(encoding="utf-8", errors="replace").strip()

    token_info = _extract_token_count_info_from_codex_jsonl(codex_jsonl)
    codex_stdout = ""
    if cfg.include_codex_logs:
        try:
            codex_stdout = codex_jsonl.read_text(encoding="utf-8", errors="replace")
        except OSError:
            codex_stdout = ""

    # Optional verify step
    verify_block = ""
    if cfg.verify_cmd:
        try:
            verify_cp = _run(
                ["bash", "-lc", cfg.verify_cmd],
                cwd=job.worktree_path,
                check=False,
                text=True,
            )
            verify_block = (
                "\n## Verify command\n"
                f"cmd: {cfg.verify_cmd}\n"
                f"exit_code: {verify_cp.returncode}\n"
                "stdout:\n"
                + verify_cp.stdout
                + "\n"
                "stderr:\n"
                + verify_cp.stderr
                + "\n"
            )
        except Exception as e:  # pragma: no cover
            verify_block = f"\n## Verify command\ncmd: {cfg.verify_cmd}\nerror: {e}\n"

    # Collect changed files + diffs (text-only).
    #
    # IMPORTANT: Models are instructed not to create commits, so changes typically live in the
    # working tree (and occasionally the index). Using "<base>...HEAD" would miss those entirely.
    diff_cached = _git(
        job.worktree_path,
        ["diff", "--name-status", "--cached", cfg.base_sha],
        check=False,
    ).stdout
    diff_worktree = _git(job.worktree_path, ["diff", "--name-status", cfg.base_sha], check=False).stdout
    untracked = _git(
        job.worktree_path,
        ["ls-files", "--others", "--exclude-standard"],
        check=False,
    ).stdout

    name_status_lines: list[str] = []
    for block in (diff_cached, diff_worktree):
        for line in block.splitlines():
            if line.strip():
                name_status_lines.append(line)
    # Ignore artifacts created by this runner itself.
    ignored_untracked = {".codex_last_message.txt", ".codex_exec.jsonl"}

    for rel in untracked.splitlines():
        rel = rel.strip()
        if rel and rel not in ignored_untracked:
            name_status_lines.append(f"??\t{rel}")

    # De-dup exact lines while preserving order for readability.
    seen_lines: set[str] = set()
    name_status_unique: list[str] = []
    for line in name_status_lines:
        if line in seen_lines:
            continue
        seen_lines.add(line)
        name_status_unique.append(line)

    name_status = "\n".join(name_status_unique)

    # Extract file paths from `--name-status` output.
    # Formats:
    #   M\tpath
    #   A\tpath
    #   D\tpath
    #   R100\told\tnew
    #   ??\tpath  (synthetic, for untracked files)
    changed_files: list[str] = []
    untracked_set = {
        p.strip() for p in untracked.splitlines() if p.strip() and p.strip() not in ignored_untracked
    }
    seen_files: set[str] = set()
    for line in name_status_unique:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0]
        rel = parts[-1] if (status.startswith("R") or status.startswith("C")) and len(parts) >= 3 else parts[1]
        if not rel or rel in seen_files:
            continue
        seen_files.add(rel)
        changed_files.append(rel)

    omitted_changes: list[str] = []
    diffs: list[tuple[str, str]] = []
    for rel in changed_files:
        # Ask git for per-file diff; it will emit "Binary files" or "GIT binary patch" for binaries.
        if rel in untracked_set:
            diff_text = _git(
                job.worktree_path,
                ["diff", "--no-color", "--unified=3", "--no-index", "--", "/dev/null", rel],
                check=False,
            ).stdout
        else:
            diff_text = _git(
                job.worktree_path,
                ["diff", "--no-color", "--unified=3", cfg.base_sha, "--", rel],
                check=False,
            ).stdout
            if not diff_text.strip():
                diff_text = _git(
                    job.worktree_path,
                    ["diff", "--cached", "--no-color", "--unified=3", cfg.base_sha, "--", rel],
                    check=False,
                ).stdout
        if "GIT binary patch" in diff_text or "\nBinary files " in diff_text or diff_text.startswith("Binary files "):
            omitted_changes.append(f"{rel}: binary_diff_omitted")
            continue
        if not diff_text.strip():
            continue
        diffs.append((rel, diff_text.rstrip()))

    # Write model report (single file per model)
    job.output_file.parent.mkdir(parents=True, exist_ok=True)
    with job.output_file.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# Codex multi-model proposal\n")
        f.write(f"run_id: {cfg.run_id}\n")
        f.write(f"model_id: {job.model_id}\n")
        f.write(f"model_slug: {job.model_slug}\n")
        f.write(f"base_sha: {cfg.base_sha}\n")
        f.write(f"worktree: {job.worktree_path}\n")
        f.write(f"codex_config: {cfg.codex_config_kvs}\n")
        f.write(f"model_config: {job.codex_config_kvs}\n")
        f.write(f"started_at_utc: {_iso(started)}\n")
        f.write(f"ended_at_utc: {_iso(ended)}\n")
        f.write(f"duration_seconds: {duration_s}\n")
        f.write(f"exit_code: {exit_code}\n")
        token_summary = _token_summary_or_none(token_info)
        if token_summary or cfg.include_codex_logs:
            f.write("\n## Token usage\n")
            f.write((token_summary or "unavailable") + "\n")
        f.write("\n## Codex last message\n")
        f.write(last_message + "\n")
        if cfg.include_codex_logs or exit_code != 0:
            f.write("\n## codex exec stderr\n")
            f.write(codex_stderr.rstrip() + "\n")
        if cfg.include_codex_logs:
            f.write("\n## codex exec stdout (JSONL)\n")
            f.write(codex_stdout.rstrip() + "\n")
        if verify_block:
            f.write(verify_block)

        f.write("\n## Changed files (git diff --name-status)\n")
        f.write(name_status.rstrip() + "\n")

        f.write("\n## Diffs (text only)\n")
        for rel, d in diffs:
            f.write(f"<<<DIFF {rel}>>>\n")
            f.write(d + "\n")
            f.write("<<<END DIFF>>>\n\n")

        if omitted_changes:
            f.write("\n## Omitted changes\n")
            for line in omitted_changes:
                f.write(f"- {line}\n")

    combined_log = (
        f"[{job.model_id}] exit_code={exit_code} duration_s={duration_s}\n"
        f"stdout:\n{codex_stdout}\n"
        f"stderr:\n{codex_stderr}\n"
    )
    return exit_code, duration_s, token_info, combined_log


def _parse_models(models_csv_or_list: list[str]) -> list[str]:
    # Back-compat helper. Intentionally keeps duplicates (no de-dupe).
    return _split_models_args(models_csv_or_list)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one prompt across N models in parallel using git worktrees; collect per-model diffs.",
    )
    p.add_argument(
        "--models",
        action="append",
        default=[],
        help="Comma-separated models or repeatable. Example: --models \"o3,o4-mini\" --models \"codex-mini\"",
    )
    p.add_argument(
        "--models-spec",
        default="",
        help="User-friendly model list with optional per-item reasoning effort, e.g. \"gpt-5.2 high, gpt-5.2-codex xhigh\".",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="Prompt text to run.")
    g.add_argument("--prompt-file", help="Read prompt from file. Use '-' for stdin.")
    p.add_argument("--base-ref", default="HEAD", help="Git ref to base worktrees on (default: HEAD).")
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: <repo>/.codex-multimodel/<run_id>/",
    )
    p.add_argument(
        "--worktrees-dir",
        default="",
        help="Worktrees base directory. Default: <repo>/.codex-worktrees/<run_id>/",
    )
    p.add_argument("--keep-worktrees", action="store_true", help="Keep worktrees on disk (default: remove).")
    p.add_argument(
        "--max-file-bytes",
        type=int,
        default=200_000,
        help="Max bytes to inline per tracked file in input.txt (default: 200000).",
    )
    p.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Exclude tracked files from base snapshot by glob (repeatable).",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Parallel jobs (default: number of models).",
    )
    p.add_argument(
        "--codex-timeout-secs",
        type=int,
        default=1800,
        help="Per-model timeout for codex exec (default: 1800).",
    )
    p.add_argument(
        "--codex-config",
        action="append",
        default=[],
        help="Repeatable codex config override (key=value) passed as -c to every model run.",
    )
    p.add_argument(
        "--model-config-json",
        default="",
        help="JSON mapping model_id -> {key: value} for per-model -c overrides.",
    )
    p.add_argument(
        "--verify-cmd",
        default="",
        help="Optional shell command to run in each worktree after codex exec (default: none).",
    )
    p.add_argument(
        "--include-codex-logs",
        action="store_true",
        help="Include full codex exec stdout/stderr in per-model reports (default: only last message; stderr only on failure).",
    )
    p.add_argument(
        "--validate-models-only",
        action="store_true",
        help="Validate model parsing/config and exit before creating output dirs or running codex exec.",
    )
    p.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow running even if the main repo has uncommitted changes.",
    )
    return p.parse_args(argv)


async def _main_async(ns: argparse.Namespace) -> int:
    start_dir = Path.cwd()
    repo_root = _repo_root(start_dir)

    if not ns.models and not ns.models_spec.strip():
        print("No models provided. Use --models or --models-spec.", file=sys.stderr)
        return 2

    if not ns.allow_dirty:
        # Ignore untracked files by default; they don't affect tracked-file snapshots or diffs.
        dirty = _git(repo_root, ["status", "--porcelain", "--untracked-files=no"], check=True).stdout.strip()
        if dirty:
            print("Refusing to run: repository has uncommitted changes. Use --allow-dirty to override.", file=sys.stderr)
            return 2

    base_sha = _git(repo_root, ["rev-parse", ns.base_ref], check=True).stdout.strip()

    if ns.prompt_file:
        if ns.prompt_file == "-":
            prompt = sys.stdin.read()
        else:
            prompt = Path(ns.prompt_file).read_text(encoding="utf-8", errors="replace")
    else:
        prompt = ns.prompt or ""

    model_cfg_map = _parse_model_config_json(ns.model_config_json) if ns.model_config_json else {}

    try:
        if ns.models_spec.strip():
            runs = _parse_models_spec(ns.models_spec)
        else:
            models = _parse_models(ns.models)
            runs = [ModelRunSpec(model_id=m, reasoning_effort=None) for m in models]
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    if not runs:
        print("No models provided after parsing.", file=sys.stderr)
        return 2

    if ns.validate_models_only:
        planned = _build_unique_job_slugs(runs=runs, model_cfg_map=model_cfg_map)
        for _run, slug, _kvs, label in planned:
            print(f"{label} -> {slug}.txt")
        return 0

    prompt_hash8 = _sha256_hex(prompt.encode("utf-8", "replace"))[:8]
    run_id = _build_run_id(base_sha, prompt_hash8)

    out_dir = Path(ns.out_dir) if ns.out_dir else (repo_root / ".codex-multimodel" / run_id)
    worktrees_dir = Path(ns.worktrees_dir) if ns.worktrees_dir else (repo_root / ".codex-worktrees" / run_id)

    jobs = ns.jobs if ns.jobs and ns.jobs > 0 else len(runs)
    verify_cmd = ns.verify_cmd.strip() or None

    cfg = RunConfig(
        repo_root=repo_root,
        base_ref=str(ns.base_ref),
        base_sha=base_sha,
        prompt=prompt,
        prompt_hash8=prompt_hash8,
        run_id=run_id,
        out_dir=out_dir,
        worktrees_dir=worktrees_dir,
        keep_worktrees=bool(ns.keep_worktrees),
        max_file_bytes=int(ns.max_file_bytes),
        exclude_globs=DEFAULT_EXCLUDE_GLOBS + list(ns.exclude_glob),
        codex_approval_policy="never",
        codex_sandbox="workspace-write",
        codex_timeout_secs=int(ns.codex_timeout_secs),
        codex_config_kvs=list(ns.codex_config),
        jobs=jobs,
        verify_cmd=verify_cmd,
        include_codex_logs=bool(ns.include_codex_logs),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    worktrees_dir.mkdir(parents=True, exist_ok=True)

    input_file = out_dir / "input.txt"
    _write_base_snapshot(cfg, input_file=input_file)

    # Build jobs
    model_jobs: list[ModelJob] = []
    planned = _build_unique_job_slugs(runs=runs, model_cfg_map=model_cfg_map)
    for run, slug, per_run_kvs, label in planned:
        worktree_path = worktrees_dir / f"{slug}__{cfg.prompt_hash8}"
        output_file = out_dir / f"{slug}.txt"
        model_jobs.append(
            ModelJob(
                model_id=run.model_id,
                model_slug=slug,
                worktree_path=worktree_path,
                output_file=output_file,
                codex_config_kvs=per_run_kvs,
                run_label=label,
            )
        )

    # Create worktrees serially (git takes a lock).
    for job in model_jobs:
        _create_worktree(cfg, job)

    sem = asyncio.Semaphore(cfg.jobs)

    async def _guarded(job: ModelJob) -> tuple[ModelJob, int, int, TokenCountInfo | None, str, str | None]:
        async with sem:
            try:
                code, duration_s, token_info, log = await _run_one_model(cfg, job)
                return job, code, duration_s, token_info, log, None
            except Exception as e:
                return job, 1, 0, None, "", f"{type(e).__name__}: {e}"

    results = await asyncio.gather(*[_guarded(j) for j in model_jobs])

    summary_lines: list[str] = []
    failed = 0
    for res in results:
        job, code, duration_s, token_info, _log, err = res
        if code != 0:
            failed += 1
        line = f"{job.run_label} {_format_duration(duration_s)}"
        token_summary = _token_summary_or_none(token_info)
        if token_summary:
            line += f" | tokens: {token_summary}"
        if code != 0:
            line += " FAILED"
            if err:
                print(f"{job.run_label} error: {err}", file=sys.stderr)
        summary_lines.append(line)

    # Cleanup
    if not cfg.keep_worktrees:
        for job in model_jobs:
            try:
                _git(cfg.repo_root, ["worktree", "remove", "-f", str(job.worktree_path)], check=False)
            except Exception:
                pass

    # Print minimal, relative output (friendlier for chat copying).
    try:
        out_dir_rel = cfg.out_dir.relative_to(cfg.repo_root)
        out_dir_str = out_dir_rel.as_posix()
    except ValueError:
        out_dir_str = str(cfg.out_dir)
    if not out_dir_str.endswith("/"):
        out_dir_str += "/"
    print(out_dir_str)
    for line in summary_lines:
        print(line)

    return 1 if failed else 0


def main(argv: list[str]) -> int:
    ns = _parse_args(argv)
    try:
        return asyncio.run(_main_async(ns))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
