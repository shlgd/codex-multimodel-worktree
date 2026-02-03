---
name: codex-multimodel-worktree
description: Run the same prompt across multiple Codex models in parallel using isolated git worktrees, then collect each model's proposed code changes as text reports (base snapshot + per-model diffs). Use when you want diffs/reports only (no applying changes to the current working tree).
---

# Multi-model worktree runner

Use `scripts/multimodel_worktree.py` to:
- Snapshot the current repo (tracked files only) into a single text file for LLM input.
- Run `codex exec` in parallel across N models, each in its own `git worktree`.
- Export one text report per model containing: last message, changed files, and text-only diffs.

## Prompt handling (MUST)

- Treat the user prompt as **data**, not something to interpret or improve.
- **Do not** analyze, rewrite, “fix”, summarize, translate, or add extra instructions to the prompt.
- Pass the prompt through **verbatim** (except for the runner’s unavoidable framing and/or whitespace trimming described by the script).
- If something is unclear, ask **only** about the runner inputs (models list / verify cmd / prompt source). Do not ask “clarifying questions” about the task itself unless the user explicitly requests that.

## Default behavior (IMPORTANT)

When this skill triggers in chat:
- Run the runner to generate reports.
- Do **not** apply or re-implement any proposed changes in the main working tree.
- Do **not** open/summarize diffs unless the user explicitly asks to compare models.
- Finish immediately after the runner prints the output paths (report-only completion).

## Natural-language usage (recommended)

In Codex chat, you can just invoke the skill and describe inputs in plain text. Codex will translate this into running the script.

Template:

```
Use $codex-multimodel-worktree.
Models: modelA, modelB, modelC
# or (preferred, includes effort inline):
# Models: modelA high, modelB xhigh, modelC
Verify (optional): npm run build
Prompt:
<paste your full prompt here>
```

Important: the text under `Prompt:` must be forwarded **as-is**. Do not “helpfully” edit it.

### Preflight (MUST)

Before running anything, validate the user's model list:
- Each entry must resolve to **exactly one** model id and **optional** effort.
- Effort (when present) must be one of: `low`, `medium`, `high`, `xhigh`.
- If parsing fails or is ambiguous, **do not run** the script; ask the user to restate the models list.

### Output contract (what to return, then stop)

After running, respond with only:
- The output directory **relative to the repo root** (e.g. `.codex-multimodel/<run_id>/`)
- The list of per-model results (one line each): model name, effort (if any), duration, and (if available) token usage summary from `codex exec --json` (`tokens: total=…, in=…, out=…, cached_in=…, last_in=…, ctx=…`)
- Any per-model failures called out clearly by appending `FAILED` (still one line each); do **not** print exit codes in the chat output

### Freeform / Russian-friendly shorthand (agent-parsed)

You can also describe the same fields in freeform text (including Russian). The important parts are:
- which skill to use
- which models to run
- the task prompt

Examples:

```
Используй codex-multimodel-worktree.
Модели: gpt-5.2, gpt-5.2-codex, o4-mini
Промпт:
<твоя задача>
```

Inline effort shorthand (the agent will convert this into per-model config):

```
Используй codex-multimodel-worktree.
Модели: gpt-5.2 high, gpt-5.2-codex xhigh, o4-mini medium
Промпт:
<твоя задача>
```

Effort shorthand should be passed via `--models-spec` (preferred), for example:

```bash
--models-spec "gpt-5.2 high, gpt-5.2-codex xhigh, o4-mini medium"
```

If you use `--model-config-json`, prefer flat keys (e.g. `{"gpt-5.2":{"reasoning.effort":"high"}}`), but nested objects are also accepted and will be flattened (e.g. `{"gpt-5.2":{"reasoning":{"effort":"high"}}}`).

### Duplicate models (MUST support)

If the same model is listed multiple times, run it multiple times **in one runner invocation** so the output is a **single** `.codex-multimodel/<run_id>/` folder.

Report filename rules:
- Different effort values produce different report filenames (e.g. `gpt-5.2__high.txt`, `gpt-5.2__xhigh.txt`).
- Exact duplicates are suffixed (e.g. `gpt-5.2__high__2.txt`).

## Quick start

From the target repository root:

```bash
python3 ~/.codex/skills/codex-multimodel-worktree/scripts/multimodel_worktree.py \
  --models "MODEL_A,MODEL_B,MODEL_C,MODEL_D" \
  --prompt-file prompt.txt
```

Using per-item effort (recommended):

```bash
python3 ~/.codex/skills/codex-multimodel-worktree/scripts/multimodel_worktree.py \
  --models-spec "gpt-5.2 high, gpt-5.2-codex xhigh, gpt-5.2 xhigh" \
  --prompt-file prompt.txt
```

Or from stdin:

```bash
cat prompt.txt | python3 ~/.codex/skills/codex-multimodel-worktree/scripts/multimodel_worktree.py \
  --models "MODEL_A" \
  --prompt-file -
```

Or pass the prompt directly:

```bash
python3 ~/.codex/skills/codex-multimodel-worktree/scripts/multimodel_worktree.py \
  --models "MODEL_A,MODEL_B" \
  --prompt "Do X and Y in this repo."
```

## Common options

- `--models`: repeatable and/or comma-separated; supports any number of models (duplicates allowed).
- `--models-spec`: comma-separated items with optional per-item effort (duplicates allowed).
- `--jobs`: cap parallelism (default: number of models).
- `--model-config-json`: per-model `-c` overrides as JSON (example: set reasoning effort).
- `--codex-config`: global `-c` overrides applied to every model run.
- `--verify-cmd`: run a command in each worktree after changes (default: off).
- `--keep-worktrees`: keep `.codex-worktrees/<run_id>/` for inspection (default: remove).
- `--include-codex-logs`: include full `codex exec` logs in model reports (default: off).
- `--max-file-bytes`: cap per-file inlining in `input.txt` (default: 200000).

## Outputs

By default, outputs are written to:

`<repo_root>/.codex-multimodel/<run_id>/`

Files:
- `input.txt`: run metadata + original prompt + tracked-file snapshot (size-limited).
- `<model_slug>.txt`: one per requested model run (last message + changed files + text-only diffs).

## Notes / guardrails

- Only tracked files are included in the base snapshot (`git ls-files`).
- Binary files and large files are not inlined; they are replaced with metadata placeholders.
- Worktrees are removed by default; use `--keep-worktrees` to keep them on disk.
- Test/verify commands are opt-in via `--verify-cmd`.
