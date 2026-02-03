# codex-multimodel-worktree

Codex skill: run the same prompt across multiple models in parallel using isolated `git worktree`s, then export per-model text reports (changed files + diffs).

## Install

```bash
python3 ~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo <owner>/<repo> \
  --path codex-multimodel-worktree
```

Restart Codex to pick up new skills.
