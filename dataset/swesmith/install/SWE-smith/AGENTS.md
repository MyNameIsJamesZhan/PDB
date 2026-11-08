Always prefix Python commands with `uv run`

Generate bugs with: `PYTHONUNBUFFERED=1 stdbuf -oL -eL uv run modal run --detach scripts/bug_gen.py --language javascript 2>&1 | tee validation.log`
