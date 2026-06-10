# SAM3 Web Prompt Playground

Browser-based SAM3 playground for headless servers.

## What it supports

- Text prompts
- Click points (foreground/background), inference runs immediately on each click
- Box prompts by two clicks (corner A, corner B), inference runs after second click
- Class browser: one representative `good` image per MVTec class, with prev/next navigation
- Prompt state panel shows latest points/box so interactions are visible and explicit
- Combined prompting: active text + points + box are used together (not mode-exclusive)

## Run on remote host

1. Install dependencies (first time):

```bash
uv sync
```

2. Start the app on remote host:

```bash
uv run --active python scripts/sam3/special/web_prompt_playground.py --host 127.0.0.1 --port 7860
```

3. On your local machine, create SSH tunnel:

```bash
ssh -L 7860:127.0.0.1:7860 <user>@<remote-host>
```

4. Open in browser:

```text
http://127.0.0.1:7860
```

## Notes

- Default checkpoint: `weights/sam3/sam3.pt`
- Optional image arg:

```bash
uv run python scripts/sam3/special/web_prompt_playground.py /path/to/image.png
```

- If GPU is busy/OOM, force CPU mode:

```bash
uv run --active python scripts/sam3/special/web_prompt_playground.py --device cpu
```

- If no image is passed, script picks a random `good` image from `dataset/glitch_hunt_mvtec`.
