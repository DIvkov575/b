
<!-- mymemories -->
## Project memory
Persistent memory for this project lives at `/Users/divkov/workplace/mymemories/biostat`.
**On session start**, sync it: `git -C /Users/divkov/workplace/mymemories pull --ff-only` (best-effort), then read `/Users/divkov/workplace/mymemories/biostat/MEMORY.md` for the index; load individual facts on demand.

When a durable, reusable fact emerges (a user preference or correction, a hard-won decision, a non-obvious constraint), save + sync it:
`python3 ~/workplace/mymemories-tool/mymem --provider codex save <slug> --type feedback --description "<hook>" --content "<fact>"`
Be selective — do not save task status or restated context.
<!-- /mymemories -->
