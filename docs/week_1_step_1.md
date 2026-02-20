# Week 1 – Step 1: Setup + dataset acquisition (local only)

## Repo + environment
- Repo root: `C:/Users/User/Desktop/vrp-spec`
- OS: Windows, Git Bash used for commands.
- Python runs via `py` launcher.

## Dataset acquisition
- Dataset: VRPTDT benchmark (BonnTour-style time-dependent VRP benchmark).
- Stored locally under:
  - `data/raw/vrptdt/vrptdt-benchmark-main/`
- Raw dataset and generated artifacts are NOT committed to git.

## Git hygiene (important)
- `.gitignore` includes:
  - `/data/raw/`
  - `/data/processed/`
  - `/data/canonical/`
  - `/logs/`
  - `*.tar.bz2` (archives)
- Purpose: keep the GitHub repo code-only and avoid large-file limits/timeouts.

## Verification commands (used)
- Repo root check:
  - `git rev-parse --show-toplevel`
- Dataset inventory:
  - `find ... -name "*.json" | head`
