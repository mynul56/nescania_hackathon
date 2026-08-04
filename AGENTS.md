# AGENTS.md — nescania_hackathon

## Project
Nascenia AI Hackathon: Bengali Medical Dialogue Generation.
Generate a Bengali doctor response from a Bengali patient prompt.

## Hard constraints (never violate)
- Inference-time model/ensemble must stay under 3B total parameters.
- Never use test.csv data for training, fine-tuning, threshold selection, or manual answer creation.
- Never manually edit generated predictions.
- Never fabricate command output, metrics, or test results — always run and show real output.
- Never modify files in data/raw/ — treat as read-only.
- Every experiment must be logged in experiments/experiment_log.csv before moving on.
- Every training/eval run must use the fixed validation split in data/processed/ — never re-split ad hoc.

## Before making changes
1. Read this file.
2. Read config/competition_info.yaml and config/config.yaml for current settings.
3. Check experiments/experiment_log.csv for what's already been tried.
4. State your plan before editing more than one file.

## Workflow order (do not skip ahead)
1. Data audit (outputs/data_audit.md, .json) — must exist and be current before anything else.
2. Retrieval baselines (src/retrieval_baselines.py) — must be scored before any model training.
3. Model research/selection — logged with rationale before training starts.
4. Training pipeline (src/train.py) — every run logged.
5. Inference + evaluation (src/infer.py, src/evaluate.py).
6. Manual error/safety review before every Kaggle submission.

## Environment
- Python 3.11, virtualenv at .venv
- requirements-local.txt = CPU/dev-only deps
- requirements.txt = full pinned deps for GPU/Kaggle training (not installed locally unless explicitly approved by team lead)

## Git
- Feature branches: feature/<short-task-name>
- Never `git add -f` on data/raw, models, checkpoints, or submissions
- Commit before large refactors
