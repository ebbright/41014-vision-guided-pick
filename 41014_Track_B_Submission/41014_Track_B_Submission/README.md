# UTS 41014 – Track B: Vision-Guided Pick with a UR3

This repository contains the Track B implementation, experiment outputs, figures and generated technical report for a vision-guided UR3 pick-and-lift task in MuJoCo.

## Project folder

The complete implementation is in [`track_b_vision_pick/`](track_b_vision_pick/). See its [detailed README](track_b_vision_pick/README.md) for the system architecture, requirements mapping, calibration, estimation, control, experiment results and run instructions.

## Main contents

- `track_b_vision_pick/` – Python implementation and detailed project documentation.
- `track_b_vision_pick/results/` – calibration outputs, 130-trial experiment results, state-space analysis, figures and PDF report.

## Quick start (Windows PowerShell)

Open PowerShell in the `track_b_vision_pick` directory, install dependencies, then run:

```powershell
python -m pip install -r requirements.txt
python -X utf8 run_experiments.py --quick
```

For the full experiment, use `python -X utf8 run_experiments.py`. Generate figures with `python -X utf8 make_figures.py`. PDF report generation also requires `wkhtmltopdf`.
