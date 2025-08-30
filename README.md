
# Refactored Project

## Changes
- Config moved to **configs/config.yaml** (converted from JSON).
- Keras training separated into **trainers/keras_trainer.py**.
- Optuna orchestration separated into **optimizers/optuna_optimizer.py**.
- `main.py` split into:
  - **pipeline/data_pipeline.py** — data loading & preprocessing
  - **pipeline/model_pipeline.py** — model training orchestration
  - **bin/run_pipeline.py** — entry point

## How to run
```bash
python bin/run_pipeline.py  # uses configs/config.yaml by default
```

To use Optuna, construct an `objective(trial)` and call:
```python
from optimizers.optuna_optimizer import run_study
study = run_study(objective, n_trials=20, study_name="my_study")
```
