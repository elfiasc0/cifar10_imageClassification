# Phase 2: CIFAR-10 architecture comparison

This starter implements three comparable classifiers (`mlp`, `cnn`, and `resnet18`), a fixed 45,000/5,000 train/validation split from the official training set, TensorBoard and CSV logs, best-validation checkpoint saving, a repeat-one-batch sanity run, and a Gradio image demo. The official 10,000-image test set is intentionally not used in Phase 2; reserve it for the final comparison in Phase 3.

## Recommended environment

Use Google Colab with a GPU for the first runs: it avoids local CUDA setup and CIFAR-10 is small enough for its standard GPU runtime. Kaggle Notebooks is also suitable, especially if you prefer keeping notebooks and datasets in a Kaggle project. A local machine works for debugging; select CUDA on an NVIDIA GPU, MPS on Apple Silicon, or CPU otherwise. Keep code in this repository and copy or sync run outputs back after cloud sessions.

## Setup and run

```bash
cd phase2
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Required sanity run: the same batch is repeated so loss should fall and accuracy rise.
python train.py --model cnn --sanity --sanity-steps 200

# First normal training run; compare other architectures with the same settings.
python train.py --model cnn --epochs 20 --seed 42
tensorboard --logdir runs

# Start demo after a normal run (replace with the printed run path).
python demo.py runs/cnn_YYYYMMDD-HHMMSS/best.pt
```

Each run writes `config.json`, `metrics.csv`, TensorBoard event files, and a checkpoint under `runs/`. In Colab, install requirements in a cell, then run these commands with `!python ...`; download or copy `runs/` before the runtime ends. Do not commit datasets, virtual environments, or large checkpoints; store the code and selected logs/checkpoints separately for submission.
