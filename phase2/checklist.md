# Phase 2 Status: CIFAR-10 Architecture Comparison

This report maps the Phase 2 requirements in the course project brief to the current implementation and work completed so far. It is intended as a progress summary for the project group; the instructor determines final credit.

## Requirement Status

| Requirement | Status | Evidence |
|---|---|---|
| Base model and training code | Implemented | `train.py` supports MLP, CNN, and ResNet18 classifiers, trained with cross-entropy loss and Adam. It loads CIFAR-10 from Hugging Face and uses a fixed 45,000/5,000 training/validation split. |
| Logging support | Implemented | Runs write per-epoch or per-step metrics to `metrics.csv` and TensorBoard event files. |
| Saving/checkpointing | Implemented | Normal training saves `best.pt` when validation accuracy improves. Sanity mode saves `checkpoint.pt`. |
| Hyperparameter selection logging | Implemented | Each run writes `config.json` with the model, learning rate, batch size, epoch count, seed, and other settings. |
| Sanity check and logs | Run successfully | A CNN sanity run repeatedly trained on one batch. Loss fell from 2.3033 to 0.0006; accuracy on that repeated batch reached 100%. This is the expected overfit check. |
| Demo application | Implemented; needs a launch check | `demo.py` accepts an uploaded image and displays the top three class predictions. Launch it with a checkpoint and confirm the upload and prediction flow work. |

## Sanity Run Details

The sanity run confirms the data loader, model, loss, and optimizer work together and can memorize a single batch. It does not measure generalization or final CIFAR-10 test accuracy. Its output is in `runs/cnn_sanity/` relative to the working directory used to launch the script (for example, `phase2/runs/cnn_sanity/` if launched from the `phase2` directory):

- `config.json` records the run settings and selected device.
- `metrics.csv` contains the sanity loss and accuracy values.
- `tensorboard/` contains event files for TensorBoard.
- `checkpoint.pt` contains the sanity-run model weights and metadata.

PyTorch was confirmed to detect a Tesla T4 in Colab (`torch.cuda.is_available()` returned `True`). A TensorFlow `cudart_stub` warning may still appear from unrelated Colab packages; check PyTorch directly to confirm its GPU status.

## Remaining Actions

1. Launch `demo.py` using a trained checkpoint and verify an image upload returns predictions.
2. Download the `cnn_sanity` run folder from Colab before its runtime ends.
3. Upload the project code and sanity-run logs to eNauczanie as required by the course brief.
4. Push the latest source and documentation changes to the Git remote.

## Git and Colab Files

The training source of truth is `train.py` in this repository. The Colab session runs that source; a separate Colab notebook is not required for the code submission unless the group has added unique code or experiment notes to the notebook. If the notebook contains project-specific work worth preserving, download its `.ipynb` file and commit it as documentation.

Do not commit the downloaded dataset, virtual environment, or routine run outputs. They are excluded by `.gitignore`. Keep a local copy of the sanity logs and submit them to eNauczanie; add run artifacts to Git only if the group intentionally wants them versioned and the repository size remains reasonable.
