# Phase 2: CIFAR-10 Architecture Comparison

This project compares three image classifiers on CIFAR-10: a multilayer perceptron (MLP), a small convolutional neural network (CNN), and ResNet18. CIFAR-10 is loaded from Hugging Face's [`uoft-cs/cifar10`](https://huggingface.co/datasets/uoft-cs/cifar10) dataset. The official 50,000-image training split is divided into 45,000 training images and 5,000 validation images. The official 10,000-image test split is reserved for the final Phase 3 comparison.

## Environment Setup

Google Colab with a GPU is recommended for the first training runs. Kaggle Notebooks is also suitable. A local machine works for development: the script automatically selects NVIDIA CUDA, Apple MPS, or CPU, in that order. The TensorFlow `cudart_stub` warning sometimes printed in Colab is separate from this PyTorch script; check PyTorch's GPU access with `torch.cuda.is_available()`.

```bash
cd phase2
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

In Colab, clone or update the repository, install `phase2/requirements.txt`, then run the commands below from the `phase2` directory. Colab runtimes are temporary, so copy required run outputs somewhere persistent before the session ends.

## Train and Demo Commands

```bash
# Sanity check: repeatedly train on one batch; loss should fall and accuracy rise.
python train.py --model cnn --sanity --sanity-steps 200

# Normal training run.
python train.py --model cnn --epochs 20 --seed 42

# Open TensorBoard charts for the runs directory.
tensorboard --logdir runs

# Launch the Gradio demo using a normal run's best checkpoint.
python demo.py runs/<run-name>/best.pt
```

Choose `--model mlp`, `--model cnn`, or `--model resnet18`. To make architecture comparisons meaningful, use the same split, seed, and training settings. For Colab commands run from the repository root, prefix script paths with `phase2/`, for example `!python phase2/train.py --model cnn --epochs 20 --seed 42`.

## What `train.py` Does

### Dataset and preprocessing

`load_dataset("uoft-cs/cifar10")` downloads and caches the Hugging Face dataset under `data/` by default. The `HFCIFAR10` adapter reads each row's `img` and `label` fields, converts the image to RGB, applies the selected transform, and returns a tensor and integer class label for PyTorch's `DataLoader`.

The split is created once with the configured random seed: 45,000 images for training and 5,000 for validation. Training images receive random crops and horizontal flips, followed by tensor conversion and channel normalization. Validation images receive tensor conversion and normalization without random augmentation. The test split is not evaluated during Phase 2.

### Models, loss, and device

All models return ten class scores (logits), one for each CIFAR-10 class. The loss is cross-entropy and the optimizer is Adam.

- **MLP:** flattens each image into a vector and uses fully connected layers with ReLU and dropout. It is a simple baseline that does not explicitly preserve spatial structure.
- **CNN:** uses convolutional layers, batch normalization, ReLU, and pooling to learn spatial image features before its classifier.
- **ResNet18:** uses residual skip connections. Its first convolution and max-pooling are adjusted for small 32×32 CIFAR-10 images.

The command-line arguments select the model, epoch count, batch size, learning rate, random seed, and data and output directories. The script seeds Python and PyTorch, then selects CUDA, MPS, or CPU according to availability.

### Sanity mode and normal mode

With `--sanity`, the script takes one batch from the training split and repeatedly trains on that same batch without random augmentation. It reports loss and accuracy at step 1 and every 10 steps, writes those metrics to CSV and TensorBoard, and saves `checkpoint.pt`. The goal is to memorize the batch: falling loss and rising accuracy check that loading, the model, loss, and optimizer work together. This is a debugging check, not a measure of generalization.

Without `--sanity`, training runs for the requested number of epochs. Each epoch trains over the training split, evaluates on validation data, logs metrics, and saves `best.pt` whenever validation accuracy improves. Use `best.pt` with the demo.

## Logs and Checkpoints

Each run creates a folder under `runs/`—for example, `runs/cnn_sanity/` for sanity mode or a timestamped folder for normal training—with:

- `config.json`: model, hyperparameters, seed, class names, and selected device.
- `metrics.csv`: training and validation loss and accuracy per epoch; sanity loss and accuracy per logged step.
- `tensorboard/`: TensorBoard event files for visualizing metrics.
- `checkpoint.pt` in sanity mode, or `best.pt` in normal mode: model weights and metadata.

Open TensorBoard with `tensorboard --logdir runs`. The repository `.gitignore` excludes downloaded data, run outputs, virtual environments, and Python cache files. Keep the source code in Git; submit required sanity or best-run logs and checkpoints separately as requested by the course.

## Demo Application

`demo.py` loads a normal run's `best.pt`, accepts an image upload through Gradio, resizes and normalizes the image, and displays the top three predicted CIFAR-10 classes with scores. The CIFAR-10 images are 32×32 pixels, so predictions on unrelated or very different images may be unreliable.
