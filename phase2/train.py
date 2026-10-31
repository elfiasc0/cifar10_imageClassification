"""Train CIFAR-10 baseline architectures or run the one-batch sanity check."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
from torchvision import transforms


CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class HFCIFAR10(Dataset):
    """Adapt the Hugging Face CIFAR-10 split to torchvision transforms."""
    def __init__(self, split, transform):
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        row = self.split[index]
        image = row["img"].convert("RGB")
        return self.transform(image), int(row["label"])


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(3 * 32 * 32, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def make_model(name):
    if name == "mlp":
        return MLP()
    if name == "cnn":
        return SmallCNN()
    if name == "resnet18":
        from torchvision.models import resnet18
        model = resnet18(weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    raise ValueError(f"Unknown model: {name}")


def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mlp", "cnn", "resnet18"), default="cnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--sanity", action="store_true", help="repeat one training batch to check the training loop")
    parser.add_argument("--sanity-steps", type=int, default=200)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    run_name = f"{args.model}_{'sanity' if args.sanity else time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"device": str(device), "classes": list(CLASSES)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    writer = SummaryWriter(run_dir / "tensorboard")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dataset = load_dataset("uoft-cs/cifar10", cache_dir=args.data_dir)
    hf_train = dataset["train"]
    train_aug = HFCIFAR10(hf_train, train_transform)
    train_eval = HFCIFAR10(hf_train, eval_transform)
    generator = torch.Generator().manual_seed(args.seed)
    train_indices, val_indices = random_split(range(len(hf_train)), [45000, 5000], generator=generator)
    train_subset = Subset(train_aug, train_indices.indices)
    # Validation uses the same fixed indices without random training augmentation.
    val_subset = Subset(train_eval, val_indices.indices)
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model = make_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_val = 0.0

    with (run_dir / "metrics.csv").open("w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "seconds"])
        if args.sanity:
            model.train()
            images, labels = next(iter(DataLoader(train_eval, batch_size=args.batch_size, shuffle=True)))
            images, labels = images.to(device), labels.to(device)
            for step in range(1, args.sanity_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                if step == 1 or step % 10 == 0:
                    acc = accuracy(logits.detach(), labels)
                    writer.add_scalar("sanity/loss", loss.item(), step)
                    writer.add_scalar("sanity/accuracy", acc, step)
                    log.writerow([step, loss.item(), acc, "", "", ""])
                    f.flush()
                    print(f"step={step:04d} loss={loss.item():.4f} accuracy={acc:.3f}")
            torch.save({"model": model.state_dict(), "model_name": args.model, "classes": CLASSES,
                        "mean": MEAN, "std": STD}, run_dir / "checkpoint.pt")
            print(f"Sanity checkpoint and logs: {run_dir}")
            writer.close()
            return

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            loss_sum = correct = seen = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                seen += labels.size(0)
            train_loss, train_acc = loss_sum / seen, correct / seen

            model.eval()
            val_loss_sum = val_correct = val_seen = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    val_loss_sum += criterion(logits, labels).item() * labels.size(0)
                    val_correct += (logits.argmax(1) == labels).sum().item()
                    val_seen += labels.size(0)
            val_loss, val_acc = val_loss_sum / val_seen, val_correct / val_seen
            seconds = time.time() - start
            log.writerow([epoch, train_loss, train_acc, val_loss, val_acc, seconds])
            f.flush()
            for key, value in (("loss/train", train_loss), ("accuracy/train", train_acc),
                               ("loss/validation", val_loss), ("accuracy/validation", val_acc)):
                writer.add_scalar(key, value, epoch)
            print(f"epoch={epoch:03d}/{args.epochs} train_acc={train_acc:.3f} "
                  f"val_acc={val_acc:.3f} val_loss={val_loss:.4f} time={seconds:.1f}s")
            if val_acc > best_val:
                best_val = val_acc
                torch.save({"model": model.state_dict(), "model_name": args.model, "classes": CLASSES,
                            "mean": MEAN, "std": STD, "epoch": epoch, "val_accuracy": val_acc},
                           run_dir / "best.pt")
    writer.close()
    print(f"Best validation accuracy: {best_val:.4f}; run files: {run_dir}")


if __name__ == "__main__":
    main()
