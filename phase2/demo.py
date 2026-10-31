"""Small Gradio image demo for a saved CIFAR-10 model checkpoint."""
import argparse

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from train import CLASSES, make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to best.pt or checkpoint.pt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = make_model(checkpoint["model_name"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize(checkpoint.get("mean", (0.4914, 0.4822, 0.4465)),
                             checkpoint.get("std", (0.2470, 0.2435, 0.2616))),
    ])

    def predict(image: Image.Image):
        if image is None:
            return {}
        x = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(x).softmax(1)[0].cpu().tolist()
        return {name: probs[i] for i, name in enumerate(CLASSES)}

    gr.Interface(predict, gr.Image(type="pil"), gr.Label(num_top_classes=3),
                 title="CIFAR-10 image classifier",
                 description="Upload a 32×32-style object image. The model predicts one of CIFAR-10's 10 classes.").launch()


if __name__ == "__main__":
    main()
