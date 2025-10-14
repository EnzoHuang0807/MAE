import os
import argparse
import random
import torch
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np

def denormalize(img_tensor):
    img_tensor = img_tensor * 0.5 + 0.5
    return img_tensor.clamp(0, 1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='vit-t-classifier-from-scratch.pt')
    parser.add_argument('--img_path', type=str, default='from_scratch.png')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    random.seed(args.seed)

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=transform)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(args.model_path, weights_only=False)
    model.eval().to(device)

    indices = random.sample(range(len(val_dataset)), 8)
    imgs, labels = zip(*[val_dataset[i] for i in indices])
    imgs = torch.stack(imgs).to(device)
    labels = torch.tensor(labels).to(device)

    # Predict
    with torch.no_grad():
        preds = model(imgs).argmax(dim=1)

    # Plot grid
    plt.figure(figsize=(12, 6))
    for i in range(8):
        img = denormalize(imgs[i].cpu())
        img = img.permute(1, 2, 0).numpy()
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(
            f"True: {classes[labels[i]]}\nPred: {classes[preds[i]]}", 
            fontsize=12,
            color=("green" if preds[i] == labels[i] else "red")
        )
        plt.axis("off")
    plt.tight_layout()

    # Save figure
    plt.savefig(os.path.join("figure", args.img_path), dpi=200)
    plt.close()
    print(f"Saved visualization to figure/{args.img_path}")