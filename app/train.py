"""
train.py — Entraînement de l'auto-encodeur convolutif

Reproduit la boucle d'entraînement du notebook, incluant :
  1. L'entraînement principal (MSE + Perceptual Loss VGG16)
  2. Le fine-tuning rapide optionnel sur 2 photos personnelles
  3. Le fine-tuning qualité++ avec Sobel edge loss (Cellule 61 du notebook)

Utilisation :
  python train.py                        # entraînement principal sur CelebA
  python train.py --finetune-sobel       # fine-tuning Sobel (requiert que le modèle existe)
"""

import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tv_models
from tqdm import tqdm

from model import build_model, load_model
from dataset import get_dataloaders


# ============================================================================
# Perceptual Loss (VGG16) — identique au notebook
# ============================================================================
class VGG16Features(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).to(device).eval()
        self.features = nn.Sequential(*list(vgg.features.children())[:16])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.features(x)


def perceptual_loss(vgg_feat, rec, target):
    return F.mse_loss(vgg_feat(rec), vgg_feat(target))


# ============================================================================
# Sobel Edge Loss — identique au notebook (Cellule 61)
# ============================================================================
def sobel_edges(x):
    """x: [B, 3, H, W] dans [-1, 1]"""
    gray = x.mean(dim=1, keepdim=True)
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


# ============================================================================
# Augmentation tensorielle (Cellule 61)
# ============================================================================
def augment_tensor_v2(x):
    y = x.clone()
    if torch.rand(1).item() < 0.25:
        y = torch.flip(y, dims=[2])
    dx = int(torch.randint(-3, 4, (1,)).item())
    dy = int(torch.randint(-3, 4, (1,)).item())
    y = torch.roll(y, shifts=(dy, dx), dims=(1, 2))
    if torch.rand(1).item() < 0.6:
        y = y + 0.015 * torch.randn_like(y)
    return y.clamp(-1.0, 1.0)


# ============================================================================
# Entraînement principal (identique au notebook — Cell 35)
# ============================================================================
def train_main(
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 0.001,
    weight_decay: float = 0.0004,
    lambda_perc: float = 0.08,
    save_path: str = "autoencoder_trained.pth",
    data_root: str = "./data/faces",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model = build_model(device)
    train_loader, _, _ = get_dataloaders(data_root=data_root, batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    vgg_feat = VGG16Features(device)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Époque {epoch}/{epochs}")
        for data, _ in pbar:
            data = data.to(device)
            optimizer.zero_grad()
            rec, latent = model(data)

            loss_mse = criterion(rec, data)
            loss_perc = perceptual_loss(vgg_feat, rec, data)
            loss = loss_mse + lambda_perc * loss_perc

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        avg = epoch_loss / len(train_loader)
        print(f"  → Époque {epoch} terminée | loss moyenne = {avg:.5f}")

    torch.save(model.state_dict(), save_path)
    print(f"✓ Modèle sauvegardé dans {save_path}")


# ============================================================================
# Fine-tuning qualité++ avec Sobel edge loss (Cell 61)
# ============================================================================
def finetune_sobel(
    photo_a: str = "photo_a.jpg",
    photo_b: str = "photo_b.jpg",
    weights_path: str = "autoencoder_trained.pth",
    save_path: str = "autoencoder_trained.pth",
    steps: int = 120,
    lr: float = 1.5e-5,
    lambda_edge: float = 0.12,
):
    """
    Fine-tuning court orienté netteté : MSE + Edge loss (Sobel).
    Reproduit exactement la cellule 61 du notebook.
    """
    from inference import load_and_encode_photo  # import ici pour éviter un import circulaire

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    model = load_model(weights_path, device)

    # Charger et encoder les deux photos
    _, img_A = load_and_encode_photo(model, photo_a, device)
    _, img_B = load_and_encode_photo(model, photo_b, device)

    base_imgs = torch.stack([img_A.detach().cpu(), img_B.detach().cpu()], dim=0)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for step in range(1, steps + 1):
        idx = torch.randint(0, 2, (1,)).item()
        inp = augment_tensor_v2(base_imgs[idx]).unsqueeze(0).to(device)

        optimizer.zero_grad()
        rec, _ = model(inp)

        loss_mse = F.mse_loss(rec, inp)
        loss_edge = F.l1_loss(sobel_edges(rec), sobel_edges(inp))
        loss = loss_mse + lambda_edge * loss_edge

        loss.backward()
        optimizer.step()

        if step % 30 == 0:
            print(f"[FT++] step {step}/{steps} | mse={loss_mse.item():.5f} edge={loss_edge.item():.5f}")

    model.eval()
    torch.save(model.state_dict(), save_path)
    print(f"✓ Fine-tuning qualité++ terminé — modèle sauvegardé dans {save_path}")


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement / Fine-tuning de l'auto-encodeur")
    parser.add_argument("--finetune-sobel", action="store_true",
                        help="Lancer le fine-tuning Sobel edge loss au lieu de l'entraînement principal")
    parser.add_argument("--photo-a", default="photo_a.jpg", help="Chemin vers la photo A")
    parser.add_argument("--photo-b", default="photo_b.jpg", help="Chemin vers la photo B")
    parser.add_argument("--weights", default="autoencoder_trained.pth", help="Chemin vers les poids du modèle")
    parser.add_argument("--epochs", type=int, default=20, help="Nombre d'époques (entraînement principal)")
    parser.add_argument("--batch-size", type=int, default=16, help="Taille de batch")
    parser.add_argument("--data-root", default="./data/faces", help="Répertoire racine CelebA")
    args = parser.parse_args()

    if args.finetune_sobel:
        finetune_sobel(
            photo_a=args.photo_a,
            photo_b=args.photo_b,
            weights_path=args.weights,
            save_path=args.weights,
        )
    else:
        train_main(
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_root=args.data_root,
        )
