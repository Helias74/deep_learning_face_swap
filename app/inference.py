"""
inference.py — Pipeline d'inférence « Mode TOP DU TOP »

Reproduit exactement la dernière cellule du notebook (Cell 63) :
  - Interpolation SLERP dans l'espace latent
  - 21 frames
  - Post-traitement unsharp pour netteté / perception
  - Les bornes t=0 et t=1 utilisent les images sources exactes

Utilisation :
  python inference.py --photo-a photo_a.jpg --photo-b photo_b.jpg
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from torchvision import transforms

from model import load_model


# ============================================================================
# Transformations et utilitaires de chargement d'images
# ============================================================================
transform_custom = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def _center_square_crop(img: PILImage.Image) -> PILImage.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _ensure_training_range(t: torch.Tensor):
    """Force un tenseur image vers la plage [-1, 1] attendue par le modèle."""
    x = t.clone().float()
    if torch.isnan(x).any() or torch.isinf(x).any():
        return None

    xmin = x.min().item()
    xmax = x.max().item()

    # Cas 1 : déjà en [-1, 1]
    if xmin >= -1.5 and xmax <= 1.5:
        return x
    # Cas 2 : en [0, 1]
    if xmin >= -0.01 and xmax <= 1.01:
        return (x - 0.5) / 0.5
    # Cas 3 : en [0, 255]
    if xmin >= -1.0 and xmax <= 255.0:
        x = x / 255.0
        return (x - 0.5) / 0.5

    return None


def load_and_encode_photo(model, path: str, device):
    """
    Charge une photo, la prétraite et l'encode.
    Retourne (latent, tensor_image) tous deux sur CPU.
    Reproduit exactement la logique de la cellule 56 du notebook.
    """
    img = PILImage.open(path).convert("RGB")
    tensor = None

    # 1) Tentative d'alignement visage via MTCNN (optionnel)
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(image_size=128, margin=20, post_process=True, device=device)
        aligned = mtcnn(img)
        if aligned is not None:
            tensor = _ensure_training_range(aligned.unsqueeze(0).to(device))
    except Exception:
        pass

    # 2) Fallback : crop carré centré + preprocessing standard
    if tensor is None:
        img_c = _center_square_crop(img)
        tensor = transform_custom(img_c).unsqueeze(0).to(device)

    with torch.no_grad():
        latent = model.encode(tensor)
    return latent.squeeze(0), tensor.squeeze(0)


# ============================================================================
# Fine-tuning qualité ++ (Contours / Sobel Edge Loss)
# ============================================================================
def augment_tensor_v2(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    if torch.rand(1).item() < 0.25:
        y = torch.flip(y, dims=[2])
    dx = int(torch.randint(-3, 4, (1,)).item())
    dy = int(torch.randint(-3, 4, (1,)).item())
    y = torch.roll(y, shifts=(dy, dx), dims=(1, 2))
    if torch.rand(1).item() < 0.6:
        y = y + 0.015 * torch.randn_like(y)
    return y.clamp(-1.0, 1.0)

def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    gray = x.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=x.device).view(1,1,3,3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)

def fine_tune_for_morphing(model, img_A: torch.Tensor, img_B: torch.Tensor, device, steps: int = 120, lr: float = 1.5e-5):
    print(f"Lancement du fine-tuning orienté netteté ({steps} steps)...")
    base_imgs = torch.stack([img_A.detach().cpu(), img_B.detach().cpu()], dim=0)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lambda_edge = 0.12

    for step in range(1, steps + 1):
        idx = torch.randint(0, 2, (1,)).item()
        inp = augment_tensor_v2(base_imgs[idx]).unsqueeze(0).to(device)

        opt.zero_grad()
        rec, _ = model(inp)

        loss_mse = F.mse_loss(rec, inp)
        loss_edge = F.l1_loss(sobel_edges(rec), sobel_edges(inp))
        loss = loss_mse + lambda_edge * loss_edge

        loss.backward()
        opt.step()

        if step % 30 == 0:
            print(f"[FT++] step {step}/{steps} | mse={loss_mse.item():.5f} edge={loss_edge.item():.5f}")

    model.eval()
    print("✓ Fine-tuning qualité++ terminé.")


# ============================================================================
# SLERP — Interpolation sphérique (identique au notebook, Cell 63)
# ============================================================================
def slerp(z0: torch.Tensor, z1: torch.Tensor, t: float) -> torch.Tensor:
    z0n = z0 / (z0.norm() + 1e-8)
    z1n = z1 / (z1.norm() + 1e-8)
    dot = torch.clamp((z0n * z1n).sum(), -0.9995, 0.9995)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if so.abs() < 1e-6:
        return (1.0 - t) * z0 + t * z1
    return (torch.sin((1.0 - t) * omega) / so) * z0 + (torch.sin(t * omega) / so) * z1


# ============================================================================
# Post-traitement : unsharp masking (identique au notebook, Cell 63)
# ============================================================================
def unsharp_np(img: np.ndarray, amount: float = 0.55) -> np.ndarray:
    """
    img : numpy float [0, 1], shape HWC
    Flou gaussien approximé via moyenne locale 3×3.
    """
    pad = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    blur = (
        pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
        pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:] +
        pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
    ) / 9.0
    sharp = np.clip(img + amount * (img - blur), 0.0, 1.0)
    # Contraste doux
    sharp = np.clip((sharp - 0.5) * 1.06 + 0.5, 0.0, 1.0)
    return sharp


# ============================================================================
# Conversion tenseur → image numpy affichable
# ============================================================================
def tensor_to_display(t: torch.Tensor) -> np.ndarray:
    """Convertit un tenseur [C, H, W] en [-1, 1] vers un numpy [H, W, C] en [0, 1]."""
    return np.clip(t.permute(1, 2, 0).numpy() * 0.5 + 0.5, 0, 1)


# ============================================================================
# Morphing TOP DU TOP (Cell 63 du notebook)
# ============================================================================
def morph_top_du_top(
    model,
    img_A: torch.Tensor,
    img_B: torch.Tensor,
    lat_A: torch.Tensor,
    lat_B: torch.Tensor,
    device,
    n_frames: int = 21,
):
    """
    Génère les n_frames du morphing en mode TOP DU TOP.
    Retourne une liste de tenseurs CPU [C, H, W].
    """
    t_vals = torch.linspace(0, 1, steps=n_frames)

    frames = []
    model.eval()
    with torch.no_grad():
        for i, t in enumerate(t_vals):
            if i == 0:
                frame = img_A.cpu()
            elif i == n_frames - 1:
                frame = img_B.cpu()
            else:
                z = slerp(lat_A.cpu(), lat_B.cpu(), float(t)).to(device).unsqueeze(0)
                rec = model.decode(z)[0].cpu()
                frame = rec
            frames.append(frame)

    return frames


# ============================================================================
# Affichage du résultat (reproduction exacte du notebook)
# ============================================================================
def display_morph(frames, n_frames: int = 21, save_path: str = None):
    """
    Affiche le morphing dans une grille 3×7 (pour 21 frames).
    Applique le post-traitement unsharp sur les frames intermédiaires.
    """
    n_cols = 7
    n_rows = (n_frames + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(frames):
            img_d = tensor_to_display(frames[i])
            # Appliquer unsharp seulement sur les frames intermédiaires
            if i not in [0, len(frames) - 1]:
                img_d = unsharp_np(img_d, amount=1.0)
            ax.imshow(img_d)
            ax.set_title(f"t={i / (n_frames - 1):.2f}", fontsize=8)
        ax.axis("off")

    plt.suptitle(f"Morphing A → B | Mode TOP DU TOP ({n_frames} frames)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Image sauvegardée dans {save_path}")

    plt.show()
    print("✓ Rendu TOP DU TOP généré")


# ============================================================================
# Point d'entrée principal
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Morphing TOP DU TOP — Rendu final")
    parser.add_argument("--photo-a", default="photo_a.jpg", help="Chemin vers la photo A")
    parser.add_argument("--photo-b", default="photo_b.jpg", help="Chemin vers la photo B")
    parser.add_argument("--weights", default="autoencoder_trained.pth", help="Chemin vers les poids du modèle")
    parser.add_argument("--n-frames", type=int, default=21, help="Nombre de frames du morphing")
    parser.add_argument("--save", default=None, help="Chemin pour sauvegarder l'image (optionnel)")
    parser.add_argument("--no-finetune", action="store_true", help="Désactiver le fine-tuning (plus rapide mais moins net)")
    parser.add_argument("--ft-steps", type=int, default=120, help="Nombre d'étapes de fine-tuning (défaut: 120)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Charger le modèle
    print("Chargement du modèle...")
    model = load_model(args.weights, device)
    print(f"✓ Modèle chargé depuis {args.weights}")

    # Charger et encoder les photos
    print("Encodage des photos...")
    lat_A, img_A = load_and_encode_photo(model, args.photo_a, device)
    lat_B, img_B = load_and_encode_photo(model, args.photo_b, device)
    print("✓ Photos encodées")

    if not args.no_finetune and args.ft_steps > 0:
        fine_tune_for_morphing(model, img_A, img_B, device, steps=args.ft_steps)

    # Générer le morphing
    print("Génération du morphing TOP DU TOP...")
    frames = morph_top_du_top(model, img_A, img_B, lat_A, lat_B, device, n_frames=args.n_frames)

    # Afficher
    display_morph(frames, n_frames=args.n_frames, save_path=args.save)


if __name__ == "__main__":
    main()
