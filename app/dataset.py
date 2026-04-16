"""
dataset.py — Chargement du dataset CelebA et utilitaires de données

Reproduit exactement la logique du notebook :
  - CelebA split='train', download=False (les données doivent déjà exister)
  - Sous-ensemble de 10 000 images pour l'entraînement
  - Sous-ensemble de 1 000 images pour la visualisation / le morphing
  - Transformations : Resize(128×128), ToTensor, Normalize(0.5)
"""

import random
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Transformations standard (identiques au notebook)
# ---------------------------------------------------------------------------
def get_transform(image_size: int = 128) -> transforms.Compose:
    """Retourne la pipeline de transformations utilisée dans le notebook."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ---------------------------------------------------------------------------
# Construction des DataLoaders
# ---------------------------------------------------------------------------
def get_dataloaders(
    data_root: str = "./data/faces",
    batch_size: int = 16,
    image_size: int = 128,
    train_size: int = 10_000,
    viz_size: int = 1_000,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, datasets.CelebA]:
    """
    Retourne (train_loader, test_loader, base_dataset).

    La logique est identique au notebook :
      1. Charger CelebA (split='train', download=False).
      2. Mélanger les indices avec seed=42.
      3. Prendre les *train_size* premiers indices pour l'entraînement.
      4. Prendre les *viz_size* indices suivants (hors training) pour la visu.
    """
    transform = get_transform(image_size)

    base_train_celeba = datasets.CelebA(
        root=data_root, split="train", download=False, transform=transform
    )

    random.seed(seed)
    all_idx = list(range(len(base_train_celeba)))
    random.shuffle(all_idx)

    train_indices = all_idx[:train_size]
    train_indices_set = set(train_indices)

    viz_indices = [i for i in all_idx[train_size:] if i not in train_indices_set][:viz_size]

    train_loader = DataLoader(
        Subset(base_train_celeba, train_indices),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        Subset(base_train_celeba, viz_indices),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, base_train_celeba


# ---------------------------------------------------------------------------
# Encodage des latents de test (pour le morphing par défaut)
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_test_latents(model, test_loader, device):
    """
    Encode toutes les images du test_loader et retourne
    (latents: Tensor, digits: Tensor) — terminologie du notebook.
    """
    latents_list, digits_list = [], []
    model.eval()
    for data, cls in test_loader:
        _, lat = model(data.to(device))
        latents_list.append(lat.cpu())
        digits_list.append(cls)
    latents = torch.cat(latents_list, dim=0)
    digits = torch.cat(digits_list, dim=0)
    return latents, digits
