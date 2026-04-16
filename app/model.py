"""
model.py — Définition de l'auto-encodeur convolutif (ConvAutoEncoder)

Architecture extraite du notebook `notebook_autoencodeur_visage.ipynb`.
Reproduit exactement les classes ConvEncoder, ConvDecoder et ConvAutoEncoder
afin que le fichier `autoencoder_trained.pth` puisse être chargé sans aucune
modification.
"""

import importlib
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Utilitaire : chargement dynamique d'une classe à partir de son chemin complet
# (ex. "torch.nn.Tanh")
# ---------------------------------------------------------------------------
def load_class(full_class_path: str):
    module_name, class_name = full_class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Encodeur convolutif
# ---------------------------------------------------------------------------
class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        input_size: int = 28,
        hidden_channels=None,
        latent_dim: int = 10,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        self.input_channels = input_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels

        encoder_layers = []
        in_channels = input_channels
        for out_channels in hidden_channels:
            encoder_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_channels = out_channels
        self.encoder_conv = nn.Sequential(*encoder_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_size, input_size)
            out = self.encoder_conv(dummy)
            _, C, H, W = out.shape
        self.encoder_fc = nn.Sequential(nn.Flatten(), nn.Linear(C * H * W, latent_dim))
        self.feature_channels, self.feature_height, self.feature_width = C, H, W

    def forward(self, x):
        return self.encoder_fc(self.encoder_conv(x))


# ---------------------------------------------------------------------------
# Décodeur convolutif
# ---------------------------------------------------------------------------
class ConvDecoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels=None,
        latent_dim: int = 10,
        feature_channels: int = 128,
        feature_height: int = 4,
        feature_width: int = 4,
        output_activation: str = "torch.nn.Sigmoid",
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        self.feature_channels = feature_channels
        self.feature_height = feature_height
        self.feature_width = feature_width

        flattened_dim = feature_channels * feature_height * feature_width
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, flattened_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        decoder_layers = []
        in_ch = hidden_channels[-1]
        for out_ch in reversed(hidden_channels[:-1]):
            decoder_layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_ch = out_ch
        decoder_layers.append(
            nn.ConvTranspose2d(in_ch, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        decoder_layers.append(load_class(output_activation)())
        self.decoder_conv = nn.Sequential(*decoder_layers)

    def forward(self, z, target_height=None, target_width=None):
        batch_size = z.size(0)
        x = self.decoder_fc(z).view(
            batch_size, self.feature_channels, self.feature_height, self.feature_width
        )
        rec = self.decoder_conv(x)
        if target_height and target_width:
            h_diff = rec.size(2) - target_height
            w_diff = rec.size(3) - target_width
            if h_diff >= 0 and w_diff >= 0:
                rec = rec[:, :, h_diff // 2 : h_diff // 2 + target_height,
                                w_diff // 2 : w_diff // 2 + target_width]
        return rec


# ---------------------------------------------------------------------------
# Auto-Encodeur complet
# ---------------------------------------------------------------------------
class ConvAutoEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        input_size: int = 28,
        hidden_channels=None,
        latent_dim: int = 10,
        output_activation: str = "torch.nn.Sigmoid",
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [32, 64, 128]
        self.encoder = ConvEncoder(input_channels, input_size, hidden_channels, latent_dim)
        self.decoder = ConvDecoder(
            input_channels,
            hidden_channels,
            latent_dim,
            self.encoder.feature_channels,
            self.encoder.feature_height,
            self.encoder.feature_width,
            output_activation,
        )

    def forward(self, x):
        self.original_height, self.original_width = x.size(2), x.size(3)
        latent = self.encode(x)
        return self.decode(latent, self.original_height, self.original_width), latent

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, target_height=None, target_width=None):
        if target_height is None and hasattr(self, "original_height"):
            target_height = self.original_height
        if target_width is None and hasattr(self, "original_width"):
            target_width = self.original_width
        return self.decoder(z, target_height, target_width)


# ---------------------------------------------------------------------------
# Fonction utilitaire pour créer le modèle utilisé par le notebook
# ---------------------------------------------------------------------------
def build_model(device: torch.device | str = "cpu") -> ConvAutoEncoder:
    """Crée le ConvAutoEncoder avec les hyper-paramètres du notebook."""
    model = ConvAutoEncoder(
        input_channels=3,
        input_size=128,
        hidden_channels=[64, 128, 256, 512],
        latent_dim=512,
        output_activation="torch.nn.Tanh",
    ).to(device)
    return model


def load_model(weights_path: str = "autoencoder_trained.pth",
               device: torch.device | str = "cpu") -> ConvAutoEncoder:
    """Crée le modèle et charge les poids pré-entraînés."""
    model = build_model(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model
