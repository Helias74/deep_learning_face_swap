import torch.nn as nn


#Utilisé pour trouvé les chiens
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # ======================
        # Extraction des features
        # ======================
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Rend le modèle indépendant de la taille de l’image
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # ======================
        # Classification
        # ======================
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

        # ======================
        # Normalisation sortie
        # ======================
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.output_activation(x)
        return x
    
    
#Première version pour repérer si humain sur photos    
class HumanCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),    
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

#version pour reconnaitre des formes
import torch
import torch.nn as nn

class ShapeCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # ======================
        # Extraction des features
        # ======================
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Rend le modèle indépendant de la taille de l’image
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # ======================
        # Classification
        # ======================
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # ======================
        # Normalisation sortie
        # ======================
        self.output_activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.output_activation(x)
        return x