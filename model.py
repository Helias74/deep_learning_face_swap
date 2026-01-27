import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential( #regarde l'image
            nn.Conv2d(3,16,3,padding=1), #format (entree, sortie, taille filtre [3x3, 5x5, etc], taille d'ajout de bordure) #padding pour conserver taille image de base
            nn.ReLU(), #supprime les valeurs negatives
            nn.MaxPool2d(2), #reduction (division) volontaire pour generaliser et aller plus vite (mais alors pourquoi le padding [point a approfondir])
            nn.Conv2d(16,32,3,padding=1), #la sortie precedente devient l'entree actuel et donc la sortie est augmenter
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential( #prend la decision
            nn.Flatten(),
            nn.Linear(64*8*8,128), #64 car derniere sortie (L13) et 8 car 64 (format de l'image) divisé en deux trois fois (ce qui donne au final 64*64 le format initial de l'image)
            nn.ReLU(),
            nn.Linear(128,2) #128 ici et L19 c'est en combien de neurones/etapes tout les pixels passent et 2 pour les deux differentes classes/cas (chat, chien)
        )
    def forward(self,x):
        return self.classifier(self.features(x)) 
