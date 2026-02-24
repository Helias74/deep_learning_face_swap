import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN,ShapeCNN
from dataset import get_dataloaders

"""
train_loader, _ = get_dataloaders() #recup les données en batch
model = SimpleCNN()
criterion = nn.CrossEntropyLoss() #fonction de perte
optimizer = optim.Adam(model.parameters(), lr=0.001) #maniere d'optimiser (Adam en l'occurence)

#systeme qui ressemble a la fonction CheminNonSature (Dijkstra) dans le tp9 de lifapc dans l'actualisation des gradients
for epoch in range(10):
    loss_sum = 0
    for x,y in train_loader: #(x = image du batch ,y = etiquette [nom de dossier] correspondants)
        optimizer.zero_grad() #reinitialise les gradients
        loss = criterion(model(x), y) #calcul de perte
        loss.backward() #calcul de gradient
        optimizer.step() #maj des poids et biais selon les gradients
        loss_sum += loss.item()
    print(f"Epoch {epoch+1} Loss {loss_sum:.3f}")

torch.save(model.state_dict(),"human_cnn_v1.pth")
print("Model saved")"""


train_loader, val_loader = get_dataloaders() #recup les données en batch

model = ShapeCNN(num_classes=3) #nombre de formes à reconnaitre
criterion = nn.CrossEntropyLoss() #fonction de perte
optimizer = optim.Adam(model.parameters(), lr=0.001) #maniere d'optimiser (Adam en l'occurence)

#systeme qui ressemble a la fonction CheminNonSature (Dijkstra) dans le tp9 de lifapc dans l'actualisation des gradients
for epoch in range(20):
    model.train() #active le dropout
    loss_sum = 0
    for x,y in train_loader: #(x = image du batch ,y = etiquette [nom de dossier] correspondants)
        optimizer.zero_grad() #reinitialise les gradients
        loss = criterion(model(x), y) #calcul de perte
        loss.backward() #calcul de gradient
        optimizer.step() #maj des poids et biais selon les gradients
        loss_sum += loss.item()
    print(f"Epoch {epoch+1} Loss {loss_sum:.3f}")

torch.save(model.state_dict(), "model.pth")
print("Model saved")