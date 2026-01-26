import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN
from dataset import get_dataloaders

train_loader, _ = get_dataloaders()
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    loss_sum = 0
    for x,y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Epoch {epoch+1} Loss {loss_sum:.3f}")

torch.save(model.state_dict(),"model.pth")
print("Model saved")