import torch
from PIL import Image
from torchvision import transforms
from model import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu")) #charge les poids de l'entrainement
model.eval() #mode evaluation desactive le Dropout (toutes les connexions sont actives) et fixe le BatchNorm (moyenne et variance) calculé differement

img = Image.open("test.jpg").convert("RGB")
t = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()]) #converti au format de notre tenseur
x = t(img).unsqueeze(0) #adapte aussi au format attendu (ajoute un parametre batch)

pred = torch.argmax(model(x),1).item() #on passe x dans le model et recupere la classe la plus probable
print(["Chat","Chien"][pred])
