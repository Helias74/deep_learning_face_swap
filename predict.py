import torch
from PIL import Image
from torchvision import transforms
from model import SimpleCNN

model = SimpleCNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

img = Image.open("test.jpg").convert("RGB")
t = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
x = t(img).unsqueeze(0)

pred = torch.argmax(model(x),1).item()
print(["Chat","Chien"][pred])
