import cv2
import torch
from PIL import Image
from torchvision import transforms
from collections import Counter
from model import SimpleCNN,HumanCNN,ShapeCNN

"""model = HumanCNN()
model.load_state_dict(torch.load("model.pth", map_location="cpu")) #charge les poids de l'entrainement
model.eval() #mode evaluation desactive le Dropout (toutes les connexions sont actives) et fixe le BatchNorm (moyenne et variance) calculé differement

img = Image.open("test.jpg").convert("RGB")
t = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()]) #converti au format de notre tenseur
x = t(img).unsqueeze(0) #adapte aussi au format attendu (ajoute un parametre batch)

pred = torch.argmax(model(x),1).item() #on passe x dans le model et recupere la classe la plus probable
print(["Human","NonHuman"][pred])"""


#chargement du modele
model = ShapeCNN(num_classes=3)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

classes = ["carre", "cercle", "triangle"]

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

#detection des formes 
def detect_shapes(image_path):                   #prends une image
    img = cv2.imread(image_path)                 #lit l'image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #transforme en niveau de gris 

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) #binarise

    contours, _ = cv2.findContours( 
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    ) #chaque contour est a peu pres une forme

    crops = [] #liste les formes decoupées
    for cnt in contours: #traite toutes les formes
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 20 or h < 20:
            continue
        crop = img[y:y+h, x:x+w] #decoupe
        crops.append(crop)       #stocke
    return crops

#comptage
def count_shapes(image_path):
    crops = detect_shapes(image_path)
    counts = Counter() #compteur automatique

    with torch.no_grad():
        for crop in crops: #pour chaque forme detectée
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) #conversion
            x = transform(pil_img).unsqueeze(0)
            pred = torch.argmax(model(x), 1).item() #prediction
            counts[classes[pred]] += 1
    return counts


if __name__ == "__main__":
    image_path = "photo/2.jpg"
    counts = count_shapes(image_path)

    print("Résultat :")
    for shape, nb in counts.items():
        print(f"{shape} : {nb}")
