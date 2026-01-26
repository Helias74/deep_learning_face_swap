from torchvision import datasets, transforms
import torch

def get_dataloaders(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((64,64)),  #redimensionne l'image
        transforms.ToTensor()        #tenseurs doit etre fixe/normaliser
    ])
    train = datasets.ImageFolder("data/train", transform=transform) #attribue l'animal a une etiquette selon 
    val = datasets.ImageFolder("data/val", transform=transform)     #l'ordre alphabetiques des dossiers
    return (
        torch.utils.data.DataLoader(train,batch_size,shuffle=True), #retourne les donnees en groupe et melanger
        torch.utils.data.DataLoader(val,batch_size) 
    )
