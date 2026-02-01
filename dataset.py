from torchvision import datasets, transforms
import torch
"""
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
    )"""


def get_dataloaders(batch_size=16):
    train_transform = transforms.Compose([     #pour l'entrainement
        transforms.Resize((64,64)),            #redimensionne l'image
        transforms.RandomRotation(360),        #apprend que l'orientation n'est pas importante
        transforms.RandomHorizontalFlip(),     #evite au model d'apprendre des positions fixes
        transforms.RandomVerticalFlip(),   
        transforms.ToTensor(),                 #tenseurs doit etre fixe/normaliser
        transforms.Normalize([0.5]*3, [0.5]*3) #permet d'avoir un gradient plus stable
    ])

    val_transform = transforms.Compose([       #pour la validation
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3) #les valeurs doivent etre les memes qu'à l'entrainement
    ])

    train = datasets.ImageFolder("data/train", transform=train_transform) #retourne les donnees en groupe et melanger
    val = datasets.ImageFolder("data/val", transform=val_transform)       #l'ordre alphabetiques des dossiers

    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True), #retourne les donnees en groupe et melanger
        torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
    )

