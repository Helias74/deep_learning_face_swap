from torchvision import datasets, transforms
import torch

def get_dataloaders(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])
    train = datasets.ImageFolder("data/train", transform=transform)
    val = datasets.ImageFolder("data/val", transform=transform)
    return (
        torch.utils.data.DataLoader(train,batch_size,shuffle=True),
        torch.utils.data.DataLoader(val,batch_size)
    )
