# extract feature vectors from pre-trained models
# Before running linear_probe.py, please run this code.

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from torch.utils.data import DataLoader

model_configs = {
    "resnet18": ResNet18_Weights.DEFAULT,
    "resnet50": ResNet50_Weights.DEFAULT
}

for model_name, weights in model_configs.items():
    print(f"Processing {model_name}...")
    
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)

    model.fc = nn.Identity()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    all_feats_train = []
    all_labels_train = []
    all_feats_test = []
    all_labels_test = []

    with torch.no_grad():
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            all_feats_train.append(feats.cpu())
            all_labels_train.append(labels)

        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            all_feats_test.append(feats.cpu())
            all_labels_test.append(labels)

    all_feats_train = torch.cat(all_feats_train,  dim=0)
    all_labels_train = torch.cat(all_labels_train, dim=0)
    all_feats_test = torch.cat(all_feats_test,  dim=0)
    all_labels_test = torch.cat(all_labels_test, dim=0)

    torch.save(all_feats_train, f"./data/cifar10_all_feats_train_{model_name}.pt")
    torch.save(all_labels_train, f"./data/cifar10_all_labels_train_{model_name}.pt")
    torch.save(all_feats_test, f"./data/cifar10_all_feats_test_{model_name}.pt")
    torch.save(all_labels_test, f"./data/cifar10_all_labels_test_{model_name}.pt")

    print(f"Finished {model_name} feature extraction")
    print(f"Extracted feature size: {all_feats_train.shape}")
    

    