import torch
from torch import nn, optim
from src.config import device

def train_model(model, name, train_loader, val_loader):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    best_acc = 0
    for epoch in range(30):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = evaluate(model, val_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{name}_best.pth")
    return model

def evaluate(model, val_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
