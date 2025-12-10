import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import time

# --- Configuration ---
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
CHECKPOINT_DIR = 'checkpoints'
DATA_DIR = './data'

# --- 1. Dataset Loading & Augmentation ---
def get_dataloaders(batch_size=128):
    print(f"Preparing Data (Batch Size: {batch_size})...")
    
    # Data Augmentation for Training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])
    
    # Normalization for Test
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True,
                                            download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False,
                                           download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

# --- 2. Custom CNN Architecture ---
class CustomResNet(nn.Module):
    """
    A simplified ResNet-like architecture.
    Structure:
    - Conv7x7 -> BN -> ReLU -> MaxPool
    - Residual Block 1
    - Residual Block 2
    - Global Avg Pool -> FC
    """
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual Blocks
        self.layer1 = self._make_layer(64, 64)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_planes, out_planes, stride=1):
        layers = []
        layers.append(ResidualBlock(in_planes, out_planes, stride))
        layers.append(ResidualBlock(out_planes, out_planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ResidualBlock(nn.Module):
    """ Basic Residual Block """
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# --- 3. Training Loop ---
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    acc = 100. * correct / total
    return running_loss / len(trainloader), acc

def evaluate(model, testloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    return running_loss / len(testloader), acc

def save_checkpoint(model, optimizer, epoch, acc, path):
    print(f"Saving checkpoint to {path}...")
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'acc': acc
    }
    torch.save(state, path)

# --- Main ---
def main():
    print(f"Using device: {DEVICE}")
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    # Get Data
    trainloader, testloader = get_dataloaders(BATCH_SIZE)
    
    # Initialize Model
    model = CustomResNet(num_classes=10).to(DEVICE)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)
    
    # Training Loop
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        train_loss, train_acc = train(model, trainloader, optimizer, criterion, DEVICE)
        val_loss, val_acc = evaluate(model, testloader, criterion, DEVICE)
        
        scheduler.step()
        
        end_time = time.time()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Time: {end_time - start_time:.1f}s | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, 
                          os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            
    print(f"Training Complete. Best Accuracy: {best_acc:.2f}%")
    print(f"Model saved to {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}")

if __name__ == "__main__":
    main()
