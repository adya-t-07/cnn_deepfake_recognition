import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Import custom components from your src package
from src.dataset import Data
from src.model import DeepfakeViTWrapper 

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc="ViT Training")
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device).long() # Keep flat as integers [batch] for CrossEntropyLoss
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass & Optimization
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * images.size(0)
        
        # CrossEntropy logits evaluation: grab the index of the highest logit value
        predictions = torch.argmax(outputs, dim=1)
        correct_preds += (predictions == labels).sum().item()
        total_samples += images.size(0)
        
        progress_bar.set_postfix(loss=loss.item(), acc=correct_preds / total_samples)
        
    return running_loss / total_samples, correct_preds / total_samples

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="ViT Validation"):
            images = images.to(device)
            labels = labels.to(device).long()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            predictions = torch.argmax(outputs, dim=1)
            correct_preds += (predictions == labels).sum().item()
            total_samples += images.size(0)
            
    return running_loss / total_samples, correct_preds / total_samples

def main():
    # 1. Hyperparameters Optimized for Transformers
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 5e-5  # ViTs need a smaller learning rate than CNNs to stabilize self-attention
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using hardware device: {DEVICE} | Initializing Vision Transformer Branch")
    os.makedirs('weights', exist_ok=True)

    # 2. Image Transformations
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])

    # 3. Load Datasets
    print("Loading Datasets...")
    # Adjust paths if your dataset location differs locally
    train_dataset = Data(root_dir='/content/cnn_deepfake_recognition/data/external/rvf10k/train', transform=train_transforms)
    val_dataset = Data(root_dir='/content/cnn_deepfake_recognition/data/external/rvf10k/valid', transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Model, Loss, & Optimizer Initialization
    print("Loading ViT-B/16 Architecture with Pretrained ImageNet Head...")
    model = DeepfakeViTWrapper(pretrained=True).to(DEVICE)
    
    # CrossEntropy expects multi-class logit arrays [Batch, 2] and targets as class indices
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 5. Training Loop
    best_val_acc = 0.0
    print(f"Beginning optimization execution for {EPOCHS} epochs...\n")
    
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%\n")
        
        # Save independent ViT weights file
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'weights/vit_baseline.pth')
            print(f"✨ New best validation checkpoint locked! Accuracy: {best_val_acc*100:.2f}% -> Weights saved to weights/vit_baseline.pth.")

    print("ViT Training completely finished.")

if __name__ == '__main__':
    main()