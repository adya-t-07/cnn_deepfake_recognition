import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import our custom components
from src.dataset import Data
from src.model import CNN

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    # tqdm adds a beautiful, real-time progress bar to your terminal
    progress_bar = tqdm(dataloader, desc="Training")
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1) # Reshape labels from [batch] to [batch, 1]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass & Optimization
        loss.backward()
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * images.size(0)
        
        # Since outputs are raw logits, a value > 0 means class 1 (Fake), <= 0 means class 0 (Real)
        predictions = (outputs > 0.0).float()
        correct_preds += (predictions == labels).sum().item()
        total_samples += images.size(0)
        
        # Update progress bar message dynamically
        progress_bar.set_postfix(loss=loss.item(), acc=correct_preds / total_samples)
        
    return running_loss / total_samples, correct_preds / total_samples

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    
    # 💡 Accumulators for the confusion matrix
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculations for pure validation speed
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            predictions = (outputs > 0.0).float()
            correct_preds += (predictions == labels).sum().item()
            total_samples += images.size(0)

            # 💡 Save predictions and ground truth labels to CPU arrays
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
    # Compute the final epoch metrics
    epoch_loss = running_loss / total_samples
    epoch_acc = correct_preds / total_samples
    cm = confusion_matrix(all_labels, all_preds)
    return epoch_loss, epoch_acc, cm

def save_confusion_matrix(cm, epoch, model_name="cnn"):
    """
    Generates and saves a visual confusion matrix plot.
    """
    plt.figure(figsize=(6, 5))
    
    # Define labels matching your dataset folder structure layout (alphabetical order usually)
    # e.g., if folders are 'fake' and 'real', 'fake' is 0, 'real' is 1
    class_names = ['Fake', 'Real'] 
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # Plot with a nice professional blue color scheme
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    
    plt.title(f"{model_name.upper()} Confusion Matrix - Epoch {epoch}")
    
    # Create an output directory for metrics if it doesn't exist
    os.makedirs('metrics', exist_ok=True)
    
    # Save the file out cleanly
    plt.savefig(f'metrics/{model_name}_cm_epoch_{epoch}.png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"📊 Confusion Matrix plot successfully saved to: metrics/{model_name}_cm_epoch_{epoch}.png")

def main():
    # 1. Hyperparameters & Configuration
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using hardware device: {DEVICE}")
    
    # Create weights folder if it doesn't exist yet
    os.makedirs('weights', exist_ok=True)

    # 2. Image Transformations (ImageNet standards)
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([
        # 💡 ADD THIS: Jitter the boundary coordinates exactly like the ViT pipeline
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
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

    # 3. Instantiate Custom Datasets and PyTorch DataLoaders
    print("Loading Datasets...")
    train_dataset = Data(root_dir='/content/cnn_deepfake_recognition/data/external/rvf10k/train', transform=train_transforms)
    val_dataset = Data(root_dir='/content/cnn_deepfake_recognition/data/external/rvf10k/valid', transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Initialize the CNN Model Wrapper
    print("Initializing ResNet50 Baseline Model...")
    model = CNN(pretrained=True).to(DEVICE)

    # 5. Define Loss Engine and Optimization Strategy
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 6. The Execution Training Loop
    best_val_acc = 0.0
    print(f"Beginning optimization execution for {EPOCHS} epochs...\n")
    
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, epoch_cm = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%\n")
        save_confusion_matrix(epoch_cm, epoch=epoch+1, model_name="cnn_baseline")
        
        # Save the best weights snapshot if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'weights/cnn_baseline.pth')
            print(f"✨ New best validation checkpoint locked! Accuracy: {best_val_acc*100:.2f}% -> Weights saved.")

    print("Training completely finished.")

if __name__ == '__main__':
    main()