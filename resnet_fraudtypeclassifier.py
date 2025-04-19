import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, models, transforms
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import random_split
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image transformations 
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(15), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

image_folder = '/home/nagarw48/Projects/DISML/fraud_images'  
dataset = datasets.ImageFolder(root=image_folder, transform=transform)

#splitting data
def stratified_split(dataset, train_ratio=0.8, val_ratio=0.2, random_seed=42):
    targets = [sample[1] for sample in dataset.samples]  
    class_indices = {i: [] for i in range(len(dataset.classes))}
    
    for idx, target in enumerate(targets):
        class_indices[target].append(idx)
    
    train_indices = []
    val_indices = []
    
    for class_id, indices in class_indices.items():
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        train_indices.extend(indices[:split_idx])
        val_indices.extend(indices[split_idx:])
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    return train_sampler, val_sampler

train_sampler, val_sampler = stratified_split(dataset)

train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

# Load ResNet18 model pre-trained with Dropout
model = models.resnet18(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),  
    nn.Linear(model.fc.in_features, len(dataset.classes))
)
model = model.to(device)

#loss function and optimizer with weight decay
criterion = nn.CrossEntropyLoss()
#l2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  

# Learning Rate Scheduler: reduce learning rate by 0.1 every 10 epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Early stopping parameters
patience = 5
best_loss = float('inf')
counter = 0

# evaluation metrics
def calculate_metrics(true_labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    return precision, recall, f1

# Train 
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, patience=5):
    epoch_loss = []
    best_model = None
    best_loss = float('inf')  
    counter = 0
    
    for epoch in range(10):  
        model.train()
        running_loss = 0.0
        true_labels = []
        predictions = []
        
       
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{10}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  
            outputs = model(inputs)  

            loss = criterion(outputs, labels)
            loss.backward()  
            optimizer.step()  

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

        epoch_loss.append(running_loss / len(train_loader.dataset))  
        precision, recall, f1 = calculate_metrics(true_labels, predictions)
        
        model.eval()
        val_loss = 0.0
        val_true_labels = []
        val_predictions = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                val_true_labels.extend(labels.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)  
        val_precision, val_recall, val_f1 = calculate_metrics(val_true_labels, val_predictions)
        
        print(f"Epoch {epoch+1}/{10} - Loss: {epoch_loss[-1]:.4f} - Val Loss: {val_loss:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1-Score: {val_f1:.4f}")

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            best_model = model.state_dict()  
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping due to no improvement")
                break
        
        # Step the scheduler
        scheduler.step()

    model.load_state_dict(best_model)
    return model, epoch_loss

# Train the model with scheduler
trained_model, loss_history = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, patience)

torch.save(trained_model.state_dict(), "fraud_detection_resnet18.pth")
print("Model saved successfully.")
