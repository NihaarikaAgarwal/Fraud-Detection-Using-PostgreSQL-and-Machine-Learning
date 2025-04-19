import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from torchvision import models
from torch.utils.data import SubsetRandomSampler
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

image_folder = '/home/nagarw48/Projects/DISML/fraud_images'  
dataset = datasets.ImageFolder(root=image_folder, transform=transform)

# Stratified Split - 80% train, 20% validation per class (for test data)
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

_, val_sampler = stratified_split(dataset)

val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

model = models.resnet18(pretrained=False) 
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),  
    nn.Linear(model.fc.in_features, len(dataset.classes))
)
model = model.to(device)

model.load_state_dict(torch.load("fraud_type_classification_resnet18_10.pth"))
model.eval()  

# evaluation metrics
def calculate_metrics(true_labels, predictions):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='macro')
    return precision, recall, f1

#Evaluate the model on the test (validation) data
def evaluate_model(model, val_loader):
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():  
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()  
            all_predictions.extend(predicted.cpu().numpy())  
            all_labels.extend(labels.cpu().numpy())  

    accuracy = correct / total  
    precision, recall, f1 = calculate_metrics(all_labels, all_predictions)
    
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    
    return accuracy, precision, recall, f1

evaluate_model(model, val_loader)
