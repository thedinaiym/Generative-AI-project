import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, models, datasets
import os

IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class PetDataset(Dataset):
    def __init__(self, root, transform=None):
        self.oxford = datasets.OxfordIIITPet(
            root=root, 
            download=True, 
            target_types=["binary-category"], 
            transform=transform
        )
    def __len__(self):
        return len(self.oxford)
    def __getitem__(self, idx):
        image, target = self.oxford[idx]
        if isinstance(target, tuple):
            target = target[0]
        return image, target

def train_model():
    root_dir = "./data"
    dataset = PetDataset(root=root_dir, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"Starting epoch {epoch+1}")
        for batch_idx, (images, species) in enumerate(train_loader):
            images = images.to(device)
            species = species.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, species)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, species in val_loader:
                images = images.to(device)
                species = species.to(device).float().unsqueeze(1)
                outputs = model(images)
                preds = torch.sigmoid(outputs) >= 0.5
                correct += (preds.float() == species).sum().item()
                total += species.size(0)
        val_acc = correct / total
        print(f"Validation Accuracy: {val_acc:.4f}")

    model_path = "cat_dog_classifier.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train_model()

