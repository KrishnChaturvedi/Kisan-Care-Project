import kagglehub
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from timeit import default_timer as timer
from tqdm.auto import tqdm

from my_library import accuracy, print_train_time, train_step, test_step
from model import Disease_Classifier

# -----------------------------
# 1. Download dataset
# -----------------------------
path = kagglehub.dataset_download("loki4514/rice-leaf-diseases-detection")
dataset_dir = os.path.join(path, "Rice_Leaf_AUG", "Rice_Leaf_AUG")

# -----------------------------
# 2. Transformations
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# 3. Dataset + Split
# -----------------------------
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
classes = len(dataset.classes)
batch = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data, test_data = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_data, batch_size=batch, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch, shuffle=False)

# -----------------------------
# 4. Model + Loss
# -----------------------------
model = Disease_Classifier(label=classes, freeze_backbone=True).to(device)
loss_fn = nn.CrossEntropyLoss()

# -----------------------------
# 5. Optimizer builder function
# -----------------------------
def build_optimizer(model, unfreeze=False):
    if not unfreeze:  # only head + last block
        return torch.optim.Adam([
            {"params": model.base_model.fc.parameters(), "lr": 1e-3},
            {"params": model.base_model.layer4.parameters(), "lr": 1e-4}
        ])
    else:  # unfreeze all with tiered LR
        return torch.optim.Adam([
            {"params": model.base_model.fc.parameters(), "lr": 1e-3},
            {"params": model.base_model.layer4.parameters(), "lr": 1e-4},
            {"params": model.base_model.layer3.parameters(), "lr": 1e-5},
            {"params": model.base_model.layer2.parameters(), "lr": 1e-5}
        ])

optimizer = build_optimizer(model, unfreeze=False)

# Scheduler (attached once, we’ll rebuild if optimizer changes)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=3, verbose=True
)

# -----------------------------
# 6. Training Loop
# -----------------------------
epochs = 12

def main():
    global optimizer, scheduler
    start = timer()

    for epoch in tqdm(range(epochs)):
        print(f"\nEpoch: {epoch + 1}\n-----------")

        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, accuracy, device)
        test_loss, test_acc   = test_step(model, test_dataloader, loss_fn, accuracy, device)

        scheduler.step(test_loss)

        # Unfreeze backbone after some warmup
        if epoch == 5:
            print("🔓 Unfreezing all backbone layers for fine-tuning...")
            for param in model.base_model.parameters():
                param.requires_grad = True
            optimizer = build_optimizer(model, unfreeze=True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=3, verbose=True
            )

    end = timer()
    print_train_time(start, end, device)

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes
    }, "Disease_Model.pth")
    print("✅ Model saved successfully")

if __name__ == "__main__":
    main()

