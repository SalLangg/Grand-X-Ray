#==================================================
#IMPORTS
#==================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision.transforms import v2
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

from tqdm.auto import tqdm
from torch.amp import GradScaler, autocast

#==================================================
#FUNCTIONS
#==================================================


def save_model(model, optimizer, scheduler, train_losses, val_losses, epoch, filepath='best_model.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_architecture': 'efficientnet_b1'
    }, filepath)
    print(f"Modes saved as: {filepath}")


def load_model(filepath='best_model.pth'):
    checkpoint = torch.load(filepath)

    model = models.efficientnet_b1()
    # for param in model.features[:-4].parameters():
    #     param.requires_grad = False
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 14)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Uploaded a model trained up to the epoch: {checkpoint['epoch']}")
    
    return model, checkpoint


def train_model():
    train_losses = []
    val_losses = []

    for ep in range(EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        train_tqdm = tqdm(train_loader, desc=f'Epoch {ep+1}/{EPOCHS} [Train]', leave=False)

        for t_img, t_label in train_tqdm:
            t_img, t_label = t_img.to(device, non_blocking=True), t_label.to(device, non_blocking=True)
            
            with autocast(device_type=device):
                t_predict = model(t_img)
                t_loss = loss_f(t_predict, t_label)
            
            optimizer.zero_grad()
            scaler.scale(t_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_train_loss += t_loss.item()
            train_tqdm.set_postfix({'Loss': f'{t_loss.item():.4f}'})
        
        scheduler.step()

        #---Валидация
        model.eval()
        epoch_val_loss = 0.0
        val_tqdm = tqdm(val_loader, desc=f'Epoch {ep+1}/{EPOCHS} [Validation]', leave=False)
        
        with torch.no_grad():
            for v_img, v_label in val_tqdm:
                v_img, v_label = v_img.to(device), v_label.to(device)

                v_predict = model(v_img)
                v_loss = loss_f(v_predict, v_label)

                epoch_val_loss += v_loss.item()
                val_tqdm.set_postfix({'Val Loss': f'{v_loss.item():.4f}'})

        epoch_train_loss /= len(train_loader)
        epoch_val_loss /= len(val_loader)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(f'===== Epoch {ep+1}/{EPOCHS} =====')
        print(f'---- Train Loss: {epoch_train_loss:.4f}')
        print(f'---- Val Loss: {epoch_val_loss:.4f}')
    
    return train_losses, val_losses


def predict_test(model, test_csv_file, test_img_dir, transform, device, batch_size=64):
    test_dataset = XRayData(test_csv_file, test_img_dir, transform=transform, mode='inf')

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count()),
        pin_memory=True
    )
    
    model.eval()
    predictions = []
    image_names = []
    
    with torch.no_grad():
        for images, names in tqdm(test_loader, desc="Predicting on test set"):
            images = images.to(device)
            
            with autocast(device_type=device):
                outputs = model(images)
                probs = torch.sigmoid(outputs)
            
            predictions.extend(probs.cpu().numpy())
            image_names.extend(names)
    
    return image_names, predictions


def create_submit(image_names, predictions, output_file='submission.csv'):
    class_names = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 
        'Lung Opacity', 'No Finding', 'Pleural Effusion', 
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]
    
    submission_df = pd.DataFrame(predictions, columns=class_names)
    submission_df.insert(0, 'Image_name', image_names)
    
    submission_df.to_csv(output_file, index=False)

    print("Prediction statistics:")
    for class_name in class_names:
        print(f"{class_name}: mean={submission_df[class_name].mean():.4f}")
    
    return submission_df


#==================================================
#CLASSES
#==================================================


class XRayData(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, mode='train'):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = self.df['Image_name'].values
        self.mode = mode

        if self.mode == 'train':
            self.labels = self.df.drop(
                columns=['Image_name','Patient_ID','Study','Sex','Age','ViewCategory','ViewPosition']
            ).values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name) 
        
        with Image.open(img_path) as image:
            if self.transform:
                image = self.transform(image)
        if self.mode == 'train':
            labels = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, labels
        else:
            return image, img_name
    

#==================================================
#MAIN
#==================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

data = pd.read_csv("train1.csv")
classes = data.drop(columns=['Image_name','Patient_ID','Study','Sex','Age','ViewCategory','ViewPosition'])

transform = v2.Compose([
    v2.ToImage(),
    v2.Grayscale(num_output_channels=3),
    v2.Resize((240, 240)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.4887, 0.4887, 0.4887], std=[0.2780, 0.2780, 0.2780]), #0.4887, std: 0.2780
])

dataset = XRayData("train1.csv", "train1", transform=transform)
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=min(4, os.cpu_count()), # In Jupyter - set 0!
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=64, 
    shuffle=False,
    num_workers=min(4, os.cpu_count()), 
    pin_memory=True,
)

#--- Modes settings
model = models.efficientnet_b1()

# for param in model.features[:-4].parameters():
#     param.requires_grad = False
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 14)
model = model.to(device)

loss_f = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
scaler = GradScaler(enabled=(device == 'cuda'))  # Automatic activation for CUDA - is attempt to optimize the learning rate on the Gpu

EPOCHS = 20

if __name__ == '__main__':
    print("Training start...")
    train_losses, val_losses = train_model()

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()

    print("Saving model...")
    save_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_losses=train_losses,
        val_losses=val_losses,
        epoch=EPOCHS,
        filepath='xray_classifier_model.pth'
    )
    
    trained_model, checkpoint = load_model('xray_classifier_model.pth')
    trained_model = trained_model.to(device)
    
    print("Making predictions...")
    test_image_names, test_predictions = predict_test(
        model=trained_model,
        test_csv_file="sample_submission_1.csv", 
        test_img_dir="test1",
        transform=transform,
        device=device
    )
    
    print("Creating submission file...")
    submission_df = create_submit(
        image_names=test_image_names,
        predictions=test_predictions,
        output_file='submission.csv'
    )
    
    print("Submission file created!")