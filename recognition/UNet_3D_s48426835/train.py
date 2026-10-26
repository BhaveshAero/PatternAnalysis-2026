import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import HipMRIDataset
from modules import UNet2D


def per_class_dice(preds, targets, smooth=1e-6):
    """Calculates Dice for each of the 6 classes on the current tensor dimensions."""
    preds = preds.contiguous()
    targets = targets.contiguous()
    intersection = (preds * targets).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + smooth)
    return dice.mean(dim=0)


def get_file_paths(directory):
    """Rebuilds the explicit list of Nifti file paths your dataset requires."""
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.nii', '.nii.gz'))])


def train_model(base_data_dir, num_epochs=30, batch_size=16, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Map paths exactly to the folder structure shown in image_f87b76.png
    train_img_paths = get_file_paths(os.path.join(base_data_dir, 'keras_slices_train'))
    train_msk_paths = get_file_paths(os.path.join(base_data_dir, 'keras_slices_seg_train'))
    val_img_paths = get_file_paths(os.path.join(base_data_dir, 'keras_slices_validate'))
    val_msk_paths = get_file_paths(os.path.join(base_data_dir, 'keras_slices_seg_validate'))

    # 2. Initialize Datasets and DataLoaders
    train_dataset = HipMRIDataset(train_img_paths, train_msk_paths)
    val_dataset = HipMRIDataset(val_img_paths, val_msk_paths)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Initialize Model, Optimizer, and Weighted Loss
    model = UNet2D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0, 50.0]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_prostate_dice = 0.0

    # 4. Training Loop
    for epoch in range(num_epochs):
        model.train()
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            # Crop tensors to 256x128 before forward pass
            images = images[:, :, :, :128].to(device)
            masks = masks[:, :, :, :128].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        # 5. Validation Loop
        model.eval()
        epoch_prostate_dice = 0.0

        with torch.no_grad():
            for val_images, val_masks in val_loader:
                # Crop validation tensors
                val_images = val_images[:, :, :, :128].to(device)
                val_masks = val_masks[:, :, :, :128].to(device)

                val_outputs = model(val_images)
                val_preds = F.one_hot(torch.argmax(val_outputs, dim=1), num_classes=6).permute(0, 3, 1, 2).float()

                batch_dice_scores = per_class_dice(val_preds, val_masks)
                epoch_prostate_dice += batch_dice_scores[5].item()

        avg_prostate_dice = epoch_prostate_dice / len(val_loader)
        print(f"Epoch {epoch + 1} - Prostate Val Dice: {avg_prostate_dice:.4f}")

        if avg_prostate_dice > best_prostate_dice:
            best_prostate_dice = avg_prostate_dice
            torch.save(model.state_dict(), 'unet2d_baseline.pth')
            print("New best prostate score! Overwriting baseline weights.")