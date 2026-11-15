import os
import torch
import argparse
import time
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from dataset import get_dataloaders, HipMRIDataset, HipMRI3DDataset
from modules import UNet2D, UNet3D
from train import get_file_paths

def calculate_dice(y_true, y_pred, class_index, smooth=1e-5):
    """Calculate Dice coefficient for a specific class."""
    y_true_c = (y_true == class_index).astype(np.float32)
    y_pred_c = (y_pred == class_index).astype(np.float32)

    intersection = np.sum(y_true_c * y_pred_c)
    denominator = np.sum(y_true_c) + np.sum(y_pred_c)

    return (2.0 * intersection + smooth) / (denominator + smooth)

def evaluate_and_visualize(base_data_dir, weights_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load test data and model
    _, _, test_loader = get_dataloaders(base_data_dir, batch_size=4)
    model = UNet2D(in_channels=1, out_classes=6).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"2D U-Net Total Trainable Parameters: {total_params:,}")

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    # Fetch one test batch
    images, masks = next(iter(test_loader))

    # Crop the tensors to 256x128 BEFORE passing them to the model
    images = images[:, :, :, :128].to(device)
    masks = masks[:, :, :, :128].to(device)

    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)
        # Convert logits to class predictions (argmax across channel dimension)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    images = images.cpu().numpy()
    masks = torch.argmax(masks, dim=1).cpu().numpy()

    # Visualize the first sample in the batch[cite: 5, 13] (Gemini: "Generate matplotlib code for rendering these graphs, as per the code in this snippet"
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("MRI Input")
    plt.imshow(images[0, 0], cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(masks[0], cmap='nipy_spectral')

    plt.subplot(1, 3, 3)
    plt.title("Model Prediction")
    plt.imshow(preds[0], cmap='nipy_spectral')

    plt.tight_layout()
    plt.savefig("2d_prediction_output.png", bbox_inches='tight', dpi=300)
    plt.show()

def evaluate_patient_volume(patient_loader, weights_path, target_class_idx=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet2D(in_channels=1, out_classes=6).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_masks = []

    torch.cuda.reset_peak_memory_stats(device)
    total_volume_latency = 0.0

    with torch.no_grad():
        for images, masks in patient_loader:  # Assumes loader yields ordered slices for 1 patient
            images = images[:, :, :, :128].to(device)
            masks = masks[:, :, :, :128].numpy()

            start_time = time.time()
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            total_volume_latency += (time.time() - start_time)

            all_preds.append(preds)
            masks_argmax = np.argmax(masks, axis=1)
            all_masks.append(masks_argmax)

    # Stack 2D slices into a 3D volume: (Z, H, W)
    volume_preds = np.concatenate(all_preds, axis=0)
    volume_masks = np.concatenate(all_masks, axis=0)

    # Global 3D Dice Score
    global_dice = calculate_dice(volume_masks, volume_preds, class_index=target_class_idx)
    print(f"Global 3D Target Dice Score: {global_dice:.4f}")

    # Per-Slice Dice Score (for Base, Mid, Apex analysis)
    slice_dices = []
    for z in range(volume_preds.shape[0]):
        # Only evaluate slices where the ground truth contains the target class
        if np.any(volume_masks[z] == target_class_idx):
            sdice = calculate_dice(volume_masks[z], volume_preds[z], class_index=target_class_idx)
            slice_dices.append((z, sdice))

    if slice_dices:
        print(f"Base Slice Dice: {slice_dices[0][1]:.4f}")
        print(f"Mid-Volume Slice Dice: {slice_dices[len(slice_dices) // 2][1]:.4f}")
        print(f"Apex Slice Dice: {slice_dices[-1][1]:.4f}")


    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    print(f"2D Inference Latency: {total_volume_latency:.4f}s per volume")
    print(f"2D Peak VRAM Utilized: {peak_vram:.2f}GB")

    return volume_preds, volume_masks


def evaluate_patient_volume_3d(test_loader_3d, weights_path, target_class_idx=5):
    """
    Evaluates a 3D U-Net on a full patient volume, extracting global and per-slice metrics
    to investigate the base/apex boundary dilemma.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet3D(in_channels=1, out_classes=6, base_filters=16).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"3D U-Net Total Trainable Parameters: {total_params:,}")

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    total_global_dice = 0.0
    num_volumes = 0

    torch.cuda.reset_peak_memory_stats(device)  # Reset VRAM tracker
    all_latencies = []

    with torch.no_grad():
        for images, masks in test_loader_3d:
            images = images.to(device)
            masks = masks.numpy()

            start_time = time.time()  # Start latency timer
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            inference_time = time.time() - start_time  # End latency timer
            all_latencies.append(inference_time)
            
            masks_argmax = np.argmax(masks, axis=1)

            volume_preds = preds[0]
            volume_masks = masks_argmax[0]

            global_dice = calculate_dice(volume_masks, volume_preds, class_index=target_class_idx)
            total_global_dice += global_dice
            num_volumes += 1

            if num_volumes == 1 :
                slice_dices = []
                for z in range(volume_preds.shape[0]):
                    if np.any(volume_masks[z] == target_class_idx):
                        sdice = calculate_dice(volume_masks[z], volume_preds[z], class_index=target_class_idx)
                        slice_dices.append((z, sdice))

                if slice_dices:
                    print(f"Base Slice Dice (Superior): {slice_dices[0][1]:.4f}")
                    print(f"Mid-Volume Slice Dice (Central): {slice_dices[len(slice_dices) // 2][1]:.4f}")
                    print(f"Apex Slice Dice (Inferior): {slice_dices[-1][1]:.4f}")

                mid_z = slice_dices[len(slice_dices) // 2][0] if slice_dices else volume_preds.shape[0] // 2

                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.title(f"3D MRI Input (Slice {mid_z})")
                plt.imshow(images[0, 0, mid_z].cpu().numpy(), cmap='gray')

                plt.subplot(1, 3, 2)
                plt.title(f"Ground Truth Mask (Slice {mid_z})")
                plt.imshow(volume_masks[mid_z], cmap='nipy_spectral')

                plt.subplot(1, 3, 3)
                plt.title(f"3D Model Prediction (Slice {mid_z})")
                plt.imshow(volume_preds[mid_z], cmap='nipy_spectral')

                plt.tight_layout()
                plt.savefig(f"3d_prediction_slice_{mid_z}.png", bbox_inches='tight', dpi=300)
                plt.show()

    # Calculate and print the true test set average
    avg_test_dice = total_global_dice / num_volumes
    avg_latency = np.mean(all_latencies)
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    print("\n--- Final 3D Test Set Evaluation ---")
    print(f"Average 3D Test Set Dice Score: {avg_test_dice:.4f}")
    print(f"Average Inference Latency: {avg_latency:.4f}s per volume")
    print(f"Peak VRAM Utilized: {peak_vram:.2f}GB")

    return volume_preds, volume_masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 2D or 3D UNet models.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["2d", "3d", "both"],
        required=True,
        help="Evaluation mode: '2d', '3d', or 'both'",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the data directory. Defaults to './training data/keras_slices_data' for 2d and './training data' for 3d.",
    )
    args = parser.parse_args()

    if args.mode in ["2d", "both"]:
        data_dir_2d = (
            args.data_dir
            if args.data_dir
            else "./training data/keras_slices_data"
        )
        evaluate_and_visualize(data_dir_2d, "unet2d_baseline.pth")

    if args.mode in ["3d", "both"]:
        data_dir_3d = args.data_dir if args.data_dir else "./training data"
        img_dir = os.path.join(data_dir_3d, "semantic_MRs")
        mask_dir = os.path.join(data_dir_3d, "semantic_labels_only")

        all_imgs = get_file_paths(img_dir)
        all_masks = get_file_paths(mask_dir)

        test_imgs = all_imgs[189:]
        test_masks = all_masks[189:]

        test_dataset_3d = HipMRI3DDataset(test_imgs, test_masks)
        test_loader_3d = DataLoader(test_dataset_3d, batch_size=1, shuffle=False)

        evaluate_patient_volume_3d(test_loader_3d, "unet3d_best.pth")