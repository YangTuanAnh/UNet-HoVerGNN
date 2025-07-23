import os
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from config import Config
from dataset import SegmentationDataset
from loss_function import HoverLoss
from training_loop import train, final_evaluate
from unet_hovergnn import GraphHoverNet
from visualization import plot_validation_samples


def parse_args():
    parser = argparse.ArgumentParser(description="Train and Evaluate UNet-HoVerGNN")
    parser.add_argument("--dataset", type=str, default=Config.DATASET,
                        help="Dataset name (e.g., MoNuSAC, PanNuke, CoNSeP_Tiled)")
    parser.add_argument("--num_classes", type=int, default=Config.NUM_CLASSES,
                        help="Number of classes in the dataset")
    parser.add_argument("--data_path", type=str, default=Config.DATA_PATH,
                        help="Path to dataset root directory")
    parser.add_argument("--output_path", type=str, default=Config.OUTPUT_PATH,
                        help="Path to save outputs (logs, checkpoints, metrics)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override Config values if provided
    Config.DATASET = args.dataset
    Config.NUM_CLASSES = args.num_classes
    Config.DATA_PATH = args.data_path
    Config.OUTPUT_PATH = args.output_path

    # Ensure output path exists
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)

    # === Transformations ===
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    dataset_path = os.path.join(Config.DATA_PATH, Config.DATASET)

    # === Datasets and DataLoaders ===
    train_dataset = SegmentationDataset(dataset_path, split="train", transform=transform)
    val_dataset = SegmentationDataset(dataset_path, split="val", transform=transform)
    test_dataset = SegmentationDataset(dataset_path, split="test", transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # === Model, Loss, and Training ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphHoverNet(num_classes=Config.NUM_CLASSES).to(device)
    criterion = HoverLoss()

    # Train model
    train(model, criterion, train_dataloader, val_dataloader, device,
          num_classes=Config.NUM_CLASSES, epochs=Config.STAGE_EPOCH, patience=Config.PATIENCE)

    # Evaluate model (pretraining and finetuning)
    final_evaluate(model, test_dataloader, criterion, device, "pretrain", num_classes=Config.NUM_CLASSES)
    final_evaluate(model, test_dataloader, criterion, device, "finetune", num_classes=Config.NUM_CLASSES)

    # Plot validation samples
    plot_validation_samples(model, test_dataloader, device, Config.NUM_CLASSES)
