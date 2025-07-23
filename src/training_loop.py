from tqdm import tqdm
import torch
from torch.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
from postprocess import get_node_labels_from_coords, postprocess_hovernet_output
import os
from config import Config
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def train_one_epoch(model, dataloader, optimizer, criterion, device, num_classes=5):
    model.train()
    scaler = GradScaler()

    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    total_pq = 0.0
    total_f1_per_class = torch.zeros(num_classes, dtype=torch.float32, device=device)
    count = 0

    loop = tqdm(dataloader, desc="Training", leave=True)
    for image, mask, h_grads, v_grads in loop:
        image = image.to(device)
        mask = mask.to(device).long()
        h_grads = h_grads.to(device)
        v_grads = v_grads.to(device)

        nc_targets = mask
        np_targets = (mask > 0).long()

        optimizer.zero_grad()

        with autocast(device.type):
            np_logits, hv_logits, nc_logits, centroids, gc_logits = model(image)

            # Get graph branch labels
            if gc_logits is not None and centroids.shape[0] > 0:
                gc_targets = get_node_labels_from_coords(centroids, nc_targets)
                valid = gc_targets != 0
                if valid.any():
                    gc_input = (gc_logits[valid], gc_targets[valid])
                else:
                    gc_input = (None, None)
            else:
                gc_input = (None, None)

            # Compute total loss
            loss = criterion(
                np_logits, np_targets,
                hv_logits, h_grads, v_grads,
                nc_logits, nc_targets,
                *gc_input
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            nc_pred = torch.argmax(nc_logits, dim=1)
            tp, fp, fn, tn = smp.metrics.get_stats(
                nc_pred, nc_targets,
                mode='multiclass',
                num_classes=num_classes  # assuming class 0 is background
            )

            iou_per_class = smp.metrics.iou_score(tp, fp, fn, tn, reduction='none').mean(dim=0)
            f1_per_class = smp.metrics.f1_score(tp, fp, fn, tn, reduction='none').mean(dim=0)
            pq_per_class = 2 * iou_per_class * f1_per_class / (iou_per_class + f1_per_class + 1e-8)

        total_loss += loss.item()
        total_iou += iou_per_class[1:].mean().item()
        total_f1 += f1_per_class[1:].mean().item()
        total_pq += pq_per_class[1:].mean().item()
        total_f1_per_class += f1_per_class.to(device)  # accumulate per class
        count += 1

        loop.set_postfix(
            loss=total_loss / count,
            iou=total_iou / count,
            f1=total_f1 / count,
            pq=total_pq / count,
            f1_per_class=[round(x.item() / count, 4) for x in f1_per_class][1:]
        )

    return (
        total_loss / count,
        total_iou / count,
        total_f1 / count,
        total_pq / count,
        (total_f1_per_class / count).tolist()
    )

@torch.no_grad()
def validate(model, dataloader, criterion, device, num_classes=5, post_process=False):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_f1 = 0.0
    total_pq = 0.0
    total_f1_per_class = torch.zeros(num_classes, dtype=torch.float32, device=device)
    count = 0

    loop = tqdm(dataloader, desc="Validation", leave=True)
    for image, mask, h_grads, v_grads in loop:
        image = image.to(device)
        mask = mask.to(device).long()
        h_grads = h_grads.to(device)
        v_grads = v_grads.to(device)

        nc_targets = mask
        np_targets = (mask > 0).long()

        np_logits, hv_logits, nc_logits, centroids, gc_logits = model(image)

        # Process graph branch if available
        if gc_logits is not None and centroids.shape[0] > 0:
            gc_targets = get_node_labels_from_coords(centroids, nc_targets)
            valid = gc_targets != 0
            if valid.any():
                gc_input = (gc_logits[valid], gc_targets[valid])
            else:
                gc_input = (None, None)
        else:
            gc_input = (None, None)

        # Compute total loss
        loss = criterion(
            np_logits, np_targets,
            hv_logits, h_grads, v_grads,
            nc_logits, nc_targets,
            *gc_input
        )

        if post_process:
            nc_pred = postprocess_hovernet_output(np_logits, hv_logits, nc_logits, device)
        else:
            nc_pred = torch.argmax(nc_logits, dim=1)

        tp, fp, fn, tn = smp.metrics.get_stats(
            nc_pred, nc_targets,
            mode='multiclass',
            num_classes=num_classes   # if class 0 is background
        )

        iou_per_class = smp.metrics.iou_score(tp, fp, fn, tn, reduction='none').mean(dim=0)  # [1:] to skip background
        f1_per_class = smp.metrics.f1_score(tp, fp, fn, tn, reduction='none').mean(dim=0)
        pq_per_class = 2 * iou_per_class * f1_per_class / (iou_per_class + f1_per_class + 1e-8)

        total_loss += loss.item()
        total_iou += iou_per_class[1:].mean().item()
        total_f1 += f1_per_class[1:].mean().item()
        total_pq += pq_per_class[1:].mean().item()
        total_f1_per_class += f1_per_class.to(device)  # accumulate
        count += 1

        loop.set_postfix(
            loss=total_loss / count,
            iou=total_iou / count,
            f1=total_f1 / count,
            pq=total_pq / count,
            f1_per_class=[round(x.item() / count, 4) for x in f1_per_class][1:]
        )

    return (
        total_loss / count,
        total_iou / count,
        total_f1 / count,
        total_pq / count,
        (total_f1_per_class / count).tolist()
    )

def train(model, criterion, train_loader, val_loader, device, 
          epochs=20, lr=1e-4, num_classes=5, patience=5):
    
    def train_loop(stage_name, use_graph, start_epoch=1, lr=lr):
        nonlocal best_val_pq
        nonlocal epochs_no_improve

        best_val_pq = -float('inf')
        epochs_no_improve = 0
        
        print(f"\n=== Starting {stage_name} phase ===")
        model.set_stage('pretrain' if not use_graph else 'finetune')

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            current_epoch = start_epoch + epoch
            print(f"\nEpoch {current_epoch}/{start_epoch + epochs - 1} [{stage_name}]")
            
            train_loss, train_iou, train_f1, train_pq, train_f1_per_class = train_one_epoch(
                model, train_loader, optimizer, criterion, device, num_classes)
            val_loss, val_iou, val_f1, val_pq, val_f1_per_class = validate(
                model, val_loader, criterion, device, num_classes)

            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_ious.append(train_iou)
            val_ious.append(val_iou)
            train_f1s.append(train_f1)
            val_f1s.append(val_f1)
            train_pqs.append(train_pq)
            val_pqs.append(val_pq)
    
            print(f"Train Loss: {train_loss:.4f} | IoU: {train_iou:.4f} | F1: {train_f1:.4f} | PQ: {train_pq:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | IoU: {val_iou:.4f} | F1: {val_f1:.4f} | PQ: {val_pq:.4f}")
            
            # Print per-class F1 (excluding background class 0)
            train_f1_str = " | ".join(f"Class {i+1}: {x:.4f}" for i, x in enumerate(train_f1_per_class[1:]))
            val_f1_str = " | ".join(f"Class {i+1}: {x:.4f}" for i, x in enumerate(val_f1_per_class[1:]))
            print(f"Train F1 per class: {train_f1_str}")
            print(f"Val   F1 per class: {val_f1_str}")
    
            if val_pq > best_val_pq:
                best_val_pq = val_pq
                epochs_no_improve = 0
                torch.save(model.state_dict(), f"best_hovernet_{stage_name}.pt")
                print("Saved best model (based on PQ).")
            else:
                epochs_no_improve += 1
                print(f"No improvement in PQ for {epochs_no_improve} epoch(s).")
    
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement in PQ.")
                break
        
        return current_epoch + 1

    # Track metrics
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    train_f1s, val_f1s = [], []
    train_pqs, val_pqs = [], []

    best_val_pq = -float('inf')
    epochs_no_improve = 0

    next_epoch = train_loop(stage_name='pretrain', use_graph=False, start_epoch=1)
    next_epoch = train_loop(stage_name='finetune', use_graph=True, start_epoch=next_epoch, lr=1e-2)

    print(f"\nTraining complete. Best Val PQ: {best_val_pq:.4f}")

    # Create output directory
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # === Plotting ===
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(epochs_range, train_ious, label='Train IoU')
    plt.plot(epochs_range, val_ious, label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU over Epochs')
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(epochs_range, train_f1s, label='Train F1')
    plt.plot(epochs_range, val_f1s, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 over Epochs')
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(epochs_range, train_pqs, label='Train PQ')
    plt.plot(epochs_range, val_pqs, label='Val PQ')
    plt.xlabel('Epoch')
    plt.ylabel('PQ')
    plt.title('PQ over Epochs')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(Config.OUTPUT_PATH, f"training_metrics_{timestamp}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved training metrics plot to {plot_path}")

    # === Save metrics ===
    metrics_df = pd.DataFrame({
        "Epoch": list(epochs_range),
        "Train Loss": train_losses,
        "Val Loss": val_losses,
        "Train IoU": train_ious,
        "Val IoU": val_ious,
        "Train F1": train_f1s,
        "Val F1": val_f1s,
        "Train PQ": train_pqs,
        "Val PQ": val_pqs,
    })
    csv_path = os.path.join(Config.OUTPUT_PATH, f"training_metrics_{timestamp}.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved training metrics to {csv_path}")

@torch.no_grad()
def final_evaluate(model, test_loader, criterion, device, stage_name, num_classes=6):
    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)

    # Load best model checkpoint
    model_path = os.path.join(Config.OUTPUT_PATH, f"best_hovernet_{stage_name}.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.set_stage(stage_name)

    # Perform validation
    test_loss, test_iou, test_f1, test_pq, f1_per_class = validate(
        model, test_loader, criterion, device, num_classes, post_process=True
    )

    # Print results
    print(f"\nFinal Test | Loss: {test_loss:.4f} | IoU: {test_iou:.4f} | F1: {test_f1:.4f} | PQ: {test_pq:.4f}")
    print("F1 per class (excluding background):", f1_per_class)

    # Save metrics to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(Config.OUTPUT_PATH, f"final_eval_{stage_name}_{timestamp}.csv")

    metrics = {
        "Loss": [test_loss],
        "IoU": [test_iou],
        "F1": [test_f1],
        "PQ": [test_pq],
    }
    # Add F1 per class (C1...Cn)
    for i, f1 in enumerate(f1_per_class, start=1):
        metrics[f"F1_Class_{i}"] = [f1]

    pd.DataFrame(metrics).to_csv(results_file, index=False)
    print(f"Saved final evaluation metrics to {results_file}")

if __name__ == "__main__":
    from config import Config
    from unet_hovergnn import GraphHoverNet
    from loss_function import HoverLoss
    import segmentation_models_pytorch as smp
    from dataset import SegmentationDataset
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from visualization import visualize_hovernet_output
    import os

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts HWC to CHW and scales to [0, 1]
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # example: scale to [-1, 1]
    ])

    dataset_path = os.path.join(Config.DATA_PATH, Config.DATASET)

    train_dataset = SegmentationDataset(dataset_path, split="train", transform=transform)
    val_dataset = SegmentationDataset(dataset_path, split="val", transform=transform)
    test_dataset = SegmentationDataset(dataset_path, split="test", transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphHoverNet(num_classes=Config.NUM_CLASSES)
    model = model.to(device)
    criterion = HoverLoss()

    train(model, criterion, train_dataloader, val_dataloader, device, num_classes=Config.NUM_CLASSES, epochs=2, patience=2)

    final_evaluate(model, test_dataloader, criterion, device, "pretrain", num_classes=Config.NUM_CLASSES)

    final_evaluate(model, test_dataloader, criterion, device, "finetune", num_classes=Config.NUM_CLASSES)