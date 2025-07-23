import os
import matplotlib.pyplot as plt
from config import Config
from datetime import datetime
from postprocess import postprocess_hovernet_output
import torch

def visualize_hovernet_output(pred_inst, filename=None):
    """
    Visualize and save the instance segmentation map with optional type labels and centroids.

    Args:
        pred_inst (np.ndarray): [H, W] array of instance IDs.
        filename (str, optional): Custom name of the output PNG file. 
                                  If None, a timestamp-based filename is generated.
    """
    # Generate timestamped filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"hovernet_output_{timestamp}.png"

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(pred_inst, cmap="jet")
    ax.set_title("Instance Segmentation")
    ax.axis("off")
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)

    # Save PNG file
    output_file = os.path.join(Config.OUTPUT_PATH, filename)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved HoverNet visualization to: {output_file}")

@torch.no_grad()
def plot_validation_samples(model, dataloader, device, num_classes=6, num_samples=10):
    model.eval()
    samples_shown = 0

    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)  # Ensure output directory exists
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for image, mask, h_grads, v_grads in dataloader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True).long()
        h_grads = h_grads.to(device, non_blocking=True)
        v_grads = v_grads.to(device, non_blocking=True)

        np_logits, hv_logits, nc_logits, _, _ = model(image)
        np_pred = torch.argmax(np_logits, dim=1)
        nc_pred = torch.argmax(nc_logits, dim=1)

        postprocessed_pred = postprocess_hovernet_output(np_logits, hv_logits, nc_logits, device)
        
        B = image.size(0)
        for i in range(B):
            if samples_shown >= num_samples:
                return

            fig, axes = plt.subplots(1, 7, figsize=(21, 4))

            # Input image
            axes[0].imshow(image[i].permute(1, 2, 0).cpu().numpy())
            axes[0].set_title("Input Image")
            axes[0].axis('off')

            # Ground truth mask
            axes[1].imshow(mask[i].cpu().numpy(), cmap="jet")
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')

            # Predicted mask
            axes[2].imshow(nc_pred[i].cpu().numpy(), cmap="jet")
            axes[2].set_title("Prediction")
            axes[2].axis('off')

            # Horizontal gradient
            axes[3].imshow(hv_logits[i, 0].cpu().numpy(), cmap='viridis')
            axes[3].set_title("HoVer Horizontal")
            axes[3].axis('off')

            # Vertical gradient
            axes[4].imshow(hv_logits[i, 1].cpu().numpy(), cmap='viridis')
            axes[4].set_title("HoVer Vertical")
            axes[4].axis('off')

            axes[5].imshow(np_pred[i].cpu().numpy(), cmap='gray')
            axes[5].set_title("Binary Segmentation")
            axes[5].axis('off')

            axes[6].imshow(postprocessed_pred[i].cpu().numpy(), cmap="jet")
            axes[6].set_title("Post Processed")
            axes[6].axis('off')

            plt.tight_layout()

            # Save the figure
            filename = f"val_sample_{samples_shown}_{timestamp}.png"
            save_path = os.path.join(Config.OUTPUT_PATH, filename)
            plt.savefig(save_path, dpi=300)
            plt.close(fig)

            print(f"Saved validation sample to {save_path}")
            samples_shown += 1