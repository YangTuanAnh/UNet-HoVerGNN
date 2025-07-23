import numpy as np
import torch
import cv2
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

def postprocess_hovernet_output(np_logits, hv_logits, nc_logits, device, return_centroids=False, nr_types=6):
    """
    Postprocess HoVerNet outputs to obtain instance segmentation and class predictions.

    Args:
        np_logits (Tensor): [B, 1, H, W] - nuclei presence logits.
        hv_logits (Tensor): [B, 2, H, W] - horizontal and vertical offset logits.
        nc_logits (Tensor): [B, C, H, W] - nuclear classification logits.
        device (torch.device): Device for output tensors.
        return_centroids (bool): Placeholder for future use.
        nr_types (int): Number of nuclear classes.

    Returns:
        Tensor: [B, H, W] instance segmentation with class IDs.
    """
    def process(pred_map: torch.Tensor) -> torch.Tensor:
        """
        Process a single HoVerNet prediction map to obtain instance-wise segmentation with class labels.

        Args:
            pred_map (Tensor): [H, W, 4] - np_mask, h_map, v_map, class_map

        Returns:
            Tensor: [H, W] with instance-wise majority-vote class labels.
        """
        pred_map = pred_map.cpu().numpy()
        np_mask = pred_map[:, :, 0] > 0.5
        h_map = pred_map[:, :, 1]
        v_map = pred_map[:, :, 2]
        class_map = pred_map[:, :, 3].astype(np.int32)

        # --- Energy landscape: gradient magnitude ---
        dx = cv2.Sobel(h_map, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(v_map, cv2.CV_64F, 0, 1, ksize=3)
        energy = np.sqrt(dx ** 2 + dy ** 2)

        energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
        energy = 1.0 - energy  # Nuclei centers become minima

        # --- Marker-based watershed ---
        distance = ndi.distance_transform_edt(np_mask)
        coords = peak_local_max(distance, labels=np_mask, footprint=np.ones((3, 3)), exclude_border=False)
        markers = np.zeros_like(np_mask, dtype=np.int32)
        for idx, (y, x) in enumerate(coords, start=1):
            markers[y, x] = idx

        instance_map = watershed(energy, markers=markers, mask=np_mask)

        # --- Majority vote for class per instance ---
        type_map = np.zeros_like(instance_map, dtype=np.int32)
        for inst_id in range(1, instance_map.max() + 1):
            mask = instance_map == inst_id
            class_ids, counts = np.unique(class_map[mask], return_counts=True)
            if len(class_ids) > 0:
                majority_class = class_ids[np.argmax(counts)]
                type_map[mask] = majority_class

        return torch.from_numpy(type_map).long().to(device)

    # --- Batch processing ---
    batch_size = np_logits.shape[0]
    np_pred = torch.argmax(np_logits, dim=1, keepdim=True)  # [B, 1, H, W]
    nc_pred = torch.argmax(nc_logits, dim=1, keepdim=True)  # [B, 1, H, W]

    results = []
    for i in range(batch_size):
        nuclei_presence = np_pred[i].permute(1, 2, 0)           # [H, W, 1]
        h_map = hv_logits[i, 0:1].permute(1, 2, 0)              # [H, W, 1]
        v_map = hv_logits[i, 1:2].permute(1, 2, 0)              # [H, W, 1]
        class_map = nc_pred[i].permute(1, 2, 0) if nr_types else torch.zeros_like(h_map)

        pred_map = torch.cat([nuclei_presence, h_map, v_map, class_map], dim=-1)  # [H, W, 4]
        instance_seg = process(pred_map)  # [H, W]
        results.append(instance_seg)

    return torch.stack(results)  # [B, H, W]

def get_node_labels_from_coords(centroid_coords, type_maps):
    """
    Args:
        centroid_coords: (N, 3) tensor, (y, x, batch_id)
        type_maps: (B, H, W) tensor of type labels

    Returns:
        labels: (N,) long tensor
    """
    y = centroid_coords[:, 0].clamp(0, type_maps.shape[1] - 1)
    x = centroid_coords[:, 1].clamp(0, type_maps.shape[2] - 1)
    b = centroid_coords[:, 2].clamp(0, type_maps.shape[0] - 1)

    return type_maps[b, y, x]

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
    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphHoverNet(num_classes=Config.NUM_CLASSES)
    model = model.to(device)
    criterion = HoverLoss()
    model.set_stage("pretrain")

    with torch.no_grad():
        images, masks, h_grads, v_grads = next(iter(train_dataloader))
        images = images.to(device)
        masks = masks.to(device).long()
        h_grads = h_grads.to(device)
        v_grads = v_grads.to(device)

        nc_targets = masks
        np_targets = (nc_targets > 0).long()

        np_logits, hv_logits, nc_logits, centroids, gc_logits = model(images)

        if gc_logits is not None and centroids.shape[0] > 0:
            node_labels = get_node_labels_from_coords(centroids, nc_targets)
            valid = node_labels != 0
        
            if valid.any():
                graph_loss_inputs = (gc_logits[valid], node_labels[valid])
            else:
                # No valid labels — skip graph loss
                graph_loss_inputs = (None, None)
        else:
            # No graph predictions — skip graph loss
            graph_loss_inputs = (None, None)
        
        loss = criterion(
            np_logits, np_targets,
            hv_logits, h_grads, v_grads,
            nc_logits, nc_targets,
            *graph_loss_inputs
        )
        
        # # Evaluate
        nc_pred = torch.argmax(nc_logits, dim=1)
        # iou = compute_multiclass_iou(pred_inst, nc_targets, num_classes=6)
        # pq = compute_multiclass_pq(pred_inst, nc_targets, num_classes=6)

        # Postprocess
        postprocessed_pred = postprocess_hovernet_output(np_logits, hv_logits, nc_logits, device)

        tp, fp, fn, tn = smp.metrics.get_stats(
            postprocessed_pred, nc_targets,
            mode='multiclass',
            num_classes=Config.NUM_CLASSES
        )

        iou_per_class = smp.metrics.iou_score(tp, fp, fn, tn, reduction='none').mean(dim=0)[1:]
        f1_per_class = smp.metrics.f1_score(tp, fp, fn, tn, reduction='none').mean(dim=0)[1:]
        pq_per_class = 2 * iou_per_class * f1_per_class / (iou_per_class + f1_per_class + 1e-8)
        
        print("Loss:", loss)
        print("AJI:", iou_per_class.mean())
        print("PQ:", pq_per_class.mean())
        print("Average F1:", f1_per_class.mean())
        print("F1 Per Class:", f1_per_class)

        visualize_hovernet_output(postprocessed_pred.cpu().numpy()[0])