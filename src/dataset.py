import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Root dataset directory, e.g., "MoNuSAC"
            split (str): One of 'train', 'val', 'test'
            image_transform (callable, optional): Transform applied only to image
        """
        self.split_dir = os.path.join(root_dir, split)
        self.sample_dirs = sorted([
            os.path.join(self.split_dir, d)
            for d in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, d))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        image = np.load(os.path.join(sample_dir, "image.npy"))  # HWC
        mask = np.load(os.path.join(sample_dir, "mask.npy"))    # HW
        h_map = np.load(os.path.join(sample_dir, "h_map.npy"))  # HW
        v_map = np.load(os.path.join(sample_dir, "v_map.npy"))  # HW

        h_map = (h_map.astype(np.float32) / 127.5) - 1.0
        v_map = (v_map.astype(np.float32) / 127.5) - 1.0

        if self.transform:
            image = self.transform(image)

        return image, \
                torch.from_numpy(mask).long(), \
                torch.from_numpy(h_map).float(), \
                torch.from_numpy(v_map).float()
    
if __name__ == "__main__":
    from config import Config
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts HWC to CHW and scales to [0, 1]
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # example: scale to [-1, 1]
    ])

    dataset_path = os.path.join(Config.DATA_PATH, Config.DATASET)

    train_dataset = SegmentationDataset(dataset_path, split="train", transform=transform)
    val_dataset = SegmentationDataset(dataset_path, split="val", transform=transform)
    test_dataset = SegmentationDataset(dataset_path, split="test", transform=transform)

    from torch.utils.data import Dataset, DataLoader

    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("First fold:", len(train_dataloader))
    print("Second fold:", len(val_dataloader))
    print("Third fold:", len(test_dataloader))

    image, mask, h_map, v_map = next(iter(train_dataloader))
    print(image.shape)
    print(mask.shape)
    print(h_map.shape)
    print(v_map.shape)