## download the following libraries:
## pip install torch torchvision numpy SimpleITK scikit-image

import os
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from skimage.transform import resize

def load_nifti(path):
    image = sitk.ReadImage(path)
    # [Z, Y, X]
    array = sitk.GetArrayFromImage(image)  
    return array, image

def normalize_ct(image):
    image = np.clip(image, -1000, 1000)
    return (image + 1000) / 2000.0

def normalize_pet(image):
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / (std + 1e-8)

def random_flip(image, mask):
    if random.random() < 0.5:
        image = np.flip(image, axis=2)
        mask = np.flip(mask, axis=2)
    return image, mask

def random_rotation(image, mask):
    if random.random() < 0.5:
        k = random.randint(1, 3)
        image = np.rot90(image, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(1, 2)).copy()
    return image, mask

def center_crop_or_pad(volume, target_shape):
    """
    Crop or pad volume to match target_shape (Z, Y, X).
    """
    output = np.zeros(target_shape, dtype=volume.dtype)
    input_shape = volume.shape
    min_shape = np.minimum(input_shape, target_shape)
    start = [(i - m) // 2 for i, m in zip(input_shape, min_shape)]
    end = [s + m for s, m in zip(start, min_shape)]

    in_crop = tuple(slice(s, e) for s, e in zip(start, end))
    out_crop = tuple(slice((ts - m) // 2, (ts - m) // 2 + m) for ts, m in zip(target_shape, min_shape))

    output[out_crop] = volume[in_crop]
    return output

class HecktorSegmentationDataset(Dataset):
    def __init__(self, case_ids, data_dir, patch_size=(128, 128, 128), augment=True):
        self.case_ids = case_ids
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        ct_path = os.path.join(self.data_dir, f"{case_id}__CT.nii.gz")
        pt_path = os.path.join(self.data_dir, f"{case_id}__PT.nii.gz")
        mask_path = os.path.join(self.data_dir, f"{case_id}.nii.gz")

        ct, _ = load_nifti(ct_path)
        pt, _ = load_nifti(pt_path)
        mask, _ = load_nifti(mask_path)

        ct = normalize_ct(ct)
        pt = normalize_pet(pt)

        # [2, Z, Y, X]
        image = np.stack([ct, pt], axis=0)
        image, mask = self.preprocess(image, mask)

        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.int64),
            'case_id': case_id
        }

    def preprocess(self, image, mask):
        # Center crop or pad to fixed size
        image = np.stack([
            center_crop_or_pad(image[0], self.patch_size),
            center_crop_or_pad(image[1], self.patch_size)
        ])
        mask = center_crop_or_pad(mask, self.patch_size)

        # Augmentation
        if self.augment:
            image, mask = random_flip(image, mask)
            image, mask = random_rotation(image, mask)

        return image, mask
