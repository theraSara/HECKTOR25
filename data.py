from loader import HecktorSegmentationDataset
from torch.utils.data import DataLoader

# Create case list
train_cases = [
  # extract the train cases better than writting it manually
    "CHUM_001", "CHUM_002", "CHUP_001",  
]

dataset = HecktorSegmentationDataset(
    case_ids=train_cases,
    data_dir="/path/to/hecktor2025/training_data",
    patch_size=(128, 128, 128),
    augment=True
)

loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

# Example training loop batch
for batch in loader:
    images = batch['image']       # Shape: [B, 2, Z, Y, X]
    masks = batch['mask']         # Shape: [B, Z, Y, X]
    case_ids = batch['case_id']
    break  # for quick testing
