import os
import SimpleITK as sitk
import numpy as np

def load_nifti(path):
    """
    Load a NIfTI file using SimpleITK and return image and metadata.
    """
    image = sitk.ReadImage(path)
    # shape: [z, y, x]
    array = sitk.GetArrayFromImage(image) 
    # (x, y, z)
    spacing = image.GetSpacing()         
    origin = image.GetOrigin()
    direction = image.GetDirection()
    return array, spacing, origin, direction

def save_nifti(array, reference_image, output_path):
    """
    Save a numpy array as NIfTI using a reference SimpleITK image.
    """
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(reference_image.GetSpacing())
    image.SetOrigin(reference_image.GetOrigin())
    image.SetDirection(reference_image.GetDirection())
    sitk.WriteImage(image, output_path)

def normalize_ct(image):
    """Clip and normalize CT image to [-1000, 1000] range."""
    image = np.clip(image, -1000, 1000)
    # Normalize to [0, 1]
    return (image + 1000) / 2000  

def normalize_pet(image):
    """
    Z-score normalization for PET image.
    """
    mean = np.mean(image)
    std = np.std(image)
    return (image - mean) / (std + 1e-8)

def preprocess_case(case_id, input_dir):
    """Load and preprocess a single case (CT, PET, Mask)."""
    ct_path = os.path.join(input_dir, f"{case_id}__CT.nii.gz")
    pt_path = os.path.join(input_dir, f"{case_id}__PT.nii.gz")
    mask_path = os.path.join(input_dir, f"{case_id}.nii.gz")

    # Load all three
    ct, ct_spacing, _, _ = load_nifti(ct_path)
    pt, pt_spacing, _, _ = load_nifti(pt_path)
    mask, mask_spacing, _, _ = load_nifti(mask_path)

    # Check spacing consistency
    if ct_spacing != pt_spacing or ct_spacing != mask_spacing:
        print(f"[Warning] Spacing mismatch in {case_id}")

    # Normalize
    ct_norm = normalize_ct(ct)
    pt_norm = normalize_pet(pt)

    # Stack into 2-channel input: [2, Z, Y, X]
    input_image = np.stack([ct_norm, pt_norm], axis=0)
    return input_image, mask

# Example usage
if __name__ == "__main__":
    # Example case
    case_id = "CHUS_001"  
    # dataset input path
    input_dir = "replace it with ur correct path" 

    image, mask = preprocess_case(case_id, input_dir)
    # Should be [2, Z, Y, X]
    print("Input shape:", image.shape) 
    print("Mask unique labels:", np.unique(mask))
