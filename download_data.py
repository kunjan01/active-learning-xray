"""
download_data.py
================
Download the Chest X-Ray Pneumonia dataset from Kaggle.

Usage:
    Option 1 (Kaggle API):
        1. pip install kaggle
        2. Place your kaggle.json in ~/.kaggle/
        3. python download_data.py

    Option 2 (Manual):
        1. Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
        2. Download and extract into data/chest_xray/
        3. Verify structure: data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/
"""

import os
import zipfile
import shutil


def download_with_kaggle_api(dest_dir: str = "data") -> None:
    """Download dataset using the Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        print("Then place your kaggle.json API token in ~/.kaggle/")
        return

    api = KaggleApi()
    api.authenticate()

    dataset = "paultimothymooney/chest-xray-pneumonia"
    print(f"Downloading {dataset} ...")
    api.dataset_download_files(dataset, path=dest_dir, unzip=False)

    # The download produces a zip file
    zip_path = os.path.join(dest_dir, "chest-xray-pneumonia.zip")
    if os.path.exists(zip_path):
        print("Extracting ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        os.remove(zip_path)
        print(f"Done. Dataset extracted to {dest_dir}/chest_xray/")
    else:
        print(f"Warning: Expected zip file not found at {zip_path}")
        print("Check if the dataset was downloaded with a different name.")


def verify_structure(data_root: str = "data/chest_xray") -> bool:
    """Verify the dataset has the expected directory structure."""
    expected_dirs = [
        "train/NORMAL",
        "train/PNEUMONIA",
        "val/NORMAL",
        "val/PNEUMONIA",
        "test/NORMAL",
        "test/PNEUMONIA",
    ]

    all_ok = True
    for d in expected_dirs:
        full_path = os.path.join(data_root, d)
        if os.path.isdir(full_path):
            n_files = len([f for f in os.listdir(full_path)
                          if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
            print(f"  [OK] {d}: {n_files} images")
        else:
            print(f"  [MISSING] {d}: MISSING")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    data_root = os.path.join("data", "chest_xray")

    if os.path.isdir(data_root):
        print(f"Dataset directory already exists at {data_root}")
        print("Verifying structure ...")
        if verify_structure(data_root):
            print("\nDataset is ready!")
        else:
            print("\nDataset structure is incomplete. Re-download or fix manually.")
    else:
        print("Dataset not found. Attempting download via Kaggle API ...")
        download_with_kaggle_api()
        print("\nVerifying ...")
        if verify_structure(data_root):
            print("\nDataset is ready!")
        else:
            print("\nDownload may have failed. Try manual download:")
            print("  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
            print(f"  Extract to: {data_root}/")
