# download_convert_kaggle.py

import os
import numpy as np
import scipy.io
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_convert(
    dataset_slug: str = 'abhijeetgo/indian-pines-hyperspectral-dataset',
    out_dir: str     = 'data/IndianPines'
):
    # 1) Authenticate
    api = KaggleApi()
    api.authenticate()

    # 2) Download & unzip
    os.makedirs(out_dir, exist_ok=True)
    print(f"🔽 Downloading `{dataset_slug}` to `{out_dir}` …")
    api.dataset_download_files(dataset_slug, path=out_dir, unzip=True)
    print("✅ Download complete.")

    # 3) Find all .npy files
    npy_files = [f for f in os.listdir(out_dir) if f.endswith('.npy')]
    if len(npy_files) < 2:
        raise RuntimeError(f"Expected ≥2 .npy files in `{out_dir}`, found: {npy_files}")
    # Heuristic: data array is the larger file, ground truth the smaller
    sizes = {f: os.path.getsize(os.path.join(out_dir, f)) for f in npy_files}
    data_file  = max(sizes, key=sizes.get)
    label_file = min(sizes, key=sizes.get)

    data_path  = os.path.join(out_dir, data_file)
    label_path = os.path.join(out_dir, label_file)
    print(f"📂 Data:   {data_file}\n📂 Labels: {label_file}")

    # 4) Load and save as .mat
    print("🔄 Converting to .mat format …")
    data  = np.load(data_path)
    labels = np.load(label_path)

    scipy.io.savemat(os.path.join(out_dir, 'indian_pines_data.mat'),
                     {'data': data})
    scipy.io.savemat(os.path.join(out_dir, 'indian_pines_labels.mat'),
                     {'labels': labels})

    print("✅ Conversion complete.")
    print(f"→ `{out_dir}/indian_pines_data.mat`")
    print(f"→ `{out_dir}/indian_pines_labels.mat`")

if __name__ == '__main__':
    download_and_convert()
