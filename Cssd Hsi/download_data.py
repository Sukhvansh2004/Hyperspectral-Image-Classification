import os
import requests
from pathlib import Path

def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {dest_path}")

def download_indian_pines(root_dir='data/IndianPines'):
    os.makedirs(root_dir, exist_ok=True)
    # URLs for MATLAB files
    urls = {
        'data.mat': 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
        'labels.mat': 'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'
    }
    for name, url in urls.items():
        dest = os.path.join(root_dir, name)
        if not os.path.exists(dest):
            download_file(url, dest)
        else:
            print(f"{dest} already exists, skipping.")

if __name__ == '__main__':
    download_indian_pines()
