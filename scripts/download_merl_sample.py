"""
Phase 1 — MERL Single Video Download
Downloads the MERL Videos ZIP, extracts just one video (01_01_crop.mp4),
then deletes the ZIP to save space.
"""
import urllib.request
import zipfile
import os
import sys

URL = "https://www.merl.com/pub/tmarks/MERL_Shopping_Dataset/Videos_MERL_Shopping_Dataset.zip"
ZIP_PATH = os.path.join("data", "merl", "Videos_MERL_Shopping_Dataset.zip")
EXTRACT_DIR = os.path.join("data", "merl")
TARGET_VIDEO = "01_01_crop.mp4"

def download_with_progress(url, dest):
    """Download with progress reporting."""
    print(f"Downloading: {url}")
    print(f"Destination: {dest}")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    resp = urllib.request.urlopen(url)
    total = int(resp.headers.get("Content-Length", 0))
    print(f"Total size: {total / (1024*1024):.1f} MB")
    
    downloaded = 0
    block_size = 1024 * 1024  # 1 MB
    with open(dest, "wb") as f:
        while True:
            data = resp.read(block_size)
            if not data:
                break
            f.write(data)
            downloaded += len(data)
            pct = (downloaded / total * 100) if total else 0
            print(f"\r  {downloaded/(1024*1024):.1f} / {total/(1024*1024):.1f} MB ({pct:.0f}%)", end="", flush=True)
    print()
    return dest

def main():
    # Download ZIP
    download_with_progress(URL, ZIP_PATH)
    
    # Extract only the target video
    print(f"\nExtracting {TARGET_VIDEO}...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        # List contents to find exact path
        names = zf.namelist()
        target = None
        for name in names:
            if name.endswith(TARGET_VIDEO):
                target = name
                break
        
        if target is None:
            print(f"ERROR: {TARGET_VIDEO} not found in ZIP")
            print(f"First 10 entries: {names[:10]}")
            sys.exit(1)
        
        print(f"  Found: {target}")
        zf.extract(target, EXTRACT_DIR)
        
        # Move to flat location
        extracted_path = os.path.join(EXTRACT_DIR, target)
        final_path = os.path.join(EXTRACT_DIR, TARGET_VIDEO)
        if extracted_path != final_path:
            os.replace(extracted_path, final_path)
        print(f"  Extracted to: {final_path}")
        print(f"  Size: {os.path.getsize(final_path) / (1024*1024):.1f} MB")
    
    # Delete ZIP to save space
    os.remove(ZIP_PATH)
    print(f"\nDeleted ZIP to save space.")
    print("=== MERL single video download complete ===")

if __name__ == "__main__":
    main()
