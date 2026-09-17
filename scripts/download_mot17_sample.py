"""
Phase 1 — MOT17-04-SDP Feasibility Download Script
Downloads only the MOT17-04-SDP training sequence (gt, seqinfo, first 10 frames)
from HuggingFace to data/mot17/ for feasibility testing.
"""
import os
from huggingface_hub import hf_hub_download, list_repo_tree

REPO_ID = "Morrison1025/MOT17"
SEQUENCE = "train/MOT17-04-SDP"
LOCAL_BASE = os.path.join("data", "mot17")

def download_file(repo_path, local_dir):
    """Download a single file from the HuggingFace repo."""
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=repo_path,
            repo_type="dataset",
            local_dir=local_dir,
        )
        print(f"  [OK] {repo_path}")
        return path
    except Exception as e:
        print(f"  [FAIL] {repo_path}: {e}")
        return None

def main():
    print(f"Downloading MOT17-04-SDP sequence to {LOCAL_BASE}/")
    print()

    # Download seqinfo.ini
    download_file(f"{SEQUENCE}/seqinfo.ini", LOCAL_BASE)

    # Download ground truth
    download_file(f"{SEQUENCE}/gt/gt.txt", LOCAL_BASE)

    # Download detection file
    download_file(f"{SEQUENCE}/det/det.txt", LOCAL_BASE)

    # Download first 10 frames only (for feasibility inspection)
    print("\nDownloading first 10 frames...")
    for i in range(1, 11):
        fname = f"{SEQUENCE}/img1/{i:06d}.jpg"
        download_file(fname, LOCAL_BASE)

    print("\n=== MOT17-04-SDP feasibility download complete ===")

if __name__ == "__main__":
    main()
