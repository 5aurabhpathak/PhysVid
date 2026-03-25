import argparse
import os
import shutil
import tarfile
import zipfile
from concurrent.futures import as_completed

from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


def extract_tar(tar_path, dest_dir):
    """
    Extracts a .tar file to the specified destination directory.
    """
    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=dest_dir)
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")


def extract_zip(zip_path, dest_dir):
    """
    Extracts a .zip file to the specified destination directory.
    """
    try:
        with zipfile.ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(path=dest_dir)
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract dataset.")
    parser.add_argument("--local_dir", type=str, default="/mnt/localssd/",
                        help="Local directory to save the dataset.")
    parser.add_argument("--repo_id", type=str,
                        default="qihoo360/WISA-80K", help="Hugging Face repository ID.")
    parser.add_argument("--folder_name", type=str, default="data",
                        help="Folder name of the huggingface repo.")
    parser.add_argument('--compressed_format', type=str, choices=['tar', 'zip'], default='tar')
    parser.add_argument('--revision', type=str, default='main')

    args = parser.parse_args()

    file_ext = args.compressed_format
    allow_patterns = [f"{args.folder_name}/*.{file_ext}", f"{args.folder_name}/*.json"]
    func = extract_tar if file_ext == 'tar' else extract_zip

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        revision=args.revision,          # or the branch/tag/commit you want
        allow_patterns=allow_patterns,
        repo_type="dataset"
    )

    # 5. Destination folder for extracted files
    output_dir = os.path.join(args.local_dir, "videos")
    os.makedirs(output_dir, exist_ok=True)

    # 4. Collect all .tar/.zip files recursively from the downloaded folder
    tar_files = []
    for root, dirs, files in os.walk(args.local_dir):
        for file in files:
            if file.endswith("." + file_ext) and not os.path.exists(os.path.join(output_dir, file.split('.')[0])):
                tar_files.append(os.path.join(root, file))
                print('Found unextracted compressed file:', os.path.join(root, file))
            elif file.endswith(".json") and not os.path.exists(os.path.join(output_dir, file)):
                shutil.copy2(os.path.join(root, file), os.path.join(output_dir, file))

    # 6. Extract each tar/zip file in parallel
    futures = []
    with ThreadPoolExecutor() as executor:
        for compressed_file in tar_files:
            futures.append(executor.submit(func, compressed_file, output_dir))

        # track progress with tqdm
        [_ for _ in tqdm(as_completed(futures), total=len(futures), desc="Extracting files", dynamic_ncols=True)]

    print(f"All .{file_ext} files have been downloaded and extracted to:", output_dir)


if __name__ == "__main__":
    main()
