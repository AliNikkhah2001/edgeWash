"""
Dataset download module for handwashing detection training pipeline.

Downloads PSKUS, METC, and Kaggle WHO6 datasets from public repositories.
Handles extraction and organization of downloaded files.
"""

import sys
import logging
import zipfile
import tarfile
import requests
import subprocess
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Import configuration
from config import (
    DATASETS,
    RAW_DIR,
    RANDOM_SEED,
    LOG_FORMAT,
    LOG_DATE_FORMAT
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT
)
logger = logging.getLogger(__name__)


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download file from URL with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save downloaded file
        chunk_size: Download chunk size in bytes
    
    Returns:
        True if download successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Downloaded: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def extract_zip(zip_path: Path, extract_to: Path, remove_after: bool = True) -> bool:
    """
    Extract ZIP archive.
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract to
        remove_after: Remove ZIP after extraction
    
    Returns:
        True if extraction successful, False otherwise
    """
    try:
        logger.info(f"Extracting: {zip_path.name}")
        extract_to.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        logger.info(f"Extracted to: {extract_to}")
        
        if remove_after:
            zip_path.unlink()
            logger.info(f"Removed: {zip_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def extract_tar(tar_path: Path, extract_to: Path, remove_after: bool = True) -> bool:
    """
    Extract TAR archive.
    
    Args:
        tar_path: Path to TAR file
        extract_to: Directory to extract to
        remove_after: Remove TAR after extraction
    
    Returns:
        True if extraction successful, False otherwise
    """
    try:
        logger.info(f"Extracting: {tar_path.name}")
        extract_to.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(tar_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
        
        logger.info(f"Extracted to: {extract_to}")
        
        if remove_after:
            tar_path.unlink()
            logger.info(f"Removed: {tar_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def download_zenodo_dataset(zenodo_id: str, output_dir: Path) -> bool:
    """
    Download dataset from Zenodo using zenodo-get.
    
    Note: Requires zenodo-get to be installed: pip install zenodo-get
    
    Args:
        zenodo_id: Zenodo record ID
        output_dir: Directory to save dataset
    
    Returns:
        True if download successful, False otherwise
    """
    try:
        logger.info(f"Downloading Zenodo record {zenodo_id}...")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = subprocess.run(
            ['zenodo_get', '-r', zenodo_id, '-o', str(output_dir)],
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"Downloaded to: {output_dir}")
        return True
        
    except FileNotFoundError:
        logger.error("zenodo_get not found. Install with: pip install zenodo-get")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Zenodo download failed (returncode={e.returncode})")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Zenodo download failed: {e}")
        return False


def download_kaggle_dataset() -> bool:
    """
    Download Kaggle WHO6 dataset from GitHub mirror.
    
    Returns:
        True if download and extraction successful, False otherwise
    """
    kaggle_info = DATASETS['kaggle']
    kaggle_dir = RAW_DIR / 'kaggle'
    kaggle_tar = kaggle_dir / 'kaggle-dataset-6classes.tar'
    kaggle_extracted = kaggle_dir / 'kaggle-dataset-6classes'
    
    # Check if already downloaded
    if kaggle_extracted.exists() and any(kaggle_extracted.iterdir()):
        logger.info(f"Kaggle dataset already exists: {kaggle_extracted}")
        return True
    
    logger.info(f"Downloading {kaggle_info['name']} ({kaggle_info['size_gb']:.2f} GB)...")
    
    # Download
    if not download_file(kaggle_info['url'], kaggle_tar):
        return False
    
    # Extract
    if not extract_tar(kaggle_tar, kaggle_dir, remove_after=True):
        return False
    
    logger.info(f"Kaggle dataset ready: {kaggle_extracted}")
    return True


def download_pskus_dataset() -> bool:
    """
    Download PSKUS Hospital dataset from Zenodo.
    
    Note: Large download (~18 GB). Requires zenodo-get.
    
    Returns:
        True if download successful, False otherwise
    """
    pskus_info = DATASETS['pskus']
    pskus_dir = RAW_DIR / 'pskus'
    
    # Check if already downloaded
    summary_csv = pskus_dir / 'summary.csv'
    if summary_csv.exists():
        logger.info(f"PSKUS dataset already exists: {pskus_dir}")
        return True
    
    logger.warning(
        f"Downloading {pskus_info['name']} ({pskus_info['size_gb']:.1f} GB). "
        f"This may take 30-60 minutes depending on connection speed. Progress below shows zenodo_get output."
    )
    
    # Download using zenodo_get
    if not download_zenodo_dataset(pskus_info['zenodo_id'], pskus_dir):
        return False
    
    # Extract all ZIP files
    logger.info("Extracting PSKUS dataset files...")
    for zip_file in pskus_dir.glob('*.zip'):
        if not extract_zip(zip_file, pskus_dir, remove_after=True):
            logger.warning(f"Failed to extract: {zip_file}")
    
    logger.info(f"PSKUS dataset ready: {pskus_dir}")
    return True


def download_metc_dataset() -> bool:
    """
    Download METC Lab dataset from Zenodo.
    
    Note: Requires zenodo-get.
    
    Returns:
        True if download successful, False otherwise
    """
    metc_info = DATASETS['metc']
    metc_dir = RAW_DIR / 'metc'
    
    # Check if already downloaded
    summary_csv = metc_dir / 'summary.csv'
    if summary_csv.exists():
        logger.info(f"METC dataset already exists: {metc_dir}")
        return True
    
    logger.info(f"Downloading {metc_info['name']} ({metc_info['size_gb']:.2f} GB)...")
    
    # Download using zenodo_get
    if not download_zenodo_dataset(metc_info['zenodo_id'], metc_dir):
        return False
    
    # Extract all ZIP files
    logger.info("Extracting METC dataset files...")
    for zip_file in metc_dir.glob('*.zip'):
        if not extract_zip(zip_file, metc_dir, remove_after=True):
            logger.warning(f"Failed to extract: {zip_file}")
    
    logger.info(f"METC dataset ready: {metc_dir}")
    return True


def download_synthetic_blender_rozakar() -> bool:
    """
    Download synthetic Blender dataset (multiple Google Drive parts).
    """
    synth_info = DATASETS['synthetic_blender_rozakar']
    synth_dir = RAW_DIR / 'synthetic_blender_rozakar'
    synth_dir.mkdir(parents=True, exist_ok=True)
    marker = synth_dir / '.complete'
    if marker.exists():
        logger.info(f"Synthetic dataset already present: {synth_dir}")
        return True
    links = synth_info.get('gdrive_links', [])
    if not links:
        logger.error("No Google Drive links configured for synthetic_blender_rozakar")
        return False
    logger.info(f"Downloading {synth_info['name']} (~{synth_info.get('size_gb','?')} GB) in {len(links)} parts...")
    success = True
    for idx, link in enumerate(links, start=1):
        out_file = synth_dir / f"part{idx}.bin"
        logger.info(f"Part {idx}/{len(links)} -> {out_file.name}")
        if out_file.exists():
            logger.info(f"  Already exists: {out_file}")
            continue
        if not download_file(link, out_file, chunk_size=1024*1024):
            logger.error(f"Failed part {idx}: {link}")
            success = False
            break
    if success:
        marker.touch()
        logger.info(f"Synthetic dataset parts downloaded to: {synth_dir}")
    return success


def verify_datasets() -> dict:
    """
    Verify downloaded datasets.
    
    Returns:
        Dictionary with dataset verification status
    """
    status = {}
    
    for dataset_name, dataset_info in DATASETS.items():
        dataset_dir = RAW_DIR / dataset_name
        
        if dataset_name == 'kaggle':
            # Check for kaggle-dataset-6classes folder
            kaggle_extracted = dataset_dir / 'kaggle-dataset-6classes'
            exists = kaggle_extracted.exists() and any(kaggle_extracted.iterdir())
            num_files = len(list(kaggle_extracted.rglob('*'))) if exists else 0
        else:
            # Check for summary.csv (PSKUS, METC)
            summary_csv = dataset_dir / 'summary.csv'
            exists = summary_csv.exists()
            num_files = len(list(dataset_dir.rglob('*'))) if exists else 0
        
        status[dataset_name] = {
            'name': dataset_info['name'],
            'exists': exists,
            'num_files': num_files,
            'path': str(dataset_dir)
        }
    
    return status


def download_all_datasets(skip_large: bool = False) -> None:
    """
    Download all datasets.
    
    Args:
        skip_large: Skip large datasets (PSKUS, METC) if True
    """
    logger.info("=" * 80)
    logger.info("DATASET DOWNLOAD")
    logger.info("=" * 80)
    
    # Download Kaggle (small, always download)
    logger.info("\n[1/3] Downloading Kaggle WHO6 dataset...")
    if download_kaggle_dataset():
        logger.info("✓ Kaggle dataset ready")
    else:
        logger.error("✗ Kaggle dataset download failed")
    
    if skip_large:
        logger.warning("\nSkipping large datasets (PSKUS, METC). Use --download-all to include them.")
    else:
        # Download PSKUS (large)
        logger.info("\n[2/3] Downloading PSKUS Hospital dataset...")
        if download_pskus_dataset():
            logger.info("✓ PSKUS dataset ready")
        else:
            logger.error("✗ PSKUS dataset download failed")
        
        # Download METC (medium)
        logger.info("\n[3/3] Downloading METC Lab dataset...")
        if download_metc_dataset():
            logger.info("✓ METC dataset ready")
        else:
            logger.error("✗ METC dataset download failed")

    # Synthetic dataset (optional)
    logger.info("\n[4/4] Downloading Synthetic Blender Rozakar dataset (optional)...")
    if download_synthetic_blender_rozakar():
        logger.info("✓ Synthetic dataset ready")
    else:
        logger.warning("✗ Synthetic dataset download failed or skipped")
    
    # Verify all datasets
    logger.info("\n" + "=" * 80)
    logger.info("DATASET VERIFICATION")
    logger.info("=" * 80)
    
    status = verify_datasets()
    for dataset_name, info in status.items():
        status_icon = "✓" if info['exists'] else "✗"
        logger.info(f"{status_icon} {info['name']}: {info['num_files']} files in {info['path']}")
    
    logger.info("\nDownload complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download handwashing datasets for training pipeline"
    )
    parser.add_argument(
        '--download-all',
        action='store_true',
        help='Download all datasets including large ones (PSKUS, METC)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing datasets without downloading'
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        logger.info("Verifying datasets...")
        status = verify_datasets()
        for dataset_name, info in status.items():
            status_icon = "✓" if info['exists'] else "✗"
            logger.info(f"{status_icon} {info['name']}: {info['num_files']} files")
    else:
        download_all_datasets(skip_large=not args.download_all)
