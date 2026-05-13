"""
Download BreastDCEDL data from Zenodo record 18114231.

Usage:
    from src.data.zenodo import download_zenodo
    download_zenodo(output_dir="/content/data", subsets=["demo"])
    download_zenodo(output_dir="/content/data", subsets=["ispy2", "duke", "metadata"])
"""

import os
import json
import tarfile
import zipfile
import urllib.request
import hashlib

ZENODO_API = "https://zenodo.org/api/records/18114231"

FILES = {
    "ispy2":    {"key": "BreastDCEDL_ISPY2_min_crop.tar.gz",    "md5": "2f1a00310bd00d344664319fb7bd679c"},
    "duke":     {"key": "BreastDCEDL_DUKE_min_crop.tar.gz",      "md5": "16e6c752b9679138e88c188d7a01895f"},
    "ispy1":    {"key": "BreastDCEDL_ISPY1_min_crop.tar.gz",     "md5": "d2d2630940a861819e6a385877f45ba5"},
    "demo":     {"key": "BreastDCEDL_demo_data.tar.gz",          "md5": "5b7d8701309c2c3807f8d126f7ddcede"},
    "models":   {"key": "BreastDCEDL_models.tar.gz",             "md5": "8096eea314a326abd284b7d95837c0d5"},
    "metadata": {"key": "BreastDCEDL_metadata_min_crop.csv",     "md5": "d58b84d0f343b4b8338cbb222e6dda08"},
    "spy1_zip": {"key": "BreastDCEDL_spy1.zip",                  "md5": "e6509c6efa5b0e3370528e8a12ed6b23"},
}


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: str, expected_md5: str | None = None):
    if os.path.isfile(dest):
        if expected_md5 and _md5(dest) == expected_md5:
            print(f"  Already downloaded: {os.path.basename(dest)}")
            return
        print(f"  Re-downloading (checksum mismatch): {os.path.basename(dest)}")

    print(f"  Downloading {os.path.basename(dest)}...")
    urllib.request.urlretrieve(url, dest)

    if expected_md5:
        actual = _md5(dest)
        if actual != expected_md5:
            print(f"  WARNING: MD5 mismatch for {os.path.basename(dest)}")
            print(f"    Expected: {expected_md5}")
            print(f"    Got:      {actual}")


def _extract(path: str, output_dir: str):
    if path.endswith(".tar.gz"):
        print(f"  Extracting {os.path.basename(path)}...")
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(output_dir)
    elif path.endswith(".zip"):
        print(f"  Extracting {os.path.basename(path)}...")
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(output_dir)


def download_zenodo(
    output_dir: str,
    subsets: list[str] | None = None,
    extract: bool = True,
):
    """
    Download files from Zenodo record 18114231.

    Args:
        output_dir: Where to save downloaded files.
        subsets: Which subsets to download. Options: ispy2, duke, ispy1, demo,
                 models, metadata, spy1_zip. If None, downloads metadata + demo.
        extract: Whether to extract tar.gz/zip archives after download.
    """
    os.makedirs(output_dir, exist_ok=True)

    if subsets is None:
        subsets = ["metadata", "demo"]

    base_url = f"{ZENODO_API}/files"

    for subset in subsets:
        if subset not in FILES:
            print(f"  Unknown subset: {subset}. Options: {list(FILES.keys())}")
            continue

        info = FILES[subset]
        url = f"{base_url}/{info['key']}/content"
        dest = os.path.join(output_dir, info["key"])

        _download_file(url, dest, info["md5"])

        if extract and (dest.endswith(".tar.gz") or dest.endswith(".zip")):
            _extract(dest, output_dir)

    print(f"Download complete. Contents of {output_dir}:")
    for item in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, item))
        if size > 1e9:
            print(f"  {item}  ({size/1e9:.1f} GB)")
        elif size > 1e6:
            print(f"  {item}  ({size/1e6:.0f} MB)")
        else:
            print(f"  {item}  ({size/1e3:.0f} KB)")
