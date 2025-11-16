"""
Helper script to upload data to Modal volumes.
This can be run locally to prepare your data for training.

Note: For large datasets, consider using Modal's volume API directly or
mounting directories during training runs.
"""

import modal
from pathlib import Path

app = modal.App("paddleocr-data-upload")

data_volume = modal.Volume.from_name("paddleocr-vl-data", create_if_missing=True)


@app.function(
    volumes={"/data": data_volume},
    timeout=3600,
)
def upload_files(files: dict[str, bytes], remote_path: str = "/data"):
    """
    Upload files to the volume.

    Args:
        files: Dictionary mapping file paths to file contents
        remote_path: Remote path in the volume to copy to
    """
    from pathlib import Path

    remote_path = Path(remote_path)
    remote_path.mkdir(parents=True, exist_ok=True)

    print(f"Uploading {len(files)} file(s) to {remote_path}")

    for rel_path, content in files.items():
        dest_path = remote_path / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(content)
        print(f"  Uploaded: {rel_path}")

    data_volume.commit()
    print(f"Upload complete! Data available at {remote_path}")


@app.local_entrypoint()
def main(local_path: str, remote_path: str = "/data"):
    """
    Upload local data to Modal volume.

    Usage:
        modal run upload_data.py --local-path ./my_training_data --remote-path /data

    Note: This reads local files and uploads them to the volume.
    For very large datasets (>1GB), consider using Modal volumes CLI.
    """
    local_path = Path(local_path).expanduser().resolve()
    if not local_path.exists():
        raise ValueError(f"Local path does not exist: {local_path}")

    # Read all files into memory
    files = {}
    if local_path.is_file():
        files[local_path.name] = local_path.read_bytes()
    else:
        for item in local_path.rglob("*"):
            if item.is_file():
                rel_path = str(item.relative_to(local_path))
                files[rel_path] = item.read_bytes()

    print(f"Uploading {len(files)} file(s) from {local_path}")
    upload_files.remote(files=files, remote_path=remote_path)

