import subprocess
import zipfile
from os import environ
from pathlib import Path

import synapseclient
import synapseutils
from invoke import task
from invoke.context import Context


@task
def download(c: Context, output_dir: str = "data/raw"):
    syn = synapseclient.Synapse()
    syn.login(email=environ.get("SYNAPSE_EMAIL"), authToken=environ.get("SYNAPSE_AUTH"))
    synapseutils.syncFromSynapse(syn, entity="syn3049714", path=output_dir)
    output_dir = Path(output_dir)
    for dir in output_dir.iterdir():
        if not dir.is_dir():
            continue
        # Replace spaces in directory names with underscores
        new_dir = dir.parent / dir.name.replace(" ", "_")
        dir.rename(new_dir)


# Utility function to unzip a file into a subdirectory named after the zip file
def unzip_file(zip_path: Path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        extract_dir = (
            zip_path.parent / zip_path.stem
        )  # Extract into a subdirectory named after the zip file
        extract_dir.mkdir(
            parents=True, exist_ok=True
        )  # Create the subdirectory if it doesn't exist
        zip_ref.extractall(extract_dir)
        print(f"Extracted: {zip_path} to {extract_dir}")


@task
def unzip_all(c: Context, base_dir: str = "data/raw"):
    """
    Unzips all .zip files in the given base directory and subdirectories.

    Args:
        base_dir (str): The base directory where the data is located. Default is 'data/raw'.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: The directory {base_path} does not exist.")
        return

    # Traverse the directory structure and unzip all .zip files
    for zip_path in base_path.rglob("*.zip"):
        unzip_file(zip_path)

    print("All zip files have been extracted.")


@task
def process(
    c: Context,
    input_dir: str = "data/raw/training_data",
    output_dir: str = "data/processed",
):
    """
    Runs the preprocessing script on the given directory.

    Args:
        script_path (Path): Path to the preprocessing script.
        input_dir (Path): Directory containing unzipped files to process.
        output_dir (Path): Directory where processed data will be stored.
        mode (str): The mode (ss or ts) to pass to the script.
    """
    input_dir = Path(input_dir)
    for dir in input_dir.iterdir():
        if not dir.is_dir():
            continue
        # Construct the command
        for mode in ["ss", "ts"]:
            cmd = ["python3", "scripts/01_preprocess.py", str(dir), output_dir, mode]
            subprocess.run(cmd, check=True)
