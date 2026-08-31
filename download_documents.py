import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv


# Load configuration from the local .env file
load_dotenv()

CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER")

# Documents will be downloaded into this local directory
DOWNLOAD_DIRECTORY = Path("documents")


def validate_configuration():
    """Ensure that all required settings are available."""

    if not CONNECTION_STRING:
        raise ValueError(
            "AZURE_STORAGE_CONNECTION_STRING is missing from .env"
        )

    if not CONTAINER_NAME:
        raise ValueError(
            "AZURE_STORAGE_CONTAINER is missing from .env"
        )


def download_documents():
    """Download all files from the configured Blob container."""

    validate_configuration()

    # Create the local documents directory if it does not exist
    DOWNLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)

    blob_service_client = BlobServiceClient.from_connection_string(
        CONNECTION_STRING
    )

    container_client = blob_service_client.get_container_client(
        CONTAINER_NAME
    )

    print(f"Connecting to Blob container: {CONTAINER_NAME}")
    print("Checking available files...")

    blobs = list(container_client.list_blobs())

    if not blobs:
        print("No files were found in the Blob container.")
        return

    print(f"Found {len(blobs)} file(s).")

    downloaded_count = 0
    skipped_count = 0

    for blob in blobs:
        blob_name = blob.name

        # Ignore virtual folder entries
        if blob_name.endswith("/"):
            continue

        # Preserve Blob folder structure locally
        local_file_path = DOWNLOAD_DIRECTORY / blob_name
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        blob_client = container_client.get_blob_client(blob_name)

        # Skip an unchanged local file based on file size
        if (
            local_file_path.exists()
            and local_file_path.stat().st_size == blob.size
        ):
            print(f"Skipped unchanged file: {blob_name}")
            skipped_count += 1
            continue

        with open(local_file_path, "wb") as local_file:
            download_stream = blob_client.download_blob()
            download_stream.readinto(local_file)

        print(f"Downloaded: {blob_name}")
        downloaded_count += 1

    print()
    print("Download completed.")
    print(f"Downloaded files: {downloaded_count}")
    print(f"Skipped unchanged files: {skipped_count}")
    print(f"Local directory: {DOWNLOAD_DIRECTORY.resolve()}")


if __name__ == "__main__":
    try:
        download_documents()
    except Exception as error:
        print()
        print("Document download failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Error details: {error}")
        raise SystemExit(1)