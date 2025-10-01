import requests
import os

def download_dataverse_dataset(persistent_id: str, out_dir: str, dataverse_url: str = "https://dataverse.harvard.edu") -> None:
    """
    Download a dataset from Dataverse using its persistentId (DOI or Handle).

    :param persistent_id: The DOI or handle of the dataset (e.g., 'doi:10.7910/DVN/TJCLKP').
    :param out_dir: Directory where the files will be saved.
    :param dataverse_url: Base URL of the Dataverse installation.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Dataset metadata
    meta_url = f"{dataverse_url}/api/datasets/:persistentId/?persistentId={persistent_id}"
    meta_resp = requests.get(meta_url)
    meta_resp.raise_for_status()
    dataset = meta_resp.json()["data"]

    files = dataset["latestVersion"]["files"]

    for f in files:
        file_id = f["dataFile"]["id"]
        file_name = f["dataFile"]["filename"]
        file_url = f"{dataverse_url}/api/access/datafile/{file_id}"

        print(f"Downloading {file_name}...")
        r = requests.get(file_url, stream=True)
        r.raise_for_status()

        out_path = os.path.join(out_dir, file_name)
        with open(out_path, "wb") as f_out:
            for chunk in r.iter_content(chunk_size=8192):
                f_out.write(chunk)

    print("Download complete.")

# Example usage:
download_dataverse_dataset("doi:10.7910/DVN/PHH72E", r"E:\open_video\dataverse")