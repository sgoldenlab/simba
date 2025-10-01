import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

BASE_URL = "https://data.caltech.edu"
RECORD_URL = "https://data.caltech.edu/records/4emt5-b0t10"
OUT_DIR = r"E:\crim13"  # target directory


def get_file_links(record_url):
    """Scrape CaltechDATA record page to extract file download links."""
    resp = requests.get(record_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    file_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/records/4emt5-b0t10/files/" in href:  # only dataset files
            file_links.append(urljoin(BASE_URL, href))
    return sorted(set(file_links))


def safe_filename_from_url(url):
    """Extract a clean filename without query parameters."""
    path = urlparse(url).path
    return os.path.basename(path)


def download_file(url, out_dir=OUT_DIR, chunk_size=8192):
    """Download one file with streaming to avoid memory issues."""
    os.makedirs(out_dir, exist_ok=True)
    filename = safe_filename_from_url(url)
    dest_path = os.path.join(out_dir, filename)

    if os.path.exists(dest_path):
        print(f"Skipping {filename}, already exists.")
        return

    print(f"Downloading {filename} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    print(f"Finished {filename}")


def main():
    links = get_file_links(RECORD_URL)
    print(f"Found {len(links)} files to download")
    for link in links:
        download_file(link)


if __name__ == "__main__":
    main()
