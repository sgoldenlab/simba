import requests

def get_package_sizes(package_name):
    # Fetch package information from PyPI
    url = f'https://pypi.org/pypi/{package_name}/json'
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching package data: {response.status_code}")
        return

    package_data = response.json()
    releases = package_data['releases']

    # Iterate through all releases and print the size of each distribution file
    results = {}

    for version, release_info in releases.items():
        for file_info in release_info:
            size_kb = file_info['size'] / 1024  # Convert bytes to kilobytes
            results[version] = size_kb
            #print(f"Version: {version}, Filename: {file_info['filename']}, Size: {size_kb:.2f} KB")
    return results
# Example usage
package_name = 'simba-uw-tf-dev'  # Replace with the package name you want to check
r = get_package_sizes(package_name)

r = {k: v for k, v in sorted(r.items(), key=lambda item: item[1])}