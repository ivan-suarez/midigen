import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def download_midi_files(url, download_folder="midi_files"):
    # Make a directory to save downloaded files
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    # Get the HTML content from the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all <a> tags on the page and download the ones with .mid or .midi
    for link in soup.find_all('a'):
        href = link.get('href')
        if href and (href.endswith('.mid') or href.endswith('.midi')):
            full_url = urljoin(url, href)
            filename = os.path.join(download_folder, href.split('/')[-1])

            # Download the MIDI file
            try:
                with requests.get(full_url, stream=True) as r:
                    r.raise_for_status()
                    with open(filename, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"Downloaded: {filename}")
            except Exception as e:
                print(f"Failed to download {filename}. Reason: {e}")

# Example Usage
if __name__ == "__main__":
    target_url = 'https://www.khinsider.com/midi/ps2/final-fantasy-x'  # Replace this with the actual URL
    download_midi_files(target_url)
