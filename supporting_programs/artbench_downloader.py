import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import random

def download_image(name, label, url):
    # Check if the image should be downloaded based on a 1/20 chance
    if random.randint(1, 20) != 1:
        return

    folder_path = f"./{label}"

    # Check if the folder already exists
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Try to download the image up to two times
    for attempt in range(2):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            image_path = os.path.join(folder_path, name)
            with open(image_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {name} to {label}")
            break  # Break the loop if the download is successful
        except requests.exceptions.RequestException as e:
            if attempt == 1:
                print(f"Failed to download: {name}. Error: {e}. No more attempts.")
            else:
                print(f"Failed to download: {name}. Error: {e}. Retrying...")

if __name__ == "__main__":
    # Read data from Excel file
    excel_file = "ArtBench-10.xlsx"
    df = pd.read_excel(excel_file)

    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor() as executor:
        executor.map(download_image, df['name'], df['label'], df['url'])

    print("Download process completed.")