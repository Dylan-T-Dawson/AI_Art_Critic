import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from bs4 import BeautifulSoup
from googlesearch import search
import requests
from io import BytesIO
import base64
import threading

class ImageDownloader:
    def __init__(self, root, query, num_results=5):
        self.root = root
        self.query = query
        self.num_results = num_results
        self.image_urls = []
        self.current_index = 0
        self.retry_count = 0

        self.create_widgets()

    def create_widgets(self):
        self.label = tk.Label(self.root, text="Do you want to save this image?")
        self.label.pack(pady=10)

        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        self.save_button = ttk.Button(self.root, text="Save", command=self.save_image)
        self.save_button.pack(side=tk.LEFT, padx=10)

        self.skip_button = ttk.Button(self.root, text="Skip", command=self.skip_image)
        self.skip_button.pack(side=tk.RIGHT, padx=10)

        self.root.bind('y', lambda event: self.save_image())
        self.root.bind('x', lambda event: self.skip_image())

        self.update_image()

    def update_image(self):
        if self.current_index < len(self.image_urls):
            img_url = self.image_urls[self.current_index]
            threading.Thread(target=self.load_and_display_image, args=(img_url,)).start()
        else:
            # Fetch a new set of results and append to the existing list
            new_results = self.download_image_urls(self.query, self.num_results)
            self.image_urls.extend(new_results)
            self.root.after(0, self.update_image)

    def load_and_display_image(self, img_url):
        image = self.load_image_from_url(img_url, max_retry=3)
        if image is not None:
            self.root.after(0, lambda: self.display_image(image))
        else:
            self.root.after(0, self.next_image)  # Skip to the next image on error

    def display_image(self, image):
        self.image_label.configure(image=image)
        self.image_label.image = image
        self.label.config(text="Do you want to save this image?")

    def load_image_from_url(self, url, max_retry=3):
        try:
            if url.startswith("data:image/"):
                # Handle data URI (e.g., inline SVG)
                return self.load_image_from_data_uri(url)
            else:
                # Fetch image from a regular URL
                response = requests.get(url)
                response.raise_for_status()  # Raise an HTTPError for bad responses
                img = Image.open(BytesIO(response.content))
                img = img.resize((300, 300), Image.ANTIALIAS)
                return ImageTk.PhotoImage(img)
        except requests.exceptions.RequestException:
            self.retry_count += 1
            if self.retry_count == max_retry:
                print(f"Failed to fetch image from {url} after {max_retry} attempts. Skipping...")
            return None
        except Exception:
            return None

    def load_image_from_data_uri(self, data_uri):
        _, data = data_uri.split(",", 1)
        image_data = BytesIO(base64.b64decode(data))
        img = Image.open(image_data)
        img = img.resize((300, 300), Image.ANTIALIAS)
        return ImageTk.PhotoImage(img)

    def save_image(self, event=None):
        if 0 <= self.current_index < len(self.image_urls):
            img_url = self.image_urls[self.current_index]
            response = requests.get(img_url)
            with open(f"saved_image_{self.current_index}.png", "wb") as file:
                file.write(response.content)

    def skip_image(self, event=None):
        self.root.after(0, self.next_image)  # Skip to the next image

    def next_image(self):
        self.current_index += 1
        self.update_image()

    def start(self):
        self.image_urls = self.download_image_urls(self.query, self.num_results)
        self.skip_image()  # Automatically skip the first image
        self.root.mainloop()

    def download_image_urls(self, query, num_results):
        search_results = list(search(query, num_results=num_results))
        image_urls = []

        for result in search_results:
            try:
                html_page = requests.get(result)
            except Exception:
                continue
            soup = BeautifulSoup(html_page.text, 'html.parser')

            # Extract image URLs from the page
            for img_tag in soup.find_all('img'):
                img_url = img_tag.get('src')

                # Skip URLs that do not have a valid scheme or do not represent normal images
                if img_url and img_url.startswith(('http://', 'https://')):
                    image_urls.append(img_url)

        return image_urls

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Image Downloader")
    #Do not set num_results (which is the number of google pages) too high or Google will put you in API jail for making too many requests.
    downloader = ImageDownloader(root, query="Black and White artwork", num_results=25)
    downloader.start()