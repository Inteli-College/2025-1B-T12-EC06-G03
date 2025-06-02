import urllib.request
import os

def download_image_from_url(url: str, save_as: str):
    os.makedirs(os.path.dirname(save_as), exist_ok=True)
    urllib.request.urlretrieve(url, save_as)
