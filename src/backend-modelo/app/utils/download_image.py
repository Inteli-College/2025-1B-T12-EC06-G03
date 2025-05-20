import urllib.request

def download_image_from_url(url, save_as):
    urllib.request.urlretrieve(url, save_as)