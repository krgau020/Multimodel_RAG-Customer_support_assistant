import os
import requests
import json
from urllib.parse import urlparse

# Paths
DATA_FILE = r"text-data_json/products_final.json"    # your JSON file
IMAGE_FOLDER = r"C:\Users\admin\Desktop\multimodel-rag-Customer_support\Dataset\images"        # folder to save images
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Load JSON data
with open(DATA_FILE, "r") as f:
    data = json.load(f)

def get_file_extension(url):
    """Get file extension from URL, default to .jpg if missing."""
    path = urlparse(url).path
    ext = os.path.splitext(path)[1]
    return ext if ext else ".jpg"

def download_image(url, filename):
    """Download an image from URL and save to filename."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filename, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {filename}")
            return True
        else:
            print(f"Failed to download: {url}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

# Loop through dataset
for item in data:
    asin = item.get("asin")
    if not asin:
        continue

    # === Main image ===
    main_url = item.get("image_url")
    if main_url:
        ext = get_file_extension(main_url)
        main_filename = f"{asin}_main{ext}"
        main_filepath = os.path.join(IMAGE_FOLDER, main_filename)
        if download_image(main_url, main_filepath):
            item["image_path"] = main_filepath  # Update JSON with local path

    # === Related images ===
    related_images = item.get("support_data", {}).get("related_images", [])
    local_related = []
    for idx, url in enumerate(related_images, start=1):
        ext = get_file_extension(url)
        rel_filename = f"{asin}_rel{idx}{ext}"
        rel_filepath = os.path.join(IMAGE_FOLDER, rel_filename)
        if download_image(url, rel_filepath):
            local_related.append(rel_filepath)
    if local_related:
        item["support_data"]["local_related_images"] = local_related

# Save updated JSON
with open("products_local.json", "w") as f:
    json.dump(data, f, indent=2)

print("All images downloaded and JSON updated successfully!")
