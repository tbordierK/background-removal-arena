import os
from PIL import Image
import pandas as pd

# Define the directories
original_directories = [
    "../data/resized-original-images"
]
processed_directories = {
    "Photoroom": "../data/processed/photoroom",
    "Clipdrop": "../data/processed/clipdrop",
    "RemoveBG": "../data/processed/removebg",
    "BRIA RMBG 2.0": "../data/processed/bria"
}

def compute_megapixels(width, height):
    return (width * height) / 1_000_000

def check_image_sizes_comparison():
    data = []

    for original_directory in original_directories:
        for filename in os.listdir(original_directory):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                original_path = os.path.join(original_directory, filename)
                with Image.open(original_path) as img:
                    original_size = f"{img.size[0]}x{img.size[1]}"
                    original_megapixels = compute_megapixels(img.size[0], img.size[1])

                sizes = {
                    "original_file_name": filename,
                    "original_size": original_size,
                    "original_megapixels": original_megapixels
                }

                png_filename = os.path.splitext(filename)[0] + ".png"
                for model, directory in processed_directories.items():
                    image_path = os.path.join(directory, png_filename)
                    if os.path.exists(image_path):
                        with Image.open(image_path) as img:
                            image_size = f"{img.size[0]}x{img.size[1]}"
                            image_megapixels = compute_megapixels(img.size[0], img.size[1])
                    else:
                        image_size = "Not found"
                        image_megapixels = "Not found"
                    sizes[model] = image_size
                    sizes[f"{model}_megapixels"] = image_megapixels

                data.append(sizes)

    df = pd.DataFrame(data)
    df.to_csv("image_sizes_comparison.csv", index=False)

if __name__ == "__main__":
    check_image_sizes_comparison()