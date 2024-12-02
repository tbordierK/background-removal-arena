import os
from PIL import Image
import concurrent.futures

# Define the directories
input_directory = "../original-images"
output_directory = "../resized-original-images-test"

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

def resize_image(input_path, output_path):
    with Image.open(input_path) as img:
        # Calculate the current megapixels
        current_megapixels = (img.width * img.height) / 1_000_000
        max_megapixels = 10

        if current_megapixels > max_megapixels:
            # Calculate the scaling factor to reduce the image to 10 megapixels
            scaling_factor = (max_megapixels / current_megapixels) ** 0.5
            new_size = (int(img.width * scaling_factor), int(img.height * scaling_factor))
            # Resize the image
            resized_img = img.resize(new_size, Image.LANCZOS)
            # Save the resized image
            resized_img.save(output_path)
        else:
            # If the image is smaller than 10 megapixels, save it as is
            img.save(output_path)

def main():
    # Iterate over the input directory
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for filename in os.listdir(input_directory):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(input_directory, filename)
                output_path = os.path.join(output_directory, filename)
                # Check if the output file already exists
                if not os.path.exists(output_path):
                    executor.submit(resize_image, input_path, output_path)
                    print(f"Submitted {filename} for resizing.")
                else:
                    print(f"Skipped {filename}, already exists in {output_directory}")

    print("All images have been resized and saved to the output directory.")

if __name__ == "__main__":
    main()