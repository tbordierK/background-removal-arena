import os
from PIL import Image

def add_green_background_to_image(image_path, output_path, background_color=(0, 255, 0)):
    """Add a green background to an image and save it as PNG."""
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, background_color + (255,))
        combined = Image.alpha_composite(background, img)
        combined.save(output_path, "PNG")

def process_directory(input_dir, output_dir, background_color=(0, 255, 0)):
    """Recursively process a directory to add a green background to all images and convert them to PNG."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.png')

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Check if the output file already exists
                if not os.path.exists(output_path):
                    # Add green background to the image and convert to PNG
                    add_green_background_to_image(input_path, output_path, background_color)
                    print(f"Processed: {input_path} -> {output_path}")
                else:
                    print(f"Skipped: {output_path} already exists")

# Example usage
input_directory = "../../background-removal-arena-v0/train/data/resized"
output_directory = "../data/resized-green/"
process_directory(input_directory, output_directory)