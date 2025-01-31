import os
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def create_gray_checkerboard(shape: tuple, square_size: int):
    """
    Create a gray-scale checkerboard pattern array with the given shape.
    The pattern alternates between 0.8 and 1.0 values in a square_size grid.
    """
    # shape = (height, width)
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # Scale 0.2 + 0.8 => (0.2, 1.0) can also be used, but in this example we'll keep 0.8 and 1.0
    # (depending on your desired brightness).
    board = ((x // square_size + y // square_size) % 2) * 0.2 + 0.8
    return board

def add_checkered_background_to_image(image_path, output_path, square_size=20):
    """
    Add a gray checkerboard background to an image and save it as PNG.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGBA")

        # Create checkerboard pattern.
        # Using the size (height, width) from the image.
        checkerboard_array = create_gray_checkerboard((img.height, img.width), square_size)

        # Convert checkerboard_array into an RGBA image (gray -> R=G=B, alpha=255)
        # First convert float values into 8-bit grayscale.
        # Expand dims to make it into (height, width, 1).
        checkerboard_gray = (checkerboard_array * 255).astype(np.uint8)
        checkerboard_gray = np.expand_dims(checkerboard_gray, axis=2)

        # Stack 3 copies (for R,G,B) plus one alpha channel of 255.
        alpha_channel = np.full_like(checkerboard_gray, 255)
        checkerboard_rgba = np.concatenate([checkerboard_gray]*3 + [alpha_channel], axis=2)

        background = Image.fromarray(checkerboard_rgba, mode="RGBA")

        # Composite the image over the checkerboard background
        combined = Image.alpha_composite(background, img)
        combined.save(output_path, "PNG")

def process_image_file(input_path, output_path, square_size):
    """
    Process a single image file to add a checkerboard background.
    """
    if not os.path.exists(output_path):
        add_checkered_background_to_image(input_path, output_path, square_size)
        print(f"Processed (checkerboard): {input_path} -> {output_path}")
    else:
        print(f"Skipped (checkerboard): {output_path} already exists")

def process_directory(input_dir, output_dir, square_size=20):
    """
    Recursively process a directory to add a checkerboard background to all images and convert them to PNG.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = []
    with ThreadPoolExecutor() as executor:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    input_path = os.path.join(root, file)
                    relative_path = os.path.relpath(input_path, input_dir)
                    output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + '.png')

                    # Ensure the output directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    # Submit the task to the executor
                    tasks.append(executor.submit(process_image_file, input_path, output_path, square_size))

    # Wait for all tasks to complete
    for task in tasks:
        task.result() 