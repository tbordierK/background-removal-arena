from PIL import Image, ExifTags
import os
from concurrent.futures import ThreadPoolExecutor

def create_directory(path):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def correct_orientation(img):
    """Correct image orientation using EXIF data."""
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(orientation, None)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # Cases: image doesn't have getexif
        pass
    return img

def resize_image(input_path, output_path, target_width):
    """Resize an image to the target width while maintaining aspect ratio."""
    with Image.open(input_path) as img:
        # Correct orientation
        img = correct_orientation(img)
        
        # Calculate the new height to maintain the aspect ratio
        width_percent = target_width / img.width
        target_height = int(img.height * width_percent)
        
        # Resize the image
        img = img.resize((target_width, target_height), Image.LANCZOS)
        
        # Save the resized image in the same format as the input
        img.save(output_path, format=img.format)

def process_image_file(file_path, result_path, target_width):
    """Process a single image file."""
    if not os.path.exists(result_path):
        print(f"Resizing {file_path} to {result_path}")
        resize_image(file_path, result_path, target_width)
    else:
        print(f"Skipped {file_path}, already resized.")

def process_images(input_directory, output_directory, target_width):
    """Process and resize images from the input directory to the output directory."""
    create_directory(output_directory)
    
    with ThreadPoolExecutor() as executor:
        for root, _, files in os.walk(input_directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    result_file_name = os.path.splitext(file)[0] + os.path.splitext(file)[1]
                    result_path = os.path.join(output_directory, result_file_name)
                    executor.submit(process_image_file, file_path, result_path, target_width)

def main():
    """Main function to resize images in specified subdirectories."""
    # Process images in the processed directory
    base_input_directory = "../data/processed"
    base_output_directory = "../data/resized"
    target_width = 800  # Set the desired width for web display

    # List of subdirectories to process
    subdirectories = ["bria", "photoroom", "clipdrop", "removebg"]

    for subdir in subdirectories:
        input_directory = os.path.join(base_input_directory, subdir)
        output_directory = os.path.join(base_output_directory, subdir)
        process_images(input_directory, output_directory, target_width)

    # Additionally, process images in the resized-original-images directory
    original_input_directory = "../data/resized-original-images"
    original_output_directory = "../data/web-original-images"
    process_images(original_input_directory, original_output_directory, target_width)

    print("Image resizing completed.")

if __name__ == "__main__":
    main()