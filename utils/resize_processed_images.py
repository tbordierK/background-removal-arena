from PIL import Image
import os

def create_directory(path):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def resize_image(input_path, output_path, target_width):
    """Resize an image to the target width while maintaining aspect ratio."""
    with Image.open(input_path) as img:
        # Calculate the new height to maintain the aspect ratio
        width_percent = target_width / img.width
        target_height = int(img.height * width_percent)
        
        # Resize the image
        img = img.resize((target_width, target_height), Image.LANCZOS)
        
        # Save the resized image in the same format as the input
        img.save(output_path, format=img.format)

def process_images(input_directory, output_directory, target_width):
    """Process and resize images from the input directory to the output directory."""
    create_directory(output_directory)
    
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.heic')):
                file_path = os.path.join(root, file)
                result_file_name = os.path.splitext(file)[0] + os.path.splitext(file)[1]
                result_path = os.path.join(output_directory, result_file_name)
                
                # Check if the output file already exists
                if not os.path.exists(result_path):
                    print(f"Resizing {file_path} to {result_path}")
                    resize_image(file_path, result_path, target_width)
                else:
                    print(f"Skipped {file_path}, already resized.")

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