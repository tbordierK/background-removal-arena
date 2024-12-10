import os
import argparse
import shutil
import sys
from dotenv import load_dotenv, find_dotenv

# Importing modules from the utils package
from utils.resize_images import main as resize_images_main
from utils.removebg import iterate_over_directory as removebg_iterate
from utils.photoroom import iterate_over_directory as photoroom_iterate
from utils.bria_rmbg20 import iterate_over_directory as bria_iterate
from utils.add_green_background import process_directory as add_green_background_process
from utils.upload_to_dataset import upload_to_dataset
from utils.resize_processed_images import process_images

def check_env_variables():
    """Check if the necessary environment variables are loaded."""
    if not find_dotenv():
        sys.exit("Error: .env file not found.")
    
    load_dotenv()
    
    required_keys = ['REMOVEBG_API_KEY', 'PHOTOROOM_API_KEY', 'BRIA_API_TOKEN']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        sys.exit(f"Error: Missing environment variables: {', '.join(missing_keys)}")

def copy_images(source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    valid_extensions = ('.png', '.jpg', '.jpeg')

    # Walk through the source directory
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                source_file = os.path.join(root, filename)
                
                # Extract the folder name
                folder_name = os.path.basename(root)
                # Append folder name to the filename
                new_filename = f"{folder_name}_{filename}"
                dest_file = os.path.join(dest_dir, new_filename)

                # Check if the file is an image and doesn't already exist in the destination
                if os.path.isfile(source_file) and not os.path.exists(dest_file):
                    shutil.copy2(source_file, dest_file)
                    print(f"Copied: {new_filename}")
                else:
                    print(f"Skipped: {filename} (already exists or not a file)")

def main():
    check_env_variables()

    parser = argparse.ArgumentParser(description="Image Processing Pipeline")
    parser.add_argument("--input-dir", type=str, default="original-images", help="Input directory for images")
    parser.add_argument("--work-dir", type=str, default="workdir", help="Working directory for intermediate images")
    parser.add_argument("--output-dir", type=str, default="final-images", help="Output directory for final images")
    parser.add_argument("--dataset-name", type=str, help="Name of the dataset to upload to Hugging Face Hub")
    parser.add_argument("--push-dataset", action="store_true", help="Push the dataset to the Hugging Face Hub")

    args = parser.parse_args()

    # Define intermediate directories within the work directory
    input_resized_dir = os.path.join(args.work_dir, "resized")
    bg_removed_dir = os.path.join(args.work_dir, "background-removed")
    green_bg_dir = os.path.join(args.work_dir, "green-background")

    # Ensure all directories exist
    for directory in [input_resized_dir, bg_removed_dir, green_bg_dir]:
        os.makedirs(directory, exist_ok=True)

    # Step 4: Move images to final output directory
    print("Moving images to final output directory...")
    original_images_dir = os.path.join(args.work_dir, "merged-categories")
    copy_images(args.input_dir, original_images_dir)

    # Step 1: Resize images
    print("Resizing images...")
    resize_images_main(input_directory=original_images_dir, output_directory=input_resized_dir)

    # Step 2: Remove background
    print("Removing backgrounds...")
    bg_removal_dirs = {
        "removebg": os.path.join(bg_removed_dir, "removebg"),
        "photoroom": os.path.join(bg_removed_dir, "photoroom"),
        "bria": os.path.join(bg_removed_dir, "bria")
    }

    for dir_path in bg_removal_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    removebg_iterate(input_resized_dir, bg_removal_dirs["removebg"])
    photoroom_iterate(input_resized_dir, bg_removal_dirs["photoroom"])
    bria_iterate(input_resized_dir, bg_removal_dirs["bria"])

    print("Adding green background...")
    add_green_background_process(bg_removed_dir, green_bg_dir)

    print("Resizing processed images...")
    target_width = 800 
    subdirectories = ["bria", "photoroom", "clipdrop", "removebg"]
    os.makedirs(args.output_dir, exist_ok=True)
    for subdir in subdirectories:
        input_directory = os.path.join(green_bg_dir, subdir)
        output_directory = os.path.join(args.output_dir, subdir)
        process_images(input_directory, output_directory, target_width)
    
    original_output_directory = os.path.join(args.output_dir, "web-original-images")
    process_images(original_images_dir, original_output_directory, target_width)

    if args.dataset_name:
        upload_to_dataset(original_output_directory, args.output_dir, args.dataset_name, dry_run=not args.push_dataset)
    else:
        print("Please provide a dataset name using --dataset-name")

if __name__ == "__main__":
    main()