import os
from photoroom import process_image as photoroom_process
from removebg import process_image as removebg_process
#from clipdrop import process_image as clipdrop_process
from bria_rmbg20 import process_image as bria_process

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_images(input_directory, output_directory, process_function, limit=None):
    count = 0
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.heic')):
                file_path = os.path.join(root, file)
                result_file_name = os.path.splitext(os.path.basename(file_path))[0] + '.png'
                result_file_directory = os.path.join(output_directory)

                if not os.path.exists(result_file_directory):
                    os.makedirs(result_file_directory)

                result_path = os.path.join(result_file_directory, result_file_name)

                if not os.path.exists(result_path):  # Check if the image has already been processed
                    print(file_path, result_path)
                    process_function(file_path, result_path)
                    count += 1
                    if limit and count >= limit:
                        return

def main(dry_run=False):
    input_directory = "../data/resized-original-images"
    output_base_directory = "../data/processed"

    # Define output directories for each API
    output_directories = {
        "photoroom": os.path.join(output_base_directory, "photoroom"),
        "removebg": os.path.join(output_base_directory, "removebg"),
        #"clipdrop": os.path.join(output_base_directory, "clipdrop"),
        "bria": os.path.join(output_base_directory, "bria")
    }

    # Create output directories if they don't exist
    for directory in output_directories.values():
        create_directory(directory)

    if dry_run:
        print("Starting dry run...")
        k = 5
        process_images(input_directory, output_directories["photoroom"], photoroom_process, limit=k)
        process_images(input_directory, output_directories["removebg"], removebg_process, limit=k)
        #process_images(input_directory, output_directories["clipdrop"], clipdrop_process, limit=k)
        process_images(input_directory, output_directories["bria"], bria_process, limit=k)
        print("Dry run completed.")
    else:
        print("Starting full processing...")
        process_images(input_directory, output_directories["photoroom"], photoroom_process)
        process_images(input_directory, output_directories["removebg"], removebg_process)
        #process_images(input_directory, output_directories["clipdrop"], clipdrop_process)
        process_images(input_directory, output_directories["bria"], bria_process)
        print("Full processing completed.")

if __name__ == "__main__":
    # Set dry_run to True for a dry run, or False for full processing
    main(dry_run=False)