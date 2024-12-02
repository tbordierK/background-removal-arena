import os
import shutil

def copy_images(source_dir, dest_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Walk through the source directory
    for root, _, files in os.walk(source_dir):
        for filename in files:
            print(filename)
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
    # Define your source and destination directories
    source_directory = '../categorised-images'
    destination_directory = '../original-images'

    # Call the function to copy images
    copy_images(source_directory, destination_directory)

if __name__ == "__main__":
    main()