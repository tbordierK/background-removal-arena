from datasets import Dataset, Features, Value, Image
from huggingface_hub import HfApi
import os
from collections import defaultdict
import pandas as pd
import argparse

def upload_to_dataset(original_images_dir, processed_images_dir, dataset_name, dry_run=False):
    # Define the dataset features with dedicated columns for each model
    features = Features({
        "original_image": Image(),  # Original image feature
        "clipdrop_image": Image(),  # Clipdrop segmented image
        "bria_image": Image(),      # Bria segmented image
        "photoroom_image": Image(), # Photoroom segmented image
        "removebg_image": Image(),  # RemoveBG segmented image
        "original_filename": Value("string")  # Original filename
    })

    # Load image paths and metadata
    data = defaultdict(lambda: {
        "clipdrop_image": None,
        "bria_image": None,
        "photoroom_image": None,
        "removebg_image": None
    })

    # Walk into the original images folder
    for root, _, files in os.walk(original_images_dir):
        for f in files:
            if f.endswith(('.png', '.jpg', '.jpeg')):
                original_image_path = os.path.join(root, f)
                data[f]["original_image"] = original_image_path
                data[f]["original_filename"] = f

                # Check for corresponding images in processed directories
                for source in ["clipdrop", "bria", "photoroom", "removebg"]:
                    # Check for processed images ending in .png or .jpg
                    for ext in ['.png', '.jpg']:
                        processed_image_filename = os.path.splitext(f)[0] + ext
                        source_image_path = os.path.join(processed_images_dir, source, processed_image_filename)
        
                        if os.path.exists(source_image_path):
                            data[f][f"{source}_image"] = source_image_path
                            break  # Stop checking other extensions if a file is found

    # Convert the data to a dictionary of lists
    dataset_dict = {
        "original_image": [],
        "clipdrop_image": [],
        "bria_image": [],
        "photoroom_image": [],
        "removebg_image": [],
        "original_filename": []
    }

    for filename, entry in data.items():
        if "original_image" in entry:
            dataset_dict["original_image"].append(entry["original_image"])
            dataset_dict["clipdrop_image"].append(entry["clipdrop_image"])
            dataset_dict["bria_image"].append(entry["bria_image"])
            dataset_dict["photoroom_image"].append(entry["photoroom_image"])
            dataset_dict["removebg_image"].append(entry["removebg_image"])
            dataset_dict["original_filename"].append(filename)

    # Save the data dictionary to a CSV file for inspection
    df = pd.DataFrame.from_dict(dataset_dict)
    df.to_csv("image_data.csv", index=False)

    # Create a Dataset
    dataset = Dataset.from_dict(dataset_dict, features=features)

    if dry_run:
        print("Dry run: Dataset prepared but not pushed to Hugging Face Hub.")
        print(df.head())  # Display the first few rows for inspection
    else:
        # Push the dataset to Hugging Face Hub in a private way
        api = HfApi()
        dataset.push_to_hub(dataset_name, token=api.token, private=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload images to a Hugging Face dataset.")
    parser.add_argument("original_images_dir", type=str, help="Directory containing the original images.")
    parser.add_argument("processed_images_dir", type=str, help="Directory containing the processed images with subfolders for each model.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to upload to Hugging Face Hub.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without uploading to the hub.")
    
    args = parser.parse_args()
    
    upload_to_dataset(args.original_images_dir, args.processed_images_dir, args.dataset_name, dry_run=args.dry_run)
