from datasets import Dataset, Features, Value, Image
from huggingface_hub import HfApi
import os
from collections import defaultdict
import pandas as pd
import argparse
from PIL import Image as PILImage
import sys
import logging

def upload_to_dataset(original_images_dir, processed_images_dir, dataset_name, dry_run=False):
    """Upload images to a Hugging Face dataset including BiRefNet results."""
    
    logging.info(f"Starting dataset upload from {original_images_dir}")
    
    # Define the dataset features with dedicated columns for each model
    features = Features({
        "original_image": Image(),
        "clipdrop_image": Image(),
        "bria_image": Image(),
        "photoroom_image": Image(),
        "removebg_image": Image(),
        "birefnet_image": Image(),
        "original_filename": Value("string")
    })

    # Load image paths and metadata
    data = defaultdict(lambda: {
        "clipdrop_image": None,
        "bria_image": None,
        "photoroom_image": None,
        "removebg_image": None,
        "birefnet_image": None
    })

    # Walk into the original images folder
    for root, _, files in os.walk(original_images_dir):
        for f in files:
            if f.endswith(('.png', '.jpg', '.jpeg')):
                original_image_path = os.path.join(root, f)
                data[f]["original_image"] = original_image_path
                data[f]["original_filename"] = f

                # Check for corresponding images in processed directories
                for source in ["clipdrop", "bria", "photoroom", "removebg", "birefnet"]:
                    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                        processed_image_filename = os.path.splitext(f)[0] + ext
                        source_image_path = os.path.join(processed_images_dir, source, processed_image_filename)
        
                        if os.path.exists(source_image_path):
                            data[f][f"{source}_image"] = source_image_path
                            break
                        
    # Convert the data to a dictionary of lists
    dataset_dict = {
        "original_image": [],
        "clipdrop_image": [],
        "bria_image": [],
        "photoroom_image": [],
        "removebg_image": [],
        "birefnet_image": [],
        "original_filename": []
    }

    errors = []
    processed_count = 0
    skipped_count = 0

    for filename, entry in data.items():
        if "original_image" in entry:
            try:
                original_size = PILImage.open(entry["original_image"]).size
                valid_entry = True

                for source in ["clipdrop_image", "bria_image", "photoroom_image", "removebg_image", "birefnet_image"]:
                    if entry[source] is not None:
                        try:
                            processed_size = PILImage.open(entry[source]).size
                            if processed_size != original_size:
                                errors.append(f"Size mismatch for {filename}: {source}")
                                valid_entry = False
                        except Exception as e:
                            errors.append(f"Error with {filename}: {source}")
                            valid_entry = False

                if valid_entry:
                    for key in dataset_dict.keys():
                        if key in entry:
                            dataset_dict[key].append(entry[key])
                    processed_count += 1
                else:
                    skipped_count += 1

            except Exception as e:
                errors.append(f"Error processing {filename}")
                skipped_count += 1

    if errors:
        logging.warning(f"Encountered {len(errors)} errors during processing")

    logging.info(f"Processed: {processed_count}, Skipped: {skipped_count}, Total: {processed_count + skipped_count}")

    # Save the data dictionary to a CSV file for inspection
    df = pd.DataFrame.from_dict(dataset_dict)
    df.to_csv("image_data.csv", index=False)

    # Create a Dataset
    dataset = Dataset.from_dict(dataset_dict, features=features)

    if dry_run:
        logging.info("Dry run completed - dataset not pushed")
    else:
        logging.info(f"Pushing dataset to {dataset_name}")
        api = HfApi()
        dataset.push_to_hub(dataset_name, token=api.token, private=True)
        logging.info("Upload completed successfully")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    parser = argparse.ArgumentParser(description="Upload images to a Hugging Face dataset.")
    parser.add_argument("original_images_dir", type=str, help="Directory containing the original images.")
    parser.add_argument("processed_images_dir", type=str, help="Directory containing the processed images with subfolders for each model.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to upload to Hugging Face Hub.")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without uploading to the hub.")
    
    args = parser.parse_args()
    
    upload_to_dataset(args.original_images_dir, args.processed_images_dir, args.dataset_name, dry_run=args.dry_run)
