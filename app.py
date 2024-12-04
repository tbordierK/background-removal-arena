import os
import json
import time
import random
import logging
import threading
from pathlib import Path
from uuid import uuid4
from typing import Tuple

import numpy as np
import gradio as gr
from PIL import Image
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import CommitScheduler

from db import (
    compute_elo_scores,
    get_all_votes,
    add_vote,
    is_running_in_space,
    fill_database_once
)

token = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Load datasets
dataset = load_dataset("bgsys/background-removal-arena-green", split='train')
fill_database_once()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Directory and path setup for JSON dataset
JSON_DATASET_DIR = Path("data/json_dataset")
JSON_DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Initialize CommitScheduler for Hugging Face only if running in space
scheduler = None
if is_running_in_space():
    scheduler = CommitScheduler(
        repo_id="bgsys/votes_datasets_test2",
        repo_type="dataset",
        folder_path=JSON_DATASET_DIR,
        path_in_repo="data",
        token=token
    )

def fetch_elo_scores():
    """Fetch and log Elo scores."""
    try:
        elo_scores = compute_elo_scores()
        logging.info("Elo scores successfully computed.")
        return elo_scores
    except Exception as e:
        logging.error("Error computing Elo scores: %s", str(e))
        return None

def update_rankings_table():
    """Update and return the rankings table based on Elo scores."""
    elo_scores = fetch_elo_scores()
    if elo_scores:
        rankings = [
            ["Photoroom", int(elo_scores.get("Photoroom", 1000))],
            #["Clipdrop", int(elo_scores.get("Clipdrop", 1000))],
            ["RemoveBG", int(elo_scores.get("RemoveBG", 1000))],
            ["BRIA RMBG 2.0", int(elo_scores.get("BRIA RMBG 2.0", 1000))],
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    else:
        return [
            ["Photoroom", -1],
            #["Clipdrop", -1],
            ["RemoveBG", -1],
            ["BRIA RMBG 2.0", -1],
        ]

def select_new_image():
    """Select a new image and its segmented versions."""
    max_attempts = 10
    last_image_index = None

    for _ in range(max_attempts):
        available_indices = [i for i in range(len(dataset)) if i != last_image_index]
        
        if not available_indices:
            logging.error("No available images to select from.")
            return None

        random_index = random.choice(available_indices)
        sample = dataset[random_index]
        input_image = sample['original_image']

        segmented_images = [sample['clipdrop_image'], sample['bria_image'],
                            sample['photoroom_image'], sample['removebg_image']]
        segmented_sources = ['Clipdrop', 'BRIA RMBG 2.0', 'Photoroom', 'RemoveBG']
        
        if segmented_images.count(None) > 2:
            logging.error("Not enough segmented images found for: %s. Resampling another image.", sample['original_filename'])
            last_image_index = random_index
            continue

        try:
            selected_indices = random.sample([i for i, img in enumerate(segmented_images) if img is not None], 2)
            model_a_index, model_b_index = selected_indices
            model_a_output_image = segmented_images[model_a_index]
            model_b_output_image = segmented_images[model_b_index]
            model_a_name = segmented_sources[model_a_index]
            model_b_name = segmented_sources[model_b_index]
            return sample['original_filename'], input_image, model_a_output_image, model_b_output_image, model_a_name, model_b_name
        except Exception as e:
            logging.error("Error processing images: %s. Resampling another image.", str(e))
            last_image_index = random_index

    logging.error("Failed to select a new image after %d attempts.", max_attempts)
    return None

def get_notice_markdown():
    """Generate the notice markdown with dynamic vote count."""
    total_votes = len(get_all_votes())
    return f"""
    # ⚔️  Background Removal Arena: Compare & Test the Best Background Removal Models

    ## 📜 How It Works
    - **Blind Test**: You will see two images with their background removed from two anonymous background removal models (Clipdrop, RemoveBG, Photoroom, BRIA RMBG 2.0).
    - **Vote for the Best**: Choose the best result, if none stand out choose "Tie". 

    ## 📊 Stats
    - **Total #votes**: {total_votes}

    ## 👇 Test now!
    """

def compute_mask_difference(segmented_a, segmented_b):
    """Compute the absolute difference between two image masks, ignoring green background."""
    mask_a = np.asarray(segmented_a)
    mask_b = np.asarray(segmented_b)

    # Define the green background color
    green_background = (0, 255, 0, 255)

    # Create a binary mask where non-green and non-transparent pixels are marked as 1
    mask_a_1d = np.where(
        (mask_a[..., :3] != green_background[:3]).any(axis=-1) & (mask_a[..., 3] != 0), 1, 0
    )
    mask_b_1d = np.where(
        (mask_b[..., :3] != green_background[:3]).any(axis=-1) & (mask_b[..., 3] != 0), 1, 0
    )

    # Compute the absolute difference between the masks
    return np.abs(mask_a_1d - mask_b_1d)

def gradio_interface():
    """Create and return the Gradio interface."""
    with gr.Blocks() as demo:
        gr.Markdown("# Background Removal Arena")
        button_name = "Difference between masks"

        with gr.Tabs() as tabs:
            with gr.Tab("⚔️ Arena (battle)", id=0):
                notice_markdown = gr.Markdown(get_notice_markdown(), elem_id="notice_markdown")

                filname, input_image, segmented_a, segmented_b, a_name, b_name = select_new_image()
                model_a_name = gr.State(a_name)
                model_b_name = gr.State(b_name)
                fpath_input = gr.State(filname)

                # Compute the absolute difference between the masks
                mask_difference = compute_mask_difference(segmented_a, segmented_b)

                with gr.Row():
                    image_a_display = gr.Image(
                        value=segmented_a,
                        type="pil",
                        label="Model A",
                        width=500,
                        height=500
                    )
                    input_image_display = gr.AnnotatedImage(
                        value=(input_image, [(mask_difference > 0, button_name)]),
                        label="Input Image",
                        width=500,
                        height=500
                    )
                    image_b_display = gr.Image(
                        value=segmented_b,
                        type="pil",
                        label="Model B",
                        width=500,
                        height=500
                    )
                tie = gr.State("Tie")
                with gr.Row():
                    vote_a_btn = gr.Button("👈  A is better")
                    vote_tie = gr.Button("🤝  Tie")
                    vote_b_btn = gr.Button("👉  B is better")

                
                vote_a_btn.click(
                    fn=lambda: vote_for_model("model_a", fpath_input, model_a_name, model_b_name),
                    outputs=[
                        fpath_input, input_image_display, image_a_display, image_b_display, model_a_name, model_b_name, notice_markdown
                    ]
                )
                vote_b_btn.click(
                    fn=lambda: vote_for_model("model_b",fpath_input, model_a_name, model_b_name),
                    outputs=[
                        fpath_input, input_image_display, image_a_display, image_b_display, model_a_name, model_b_name, notice_markdown
                    ]
                )
                vote_tie.click(
                    fn=lambda: vote_for_model("tie", fpath_input, model_a_name, model_b_name),
                    outputs=[
                        fpath_input, input_image_display, image_a_display, image_b_display, model_a_name, model_b_name, notice_markdown
                    ]
                )
            
                def vote_for_model(choice, original_filename, model_a_name, model_b_name):
                    """Submit a vote for a model and return updated images and model names."""
                    logging.info("Voting for model: %s", choice)
                    vote_data = {
                        "image_id": original_filename.value,
                        "model_a": model_a_name.value,
                        "model_b": model_b_name.value,
                        "winner": choice,
                    }

                    try:
                        logging.debug("Adding vote data to the database: %s", vote_data)
                        result = add_vote(vote_data)
                        logging.info("Vote successfully recorded in the database with ID: %s", result["id"])
                    except Exception as e:
                        logging.error("Error recording vote: %s", str(e))

                    new_fpath_input, new_input_image, new_segmented_a, new_segmented_b, new_a_name, new_b_name = select_new_image()
                    model_a_name.value = new_a_name
                    model_b_name.value = new_b_name
                    fpath_input.value = new_fpath_input

                    mask_difference = compute_mask_difference(new_segmented_a, new_segmented_b)

                    # Update the notice markdown with the new vote count
                    new_notice_markdown = get_notice_markdown()

                    return (fpath_input.value, (new_input_image, [(mask_difference, button_name)]), new_segmented_a,
                             new_segmented_b, model_a_name.value, model_b_name.value, new_notice_markdown)
           
            with gr.Tab("🏆 Leaderboard", id=1) as leaderboard_tab:
                rankings_table = gr.Dataframe(
                    headers=["Model", "Ranking"],
                    value=update_rankings_table(),
                    label="Current Model Rankings",
                    column_widths=[180, 60],
                    row_count=4
                )

                leaderboard_tab.select(
                    fn=lambda: update_rankings_table(),
                    outputs=rankings_table
                )

            with gr.Tab("📊 Vote Data", id=2) as vote_data_tab:
                def update_vote_data():
                    votes = get_all_votes()
                    return [[vote.id, vote.image_id, vote.model_a, vote.model_b, vote.winner, vote.timestamp] for vote in votes]

                vote_table = gr.Dataframe(
                    headers=["ID", "Image ID", "Model A", "Model B", "Winner", "Timestamp"],
                    value=update_vote_data(),
                    label="Vote Data",
                    column_widths=[20, 150, 100, 100, 100, 150],
                    row_count=0
                )

                vote_data_tab.select(
                    fn=lambda: update_vote_data(),
                    outputs=vote_table
                )

    return demo

def dump_database_to_json():
    """Dump the database to a JSON file and upload it to Hugging Face."""
    if not is_running_in_space():
        logging.info("Not running in Hugging Face Spaces. Skipping database dump.")
        return

    votes = get_all_votes()
    json_data = [
        {
            "id": vote.id,
            "image_id": vote.image_id,
            "model_a": vote.model_a,
            "model_b": vote.model_b,
            "winner": vote.winner,
            "user_id": vote.user_id,
            "timestamp": vote.timestamp.isoformat()
        }
        for vote in votes
    ]

    json_file_path = JSON_DATASET_DIR / "votes.json"
    # Upload to Hugging Face
    with scheduler.lock:
        with json_file_path.open("w") as f:
            json.dump(json_data, f, indent=4)

    logging.info("Database dumped to JSON")

def schedule_dump_database(interval=60):
    """Schedule the database dump to JSON every specified interval in seconds."""
    def run():
        while True:
            logging.info("Starting database dump to JSON.")
            dump_database_to_json()
            logging.info("Database dump completed. Sleeping for %d seconds.", interval)
            time.sleep(interval)

    if is_running_in_space():
        logging.info("Initializing database dump scheduler with interval: %d seconds.", interval)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        logging.info("Database dump scheduler started.")
    else:
        logging.info("Not running in Hugging Face Spaces. Database dump scheduler not started.")

if __name__ == "__main__":
    schedule_dump_database()  # Start the periodic database dump
    demo = gradio_interface()
    demo.launch()