import os
import json
import time
import random
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import gradio as gr
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

# Load environment variables
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load datasets and initialize database
dataset = load_dataset("bgsys/background-removal-arena-green", split='train')
fill_database_once()

# Directory setup for JSON dataset
JSON_DATASET_DIR = Path("data/json_dataset")
JSON_DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Initialize CommitScheduler if running in space
commit_scheduler = CommitScheduler(
    repo_id="bgsys/votes_datasets_test2",
    repo_type="dataset",
    folder_path=JSON_DATASET_DIR,
    path_in_repo="data",
    token=huggingface_token
) if is_running_in_space() else None

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
    elo_scores = fetch_elo_scores() or {}
    default_score = 1000
    rankings = [
        ["Photoroom", int(elo_scores.get("Photoroom", default_score))],
        ["RemoveBG", int(elo_scores.get("RemoveBG", default_score))],
        ["BRIA RMBG 2.0", int(elo_scores.get("BRIA RMBG 2.0", default_score))],
    ]
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings

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

        segmented_images = [sample.get(key) for key in ['clipdrop_image', 'bria_image', 'photoroom_image', 'removebg_image']]
        segmented_sources = ['Clipdrop', 'BRIA RMBG 2.0', 'Photoroom', 'RemoveBG']
        
        if segmented_images.count(None) > 2:
            logging.error("Not enough segmented images found for: %s. Resampling another image.", sample['original_filename'])
            last_image_index = random_index
            continue

        try:
            selected_indices = random.sample([i for i, img in enumerate(segmented_images) if img is not None], 2)
            model_a_index, model_b_index = selected_indices
            return (
                sample['original_filename'], input_image,
                segmented_images[model_a_index], segmented_images[model_b_index],
                segmented_sources[model_a_index], segmented_sources[model_b_index]
            )
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
                image_width = 800
                image_height = 800
                notice_markdown = gr.Markdown(get_notice_markdown(), elem_id="notice_markdown")
                with gr.Row(equal_height=True):
                    def on_enter_contest(username):
                        feedback_message = f"Thank you, {username or 'anonymous'}! You can see how you rank in the Hall of Fame."
                        logging.info(feedback_message)
                        return feedback_message
                     
                    with gr.Column(scale=2):
                        username_input = gr.Textbox(
                            label="Enter your username (optional)",
                            placeholder="✨ Enter your username (optional)",
                            show_label=False,
                            submit_btn="Enter",
                            interactive=True
                        )

                    with gr.Column(scale=3):
                        feedback_output = gr.Textbox(
                            label="Feedback",
                            interactive=False,
                            show_label=False
                        )

                    username_input.submit(
                        fn=on_enter_contest,
                        inputs=username_input,
                        outputs=feedback_output
                    )
                    
                def refresh_states():
                    filename, input_image, segmented_a, segmented_b, model_a_name, model_b_name = select_new_image()
                    mask_difference = compute_mask_difference(segmented_a, segmented_b)
                    image_a = gr.Image(
                        value=segmented_a,
                        label="Image",
                        width=image_width,
                        height=image_height
                    )
                    
                    input_image_display = gr.AnnotatedImage(
                        value=(input_image, [(mask_difference > 0, button_name)]),
                        label="Input Image",
                        width=image_width,
                        height=image_height
                    )

                    image_b = gr.Image(
                        value=segmented_b,
                        label="Image",
                        width=image_width,
                        height=image_height
                    )
                    
                    zoomed_state_a = gr.State(False)
                    zoomed_state_b = gr.State(False)

                    state_model_a_name = gr.State(model_a_name)
                    state_model_b_name = gr.State(model_b_name)
                    state_filename = gr.State(filename)
                    state_segmented_a = gr.Image(value=segmented_a, visible=False)
                    state_segmented_b = gr.Image(value=segmented_b, visible=False)
            
                    outputs = [
                        state_filename, image_a, image_b, state_model_a_name, state_model_b_name, 
                        input_image_display, zoomed_state_a, zoomed_state_b, state_segmented_a, state_segmented_b
                    ]
                    return outputs

                with gr.Row():
                    state_filename, image_a, image_b, state_model_a_name, state_model_b_name, input_image_display, zoomed_state_a, zoomed_state_b, state_segmented_a, state_segmented_b = refresh_states()
                  
                with gr.Row():
                    vote_a_button = gr.Button("👈  A is better")
                    vote_tie_button = gr.Button("🤝  Tie")
                    vote_b_button = gr.Button("👉  B is better")

                def vote_for_model(choice, original_filename, model_a_name, model_b_name, user_username):
                    """Submit a vote for a model and return updated images and model names."""
                    logging.info("Voting for model: %s", choice)
                    vote_data = {
                        "image_id": original_filename.value,
                        "model_a": model_a_name.value,
                        "model_b": model_b_name.value,
                        "winner": choice,
                        "user_id": user_username or "anonymous"
                    }

                    try:
                        logging.debug("Adding vote data to the database: %s", vote_data)
                        result = add_vote(vote_data)
                        logging.info("Vote successfully recorded in the database with ID: %s", result["id"])
                    except Exception as e:
                        logging.error("Error recording vote: %s", str(e))

                    outputs = refresh_states()
                    new_notice_markdown = get_notice_markdown()

                    return outputs + [new_notice_markdown]

                vote_a_button.click(
                    fn=lambda username: vote_for_model("model_a", state_filename, state_model_a_name, state_model_b_name, username),
                    inputs=[username_input],
                    outputs=[
                        state_filename, image_a, image_b, state_model_a_name, state_model_b_name, 
                        input_image_display, zoomed_state_a, zoomed_state_b, 
                        state_segmented_a, state_segmented_b, notice_markdown
                    ]
                )
                vote_b_button.click(
                    fn=lambda username: vote_for_model("model_b", state_filename, state_model_a_name, state_model_b_name, username),
                    inputs=[username_input],
                    outputs=[
                        state_filename, image_a, image_b, state_model_a_name, state_model_b_name, 
                        input_image_display, zoomed_state_a, zoomed_state_b, 
                        state_segmented_a, state_segmented_b, notice_markdown
                    ]
                )
                vote_tie_button.click(
                    fn=lambda username: vote_for_model("tie", state_filename, state_model_a_name, state_model_b_name, username),
                    inputs=[username_input],
                    outputs=[
                        state_filename, image_a, image_b, state_model_a_name, state_model_b_name, 
                        input_image_display, zoomed_state_a, zoomed_state_b, 
                        state_segmented_a, state_segmented_b, notice_markdown
                    ]
                )


                def handle_zoom(current_image, zoomed_state_input, original_image, event: gr.SelectData):
                    """Toggle between zoomed and original image based on click events."""
                    
                    # Check if zoomed_state is a gr.State and get its value
                    if isinstance(zoomed_state_input, gr.State):
                        zoomed_state = zoomed_state_input.value
                    else:
                        zoomed_state = zoomed_state_input

                    if zoomed_state:
                        return gr.Image(
                            value=original_image,
                            label="Image",
                            width=image_width,
                            height=image_height
                        ), False

                    start_row, start_col = event.index[1], event.index[0]
                    zoom_size = max(10, min(current_image.shape[:2]) // 10)

                    row_start, row_end = max(start_row - zoom_size, 0), min(start_row + zoom_size, current_image.shape[0])
                    col_start, col_end = max(start_col - zoom_size, 0), min(start_col + zoom_size, current_image.shape[1])

                    grey_image = np.mean(current_image, axis=-1, keepdims=True).astype(current_image.dtype)
                    grey_image = np.repeat(grey_image, current_image.shape[-1], axis=-1)
                    output_image = grey_image.copy()

                    zoomed_area = current_image[row_start:row_end, col_start:col_end]
                    upscale_factor = 6
                    zoomed_area_upscaled = np.kron(zoomed_area, np.ones((upscale_factor, upscale_factor, 1)))

                    center_row, center_col = start_row, start_col
                    row_start_upscaled = max(center_row - zoomed_area_upscaled.shape[0] // 2, 0)
                    row_end_upscaled = min(center_row + zoomed_area_upscaled.shape[0] // 2, output_image.shape[0])
                    col_start_upscaled = max(center_col - zoomed_area_upscaled.shape[1] // 2, 0)
                    col_end_upscaled = min(center_col + zoomed_area_upscaled.shape[1] // 2, output_image.shape[1])

                    row_start_zoomed = max(0, -row_start_upscaled)
                    row_end_zoomed = row_start_zoomed + (row_end_upscaled - row_start_upscaled)
                    col_start_zoomed = max(0, -col_start_upscaled)
                    col_end_zoomed = col_start_zoomed + (col_end_upscaled - col_start_upscaled)

                    row_end_zoomed = min(row_end_zoomed, zoomed_area_upscaled.shape[0])
                    col_end_zoomed = min(col_end_zoomed, zoomed_area_upscaled.shape[1])

                    output_image[row_start_upscaled:row_end_upscaled, col_start_upscaled:col_end_upscaled] = \
                        zoomed_area_upscaled[row_start_zoomed:row_end_zoomed, col_start_zoomed:col_end_zoomed]

                    return output_image, True

            image_a.select(handle_zoom, [image_a, zoomed_state_a, state_segmented_a], [image_a, zoomed_state_a])
            image_b.select(handle_zoom, [image_b, zoomed_state_b, state_segmented_b], [image_b, zoomed_state_b])
               
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
                    return [[vote.id, vote.image_id, vote.model_a, vote.model_b, vote.winner, vote.user_id, vote.timestamp] for vote in votes]

                vote_table = gr.Dataframe(
                    headers=["ID", "Image ID", "Model A", "Model B", "Winner", "user_id", "Timestamp"],
                    value=update_vote_data(),
                    label="Vote Data",
                    column_widths=[20, 150, 100, 100, 100, 100, 150],
                    row_count=0
                )

                vote_data_tab.select(
                    fn=lambda: update_vote_data(),
                    outputs=vote_table
                )

            with gr.Tab("👥 Hall of Fame", id=3) as user_leaderboard_tab:
                current_time = datetime.now()
                start_of_week = current_time - timedelta(days=current_time.weekday())

                def get_weekly_user_leaderboard():
                    """Get the leaderboard of users with the most votes in the current week, excluding anonymous votes."""
                    votes = get_all_votes()
                    weekly_votes = [
                        vote for vote in votes 
                        if vote.timestamp >= start_of_week 
                        and vote.user_id 
                        and vote.user_id != "anonymous"
                    ]
                    user_vote_count = {}
                    for vote in weekly_votes:
                        user_vote_count[vote.user_id] = user_vote_count.get(vote.user_id, 0) + 1
                    sorted_users = sorted(user_vote_count.items(), key=lambda x: x[1], reverse=True)
                    
                    # Add medals for the top 3 users
                    medals = ["🥇", "🥈", "🥉"]
                    leaderboard = []
                    for index, (user, count) in enumerate(sorted_users):
                        medal = medals[index] if index < len(medals) else ""
                        leaderboard.append([f"{medal} {user}", count])
                    
                    return leaderboard

                user_leaderboard_table = gr.Dataframe(
                    headers=["User", "Votes"],
                    value=get_weekly_user_leaderboard(),
                    label="User Vote Leaderboard (This Week)",
                    column_widths=[150, 100],
                    row_count=0
                )

                leaderboard_info = gr.Markdown(
                    value=f"""
                    This leaderboard shows the ranking of users based on the number of votes they have cast in the current week. The current ranking is based on votes cast from {start_of_week.strftime('%Y-%m-%d')} to {current_time.strftime('%Y-%m-%d')}.
                    It will be updated each week. 
                    """
                )

                user_leaderboard_tab.select(
                    fn=lambda: get_weekly_user_leaderboard(),
                    outputs=user_leaderboard_table
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
    with commit_scheduler.lock:
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