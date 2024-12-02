import os
import logging
from typing import Tuple
from dotenv import load_dotenv
import gradio as gr
import numpy as np
from PIL import Image
import random
import time
from db import compute_elo_scores, get_all_votes

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Access the API key
PHOTOROOM_API_KEY = os.getenv('PHOTOROOM_API_KEY')
CLIPDROP_API_KEY = os.getenv('CLIPDROP_API_KEY')


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
    image_paths = load_images_from_directory("data/web-original-images")
    last_image_path = None
    max_attempts = 10
    
    random.seed(time.time())

    for _ in range(max_attempts):
        available_images = [path for path in image_paths if path != last_image_path]
        
        if not available_images:
            logging.error("No available images to select from.")
            return None

        random_image_path = random.choice(available_images)
        input_image = Image.open(random_image_path)
        
        image_filename = os.path.splitext(os.path.basename(random_image_path))[0] + ".png"
        segmented_image_paths = {
            "Photoroom": os.path.join("data/resized/photoroom", image_filename),
            #"Clipdrop": os.path.join("data/processed/clipdrop", image_filename),
            "RemoveBG": os.path.join("data/resized/removebg", image_filename),
            "BRIA RMBG 2.0": os.path.join("data/resized/bria", image_filename)
        }
        
        try:
            selected_models = random.sample(list(segmented_image_paths.keys()), 2)
            model_a_name, model_b_name = selected_models
            model_a_output_path = segmented_image_paths[model_a_name]
            model_b_output_path = segmented_image_paths[model_b_name]
            model_a_output_image = Image.open(model_a_output_path)
            model_b_output_image = Image.open(model_b_output_path)
            return (random_image_path, input_image, model_a_output_path, model_a_output_image,
                    model_b_output_path, model_b_output_image, model_a_name, model_b_name)
        except FileNotFoundError as e:
            logging.error("File not found: %s. Resampling another image.", e)
            last_image_path = random_image_path

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
    """Compute the absolute difference between two image masks."""
    mask_a = np.asarray(segmented_a)
    mask_b = np.asarray(segmented_b)

    # Create a binary mask where non-transparent pixels are marked as 1
    mask_a_1d = np.where(mask_a[..., 3] != 0, 1, 0)
    mask_b_1d = np.where(mask_b[..., 3] != 0, 1, 0)

    # Compute the absolute difference between the masks
    return np.abs(mask_a_1d - mask_b_1d)

def gradio_interface():
    """Create and return the Gradio interface."""
    with gr.Blocks() as demo:
        gr.Markdown("# Background Removal Arena")

        with gr.Tabs() as tabs:
            with gr.Tab("⚔️ Arena (battle)", id=0):
                notice_markdown = gr.Markdown(get_notice_markdown(), elem_id="notice_markdown")

                (fpath_input, input_image, fpath_a, segmented_a, fpath_b, segmented_b,
                 a_name, b_name) = select_new_image()
                model_a_name = gr.State(a_name)
                model_b_name = gr.State(b_name)
                fpath_input = gr.State(fpath_input)
                fpath_a = gr.State(fpath_a)
                fpath_b = gr.State(fpath_b)

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
                        value=(input_image, [(mask_difference > 0, "Difference Mask")]),
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
                    fn=lambda: vote_for_model("model_a", (fpath_input, fpath_a, fpath_b), model_a_name, model_b_name),
                    outputs=[
                        fpath_input, input_image_display, fpath_a, image_a_display, fpath_b, image_b_display, model_a_name, model_b_name, notice_markdown
                    ]
                )
                vote_b_btn.click(
                    fn=lambda: vote_for_model("model_b", (fpath_input, fpath_a, fpath_b), model_a_name, model_b_name),
                    outputs=[
                        fpath_input, input_image_display, fpath_a, image_a_display, fpath_b, image_b_display, model_a_name, model_b_name, notice_markdown
                    ]
                )
                vote_tie.click(
                    fn=lambda: vote_for_model("tie", (fpath_input, fpath_a, fpath_b), model_a_name, model_b_name),
                    outputs=[
                        fpath_input, input_image_display, fpath_a, image_a_display, fpath_b, image_b_display, model_a_name, model_b_name, notice_markdown
                    ]
                )
            
                def vote_for_model(choice, fpaths, model_a_name, model_b_name):
                    """Submit a vote for a model and return updated images and model names."""
                    logging.info("Voting for model: %s", choice)

                    vote_data = {
                        "image_id": fpaths[0].value,
                        "model_a": model_a_name.value,
                        "model_b": model_b_name.value,
                        "winner": choice,
                        "fpath_a": fpaths[1].value,  
                        "fpath_b": fpaths[2].value, 
                    }

                    try:
                        logging.debug("Adding vote data to the database: %s", vote_data)
                        from db import add_vote
                        result = add_vote(vote_data)
                        logging.info("Vote successfully recorded in the database with ID: %s", result["id"])
                    except Exception as e:
                        logging.error("Error recording vote: %s", str(e))

                    (new_fpath_input, new_input_image, new_fpath_a, new_segmented_a,
                     new_fpath_b, new_segmented_b, new_a_name, new_b_name) = select_new_image()
                    model_a_name.value = new_a_name
                    model_b_name.value = new_b_name
                    fpath_input.value = new_fpath_input
                    fpath_a.value = new_fpath_a
                    fpath_b.value = new_fpath_b

                    mask_difference = compute_mask_difference(new_segmented_a, new_segmented_b)

                    # Update the notice markdown with the new vote count
                    new_notice_markdown = get_notice_markdown()

                    return (fpath_input.value, (new_input_image, [(mask_difference, "Mask")]), fpath_a.value, new_segmented_a,
                            fpath_b.value, new_segmented_b, model_a_name.value, model_b_name.value, new_notice_markdown)
           
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

def load_images_from_directory(directory):
    """Load and return image paths from a directory."""
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return [os.path.join(directory, f) for f in image_files]

if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()