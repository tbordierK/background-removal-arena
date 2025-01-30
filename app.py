import os
import json
import time
import random
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

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
    fill_database_once,
    compute_votes_per_model
)

# Load environment variables
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
google_analytics_tracking_id = os.getenv("GOOGLE_ANALYTICS_TRACKING_ID")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load datasets and initialize database
dataset = load_dataset("bgsys/background-removal-arena-green_v0_clothing_checkered", split='train')
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
        median_elo_scores, model_rating_q025, model_rating_q975, variance = compute_elo_scores()
        logging.info("Elo scores successfully computed.")
        return median_elo_scores, model_rating_q025, model_rating_q975, variance
    except Exception as e:
        logging.error("Error computing Elo scores: %s", str(e))
        return None

def update_rankings_table():
    """Update and return the rankings table based on Elo scores and vote counts."""
    median_elo_scores, model_rating_q025, model_rating_q975, variance = fetch_elo_scores() or {}
    model_vote_counts = compute_votes_per_model()
    try:
        # Create a list of models to iterate over
        models = ["Clipdrop", "Photoroom", "RemoveBG", "BRIA RMBG 2.0"]
        rankings = []

        for model in models:
            elo_score = int(median_elo_scores.get(model, 0))
            model_variance = int(variance.get(model, 0))
            ci_95 = f"{int(model_rating_q025.get(model, 0))} - {int(model_rating_q975.get(model, 0))}"
            vote_count = model_vote_counts.get(model, 0)
            rankings.append([model, elo_score, model_variance, ci_95, vote_count])

        # Sort rankings by Elo score in descending order
        rankings.sort(key=lambda x: x[1], reverse=True)
    except KeyError as e:
        logging.error("Missing score for model: %s", str(e))
        return []
    return rankings

def select_new_image(last_used_indices):
    """Select a new image and its segmented versions."""
    max_attempts = 10
    
    # Initialize empty deque if None
    if last_used_indices is None:
        last_used_indices = deque(maxlen=10)
    
    for _ in range(max_attempts):
        # Filter out recently used indices
        available_indices = [
            i for i in range(len(dataset)) 
            if i not in last_used_indices
        ]
        
        if not available_indices:
            logging.error("No available images to select from.")
            return None, last_used_indices

        random_index = random.choice(available_indices)
        sample = dataset[random_index]
        input_image = sample['original_image']

        segmented_images = [sample.get(key) for key in ['clipdrop_image', 'bria_image', 'photoroom_image', 'removebg_image']]
        segmented_sources = ['Clipdrop', 'BRIA RMBG 2.0', 'Photoroom', 'RemoveBG']
        
        if segmented_images.count(None) > 2:
            logging.error("Not enough segmented images found for: %s. Resampling another image.", sample['original_filename'])
            continue

        try:
            selected_indices = random.sample([i for i, img in enumerate(segmented_images) if img is not None], 2)
            model_a_index, model_b_index = selected_indices
            
            # Add the used index to our history
            last_used_indices.append(random_index)
            
            return (
                (sample['original_filename'], input_image,
                segmented_images[model_a_index], segmented_images[model_b_index],
                segmented_sources[model_a_index], segmented_sources[model_b_index]),
                last_used_indices
            )
        except Exception as e:
            logging.error("Error processing images: %s. Resampling another image.", str(e))

    logging.error("Failed to select a new image after %d attempts.", max_attempts)
    return None, last_used_indices

def get_notice_markdown():
    """Generate the notice markdown with dynamic vote count."""
    total_votes = len(get_all_votes())
    return f"""

    ## 📜 How It Works
    - **Blind Test**: You will see two images with their background removed from two anonymous background removal models (Clipdrop, RemoveBG, Photoroom, BRIA RMBG 2.0).
    - **Vote for the Best**: Choose the best result, if none stand out choose "Tie". 

    ## 📊 Stats
    - **Total #votes**: {total_votes}


    """

def compute_mask_difference(segmented_a, segmented_b):
    """Compute the difference between two images across all channels."""
    mask_a = np.asarray(segmented_a, dtype=np.float32) / 255.0
    mask_b = np.asarray(segmented_b, dtype=np.float32) / 255.0
    
    # Compute absolute difference across all channels (RGBA)
    difference = np.abs(mask_a - mask_b)
    
    # Take the maximum difference across channels at each pixel
    max_diff = np.max(difference, axis=-1)
    
    # Apply non-linear transformation to enhance visibility of smaller differences
    # Using square root to compress the range while maintaining continuity
    # Multiply by 3 to make it more visible
    return np.sqrt(max_diff) * 3

js = r"""
function load_zoom() {
    setTimeout(function() {

    // Select all images from the three displayed image containers.
    const images = document.querySelectorAll('.image-container img');

    // Set transform origin so scaling and translating feels "natural".
    images.forEach(img => {
        img.style.transformOrigin = 'top left';
        img.style.transition = 'transform 0.1s ease-out';
        img.style.cursor = 'zoom-in';
    });

    // Choose a scale factor
    const scale = 5;

    function handleMouseMove(e) {
        const rect = e.currentTarget.getBoundingClientRect();
        const xPercent = (e.clientX - rect.left) / rect.width;
        const yPercent = (e.clientY - rect.top) / rect.height;
        const offsetX = xPercent * (scale - 1) * 100;
        const offsetY = yPercent * (scale - 1) * 100;

        images.forEach(img => {
            img.style.transform = `translate(-${offsetX}%, -${offsetY}%) scale(${scale})`;
        });
    }

    function handleMouseEnter(e) {
        e.currentTarget.addEventListener('mousemove', handleMouseMove);
    }

    function handleMouseLeave(e) {
        e.currentTarget.removeEventListener('mousemove', handleMouseMove);
        images.forEach(img => {
            img.style.transform = 'translate(0,0) scale(1)';
        });
    }

    const containers = document.querySelectorAll('.image-container');

    containers.forEach(container => {
        container.addEventListener('mouseenter', handleMouseEnter);
        container.addEventListener('mouseleave', handleMouseLeave);
    });
}, 1000); // 1 second timeout
}
"""

head = """

<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-7EKCV97SG2"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', '""" + f"{google_analytics_tracking_id}" + """');
</script>
"""

def get_default_username(profile: gr.OAuthProfile | None) -> str:
    """
    Returns the username if the user is logged in, or an empty string if not logged in.
    """
    if profile is None:
        return ""
    return profile.username

def gradio_interface():
    """Create and return the Gradio interface."""
    with gr.Blocks(js=js, head=head, fill_width=True) as demo:
        # Initialize session state for last used indices
        last_used_indices_state = gr.State()
        
        button_name = "Difference between masks"

        with gr.Tabs() as tabs:
            with gr.Row(equal_height=True):
                    def on_enter_contest(username):
                        feedback_message = f"Thank you, {username or 'anonymous'}! You can see how you rank in the Hall of Fame."
                        logging.info(feedback_message)
                        return feedback_message

                    with gr.Column(scale=1):
                        username_input = gr.Textbox(
                            label="Enter your username (optional)",
                            placeholder="✨ Enter your username (optional)",
                            show_label=False,
                            submit_btn="Enter",
                            interactive=True
                        )
                    demo.load(fn=get_default_username, inputs=None, outputs=username_input)


                    with gr.Column(scale=3):
                        feedback_output = gr.Textbox(
                            label="Feedback",
                            interactive=False,
                            show_label=False
                        )

                    with gr.Column(scale=1):
                         gr.LoginButton()
                    

                    username_input.submit(
                        fn=on_enter_contest,
                        inputs=username_input,
                        outputs=feedback_output
                    )
                    
            with gr.Tab("⚔️ Arena (battle)", id=0):
                image_width = None
                image_height = 600  # Limit image height to fit on a standard screen
              

                with gr.Row():
                    # Initialize components with empty states
                    state_filename = gr.State("")
                    state_model_a_name = gr.State("")
                    state_model_b_name = gr.State("")
                    image_a = gr.Image(label="Image A", width=image_width, height=image_height)
                    input_image_display = gr.AnnotatedImage(label="Input Image", width=image_width, height=image_height)
                    image_b = gr.Image(label="Image B", width=image_width, height=image_height)

                    def refresh_states(state_filename, state_model_a_name, state_model_b_name, last_used_indices):
                        # Call select_new_image to get new image data
                        result, new_last_used_indices = select_new_image(last_used_indices)
                        if result is None:
                            return [state_filename, image_a, image_b, state_model_a_name, state_model_b_name, 
                                   input_image_display, new_last_used_indices]
                            
                        filename, input_image, segmented_a, segmented_b, model_a_name, model_b_name = result
                        mask_difference = compute_mask_difference(segmented_a, segmented_b)
                        
                        # Update states with new data
                        state_filename.value = filename
                        state_model_a_name.value = model_a_name
                        state_model_b_name.value = model_b_name

                        # Create new gr.Image components with updated values
                        image_a = gr.Image(value=segmented_a, label="Image A", width=image_width, height=image_height)
                        image_b = gr.Image(value=segmented_b, label="Image B", width=image_width, height=image_height)
                        input_image_display = gr.AnnotatedImage(
                            value=(input_image, [(mask_difference, button_name)]), 
                            width=image_width,
                            height=image_height
                        )
                        
                        return [
                            state_filename, image_a, image_b, state_model_a_name, state_model_b_name, 
                            input_image_display, new_last_used_indices
                        ]

                    
                with gr.Row():
                    vote_a_button = gr.Button("👈  A is better")
                    vote_tie_button = gr.Button("🤝  Tie")
                    vote_b_button = gr.Button("👉  B is better")

                def vote_for_model(choice, original_filename, model_a_name, model_b_name, user_username, last_used_indices):
                    """Submit a vote for a model and return updated images and model names."""
                  

                    if not original_filename.value:
                        logging.error("The field 'original_filename' is empty or None.")
                        raise ValueError("The field 'original_filename' must be provided and non-empty.")
                    if not model_a_name.value:
                        logging.error("The field 'model_a_name' is empty or None.")
                        raise ValueError("The field 'model_a_name' must be provided and non-empty.")
                    if not model_b_name.value:
                        logging.error("The field 'model_b_name' is empty or None.")
                        raise ValueError("The field 'model_b_name' must be provided and non-empty.")
                    if not choice:
                        logging.error("The field 'choice' is empty or None.")
                        raise ValueError("The field 'choice' must be provided and non-empty.")

                    vote_data = {
                        "image_id": original_filename.value,
                        "model_a": model_a_name.value,
                        "model_b": model_b_name.value,
                        "winner": choice,
                        "user_id": user_username or "anonymous"
                    }
                    logging.debug(vote_data)

                     # Create a gr.Info message with model names and the user's choice
                    voted_model = vote_data[vote_data["winner"]] if vote_data["winner"] in ["model_a", "model_b"] else "Tie"
                    voted_model_emoji = "👈" if choice == "model_a" else "👉" if choice == "model_b" else "🤝"
                    voted_model_color = "green" if choice == "model_a" else "blue" if choice == "model_b" else "gray"
                    info_message = (
                        f"<p>You voted for <strong style='color:{voted_model_color};'>{voted_model_emoji} {voted_model}</strong>.</p>"
                        f"<p><span style='color:green;'>👈 {model_a_name.value}</span> - "
                        f"<span style='color:blue;'>👉 {model_b_name.value}</span></p>"
                    )
                    gr.Info(info_message)

                    try:
                        logging.debug("Adding vote data to the database: %s", vote_data)
                        result = add_vote(vote_data)
                        logging.info("Vote successfully recorded in the database with ID: %s", result["id"])
                    except Exception as e:
                        logging.error("Error recording vote: %s", str(e))

                    outputs = refresh_states(state_filename, state_model_a_name, state_model_b_name, last_used_indices)
                    new_notice_markdown = get_notice_markdown()

                    return outputs + [new_notice_markdown]
                            
                notice_markdown = gr.Markdown(get_notice_markdown(), elem_id="notice_markdown")
                vote_a_button.click(
                    fn=lambda username, last_used_indices: vote_for_model(
                        "model_a", state_filename, state_model_a_name, state_model_b_name, username, last_used_indices
                    ),
                    inputs=[username_input, last_used_indices_state],
                    outputs=[
                        state_filename, image_a, image_b, state_model_a_name, state_model_b_name, 
                        input_image_display, last_used_indices_state, notice_markdown
                    ]
                )
                vote_b_button.click(
                    fn=lambda username, last_used_indices: vote_for_model(
                        "model_b", state_filename, state_model_a_name, state_model_b_name, username, last_used_indices
                    ),
                    inputs=[username_input, last_used_indices_state],
                    outputs=[
                        state_filename, image_a, image_b, state_model_a_name, state_model_b_name, 
                        input_image_display, last_used_indices_state, notice_markdown
                    ]
                )
                vote_tie_button.click(
                    fn=lambda username, last_used_indices: vote_for_model(
                        "tie", state_filename, state_model_a_name, state_model_b_name, username, last_used_indices
                    ),
                    inputs=[username_input, last_used_indices_state],
                    outputs=[
                        state_filename, image_a, image_b, state_model_a_name, state_model_b_name, 
                        input_image_display, last_used_indices_state, notice_markdown
                    ]
                )
            
               

            with gr.Tab("🏆 Leaderboard", id=1) as leaderboard_tab:
                rankings_table = gr.Dataframe(
                    headers=["Model", "Elo score", "Variance", "95% CI", "Selections"],
                    value=update_rankings_table(),
                    label="Current Model Rankings",
                    column_widths=[180, 60, 60, 60, 60],
                    row_count=4
                )

                leaderboard_tab.select(
                    fn=lambda: update_rankings_table(),
                    outputs=rankings_table
                )

                # Explanation of Bootstrapped Elo Score
                explanation_text = """
                The Elo score was calculated using bootstrapping with num_rounds=1000. This method provides a 
                distribution of Elo scores by repeatedly sampling the data, which helps in 
                understanding the variability and confidence in the model's ranking.

                We used the approach from the Chatbot Arena [rating system code](https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/monitor/rating_systems.py#L153).
                """
                gr.Markdown(explanation_text)

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
                        if vote.user_id and vote.user_id != "anonymous"
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
                    This leaderboard shows the ranking of users based on the number of votes they have cast since the start of the arena.
                    """
                )

                user_leaderboard_tab.select(
                    fn=lambda: get_weekly_user_leaderboard(),
                    outputs=user_leaderboard_table
                )
        demo.load(
            lambda: refresh_states(state_filename, state_model_a_name, state_model_b_name, None), 
            inputs=None, 
            outputs=[
                state_filename, image_a, image_b, state_model_a_name, state_model_b_name, 
                input_image_display, last_used_indices_state
            ]
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
