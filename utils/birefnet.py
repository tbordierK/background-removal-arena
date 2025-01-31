import os
import logging
import asyncio
import requests
import fal_client
import json
from typing import Optional

# Configure logging to show more detailed information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

async def remove_background_birefnet(image_path: str) -> Optional[str]:
    """Remove background using BiRefNet API asynchronously."""
    logging.info(f"Starting BiRefNet processing for: {image_path}")
    try:
        # Submit the request
        logging.info("Submitting request to BiRefNet API...")
        handler = await fal_client.submit_async(
            "fal-ai/birefnet/v2",
            arguments={
                "image_url": image_path,
                "model": "General Use (Heavy)",
                "operating_resolution": "1024x1024",
                "output_format": "png",
                "refine_foreground": True
            }
        )
        request_id = handler.request_id
        logging.info(f"🔄 Request submitted with ID: {request_id}")

        # Poll for status with logs
        while True:
            status = await fal_client.status_async("fal-ai/birefnet/v2", request_id, with_logs=True)
            
            # Handle logs if available
            if hasattr(status, 'logs') and status.logs:
                for log in status.logs:
                    level = log.get('level', 'INFO')
                    message = log.get('message', '')
                    logging.info(f"🔄 BiRefNet {level}: {message}")
            
            # Check status based on object type
            if isinstance(status, fal_client.Queued):
                logging.info(f"⏳ Request in queue")
            elif isinstance(status, fal_client.InProgress):
                logging.info("🔄 Request is being processed...")
            elif isinstance(status, fal_client.Completed):
                logging.info("✅ Request completed")
                break
            elif isinstance(status, fal_client.Failed):
                logging.error(f"❌ Request failed: {status.error}")
                return None
            else:
                logging.error(f"❌ Unknown status type: {type(status)}")
                return None
            
            await asyncio.sleep(1)  # Wait before checking again

        # Get the result
        result = await fal_client.result_async("fal-ai/birefnet/v2", request_id)
        
        if not result or not isinstance(result, dict):
            logging.error("❌ Invalid result from BiRefNet")
            return None

        image_data = result.get('image', {})
        if not image_data or not isinstance(image_data, dict):
            logging.error(f"❌ Missing or invalid image data in result: {result}")
            return None

        image_url = image_data.get('url')
        if not image_url:
            logging.error(f"❌ Missing image URL in result: {image_data}")
            return None

        # Log successful result with image details
        logging.info(f"✅ Got image: {image_data.get('width')}x{image_data.get('height')} "
                    f"({image_data.get('file_size', 0) / 1024 / 1024:.1f}MB)")
        return image_url

    except Exception as e:
        logging.error(f"❌ Unexpected error using BiRefNet API: {str(e)}", exc_info=True)
        return None

async def process_single_image(input_path: str, output_path: str) -> bool:
    """Process a single image asynchronously."""
    try:
        # Upload the file
        logging.info(f"📤 Uploading to temporary storage...")
        image_url = await fal_client.upload_file_async(input_path)
        logging.info(f"✅ Upload successful: {image_url}")
        
        # Process with BiRefNet
        result_url = await remove_background_birefnet(image_url)
        
        if result_url:
            # Download the result
            logging.info(f"📥 Downloading result...")
            response = requests.get(result_url)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type:
                logging.error(f"❌ Invalid content type: {content_type}")
                return False
                
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"✅ Successfully saved to {output_path}")
            return True
            
        return False
    except Exception as e:
        logging.error(f"❌ Error processing image: {str(e)}", exc_info=True)
        return False

async def iterate_over_directory(input_dir: str, output_dir: str):
    """Process all images in a directory using BiRefNet with async processing."""
    logging.info(f"🚀 Starting BiRefNet processing for directory: {input_dir}")
    logging.info(f"📁 Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of files to process
    files = [f for f in os.listdir(input_dir) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    total_files = len(files)
    
    processed = 0
    skipped = 0
    failed = 0
    
    logging.info(f"📊 Found {total_files} images to process")

    # Process files in batches to control concurrency
    batch_size = 3  # Reduced batch size to avoid overwhelming the API
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        tasks = []
        
        for filename in batch:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            logging.info(f"\n{'='*50}")
            logging.info(f"Processing [{i + len(tasks) + 1}/{total_files}]: {filename}")
            
            if os.path.exists(output_path):
                logging.info(f"⏭️ Skipping {filename} - already processed")
                skipped += 1
                continue
            
            tasks.append(process_single_image(input_path, output_path))
        
        if tasks:  # Only process if we have tasks
            # Wait for batch to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for filename, result in zip(batch, results):
                if isinstance(result, Exception):
                    logging.error(f"❌ Failed to process {filename}: {str(result)}")
                    failed += 1
                elif result:
                    processed += 1
                else:
                    failed += 1
            
            # Add a small delay between batches
            await asyncio.sleep(1)

    logging.info(f"\n{'='*50}")
    logging.info(f"📊 Processing Summary:")
    logging.info(f"✅ Successfully processed: {processed}")
    logging.info(f"⏭️ Skipped (already existed): {skipped}")
    logging.info(f"❌ Failed: {failed}")
    logging.info(f"📁 Total files: {total_files}")

def process_directory(input_dir: str, output_dir: str):
    """Synchronous wrapper for iterate_over_directory."""
    asyncio.run(iterate_over_directory(input_dir, output_dir)) 