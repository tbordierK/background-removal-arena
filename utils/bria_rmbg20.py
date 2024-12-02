import os
import requests
import concurrent.futures
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
API_TOKEN = os.getenv('BRIA_API_TOKEN')

def process_image(input_image_path, output_image_path):
    try:
        url = "https://engine.prod.bria-api.com/v1/background/remove"

        # Prepare the file payload
        files = [
            ('file', (os.path.basename(input_image_path), open(input_image_path, 'rb'), 'image/jpeg'))
        ]

        headers = {
            'api_token': API_TOKEN
        }

        # Make the POST request to the Bria API
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()

        # Get the result from the response
        data = response.json()
        
        # Extract the result URL from the response
        processed_image_url = data.get('result_url')  

        # Download the processed image
        if processed_image_url:
            image_response = requests.get(processed_image_url)
            image_response.raise_for_status()  # Ensure the request was successful

            with open(output_image_path, 'wb') as f:
                f.write(image_response.content)
                print(f"Image processed and saved to {output_image_path}")

    except requests.RequestException as e:
        print(f"Error: {str(e)} ({input_image_path})")
        return str(e)

def iterate_over_directory(directory_path, result_directory):
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.heic')):
                    file_path = os.path.join(root, file)

                    result_file_name = os.path.splitext(os.path.basename(file_path))[0] + '.png'
                    result_file_directory = os.path.join(result_directory, os.path.basename(root))

                    if not os.path.exists(result_file_directory):
                        os.makedirs(result_file_directory)

                    result_path = os.path.join(result_file_directory, result_file_name)

                    if not os.path.exists(result_path): # don't re-process images 
                        executor.submit(process_image, file_path, result_path)

if __name__ == "__main__":
    INPUT_DIRECTORY = "../original-images/"
    OUTPUT_DIRECTORY = "../result-bria-rmbg20/"

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    iterate_over_directory(directory_path=INPUT_DIRECTORY, result_directory=OUTPUT_DIRECTORY)