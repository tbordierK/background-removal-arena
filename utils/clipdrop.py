import os
import requests

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv('CLIPDROP_API_KEY')

def process_image(input_image_path, output_image_path):
    try:
        url = "https://clipdrop-api.co/remove-background/v1"

        with open(input_image_path, 'rb') as image_file:
            files = { "image_file": image_file }

            payload = {
                "transparency_handling": "discard_alpha_layer"
            }
            headers = {
                "Accept": "image/png",
                "x-api-key": API_KEY
            }

            response = requests.post(url, data=payload, files=files, headers=headers)
            response.raise_for_status()

            with open(output_image_path, 'wb') as f:
                f.write(response.content)
                print(f"Image downloaded and saved to {output_image_path}")

    except requests.RequestException as e:
        print(f"Error: {str(e)} ({input_image_path})")
        return str(e)
    
def iterate_over_directory(directory_path, result_directory):
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
                    process_image(file_path, result_path)    

if __name__ == "__main__":
    INPUT_DIRECTORY = "../original-images/"
    OUTPUT_DIRECTORY = "../result-clipdrop/"

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    iterate_over_directory(directory_path=INPUT_DIRECTORY, result_directory=OUTPUT_DIRECTORY)