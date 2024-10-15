import openai
from PIL import Image
import io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--api_key', type=str, required=True)
args = parser.parse_args()

openai.api_key = args.api_key

def ask_gpt4_with_images(question, image_path_1, image_path_2):
    img1 = Image.open(image_path_1)
    img1_bytes = io.BytesIO()
    img1.save(img1_bytes, format=img1.format)

    img2 = Image.open(image_path_2)
    img2_bytes = io.BytesIO()
    img2.save(img2_bytes, format=img2.format)

    prompt = f"Here are two images: {image_path_1} and {image_path_2}. {question}"

    response = openai.ChatCompletion.create(
        model="gpt-4-vision", 
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        files=[
            {"name": "image1.png", "data": img1_bytes.getvalue()},
            {"name": "image2.png", "data": img2_bytes.getvalue()},
        ]
    )
    
    answer = response['choices'][0]['message']['content']
    return answer

image_path_1 = "path_to_your_first_image.png"
image_path_2 = "path_to_your_second_image.png"
question = f"{image_path_1} is a photo of an object, while {image_path_2} visualises its SAM masks. Which colors correspond to which parts?"

answer = ask_gpt4_with_images(question, image_path_1, image_path_2)

print("GPT-4's answer:", answer)
