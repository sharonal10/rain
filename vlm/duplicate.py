import argparse
import os
import base64
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

parser = argparse.ArgumentParser(description="Process images and interact with OpenAI API.")
parser.add_argument("--api_key", required=True, help="OpenAI API key.")
parser.add_argument("--image_dir", required=True, help="Directory containing images with SAM masks.")
parser.add_argument("--base_image_path", required=True, help="Path to the base image without masks.")
args = parser.parse_args()

client = OpenAI(api_key=args.api_key)

image_files = sorted([f for f in os.listdir(args.image_dir) if f.endswith(".png") and int(f.split('.')[0]) % 30 == 1])
image_paths = [os.path.join(args.image_dir, f) for f in image_files]

image_paths.insert(0, args.base_image_path)

base64_images = [encode_image(path) for path in image_paths]

sys_prompt = """I am providing multiple colored frame images of a 3D object along with an image of the object itself.
Each segment of the object is marked by a unique color.
Your task is to identify all groups of segments with identical shapes or structures, regardless of color, and return each group of duplicates as a separate entry.

1. Return each group of duplicate colors separately in the format: duplicate(color1, color2, ...). Do this for every distinct group of identical segments, without omitting any duplicates.
2. Include all colors in each duplicate group, even if more than two segments share the same structure.
3. Do not limit the number of groups or colors in each group based on examples. The model should capture every possible duplicate accurately.
4. If no duplicates exist, return: None.

For example:
If three identical chair legs are colored red, green, and blue, and two identical arms are colored orange and yellow, your output should be: duplicate(red, green, blue), duplicate(orange, yellow).
Ensure that every duplicate group is identified and reported in full."""

messages = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": []}
]

for base64_image in base64_images:
    messages[1]["content"].append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_image}"
        }
    })

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300
    )
    print("Assistant: " + response.choices[0].message.content)
except Exception as e:
    print(f"An error occurred: {str(e)}")
