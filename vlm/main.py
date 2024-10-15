from openai import OpenAI
from PIL import Image
import io
import argparse
import base64

parser = argparse.ArgumentParser()
parser.add_argument('--api_key', type=str, required=True)
args = parser.parse_args()

from openai import OpenAI

client = OpenAI(
  api_key=args.api_key,
)

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    img_type = Image.open(image_path).format.lower()
    return f"data:image/{img_type};base64,{encoded_string}"

def ask_gpt4_with_images(question, image_path_1, image_path_2):
    img1_b64 = image_to_base64(image_path_1)
    img2_b64 = image_to_base64(image_path_2)

    prompt = f"Here are two images: {image_path_1} and {image_path_2}. {question}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": img1_b64},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": img2_b64},
                    },
                ],
            }
        ],
    )
    import pdb; pdb.set_trace()
    return response.choices[0].message.content

image_path_1 = "vlm/testdata/dresser.jpg"
image_path_2 = "vlm/testdata/masks.jpg"
question = f"{image_path_1} is a photo of an object, while {image_path_2} visualises its SAM masks. Which colors correspond to which parts?"

answer = ask_gpt4_with_images(question, image_path_1, image_path_2)

print("GPT-4's answer:", answer)
