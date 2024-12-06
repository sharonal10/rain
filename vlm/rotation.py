from openai import OpenAI
from PIL import Image
import io
import argparse
import base64

parser = argparse.ArgumentParser()
parser.add_argument('--api_key', type=str, required=True)
parser.add_argument('--input_image', type=str, required=True)
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

def ask_gpt4_with_images(question, image_path_1, image_path_2, image_path_3, image_path_4, image_path_5):
    img1_b64 = image_to_base64(image_path_1)
    img2_b64 = image_to_base64(image_path_2)
    img3_b64 = image_to_base64(image_path_3)
    img4_b64 = image_to_base64(image_path_4)
    img5_b64 = image_to_base64(image_path_5)

    prompt = f"Here are five images: {image_path_1}, {image_path_2}, {image_path_3}, {image_path_4}, {image_path_5}. {question}"

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
                    {
                        "type": "image_url",
                        "image_url": {"url": img3_b64},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": img4_b64},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": img5_b64},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content

image_path_1 = "vlm/for_vlm_arrows/0001.png"
image_path_2 = "vlm/for_vlm_arrows/0031.png"
image_path_3 = "vlm/for_vlm_arrows/0061.png"
image_path_4 = "vlm/for_vlm_arrows/0091.png"
image_path_5 = args.input_image
question = f"""{image_path_1} is a chair facing right. {image_path_2} is a chair facing forwards. {image_path_3} is a chair facing the left. {image_path_4} is a chair facing backwards.

The yellow arrow represents left, red represents backwards, green represents right, blue represents forwards.

Identify which arrow is pointing in the same direction that the chair in {image_path_5} is facing. Is the chair in {image_path_5} facing right, forwards, left, or backwards?"""

answer = ask_gpt4_with_images(question, image_path_1, image_path_2, image_path_3, image_path_4, image_path_5)

print(answer)
