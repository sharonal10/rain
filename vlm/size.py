import argparse
from openai import OpenAI
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    parser = argparse.ArgumentParser(description="Analyze image size using GPT 4")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    parser.add_argument("--image_path", required=True, help="Path to the image file")
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)

    base64_image = encode_image(args.image_path)

    sys_prompt = "Mention any objects in the image that are equal in size some of the example objects are given to you (not all). JUST MENTION THE OBJECT NAMES THAT ARE EQUAL IN SIZE AND NOTHING ELSE (For example for a table dresser with equal size of drawers and wheels, the output should look like, Assistant: Wheel,Drawer"
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300
        )
        print("Assistant: " + response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()