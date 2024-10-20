import argparse
import os
import base64
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_symmetry(client, image_paths):
    base64_images = [encode_image(path) for path in image_paths]

    sys_prompt = "Mention if the image is symmetric. I am giving you two views of the same object: a front-facing view and a right-facing view, with some axes and angles marked. Tell if the object is symmetric or not (Symmetry means that the image is EXACTLY SAME ON BOTH SIDE OF AXES). If it is symmetric, return the axis of symmetry (along an angle if any) with respect to the original front-facing image."
    
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
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

def process_images(input_folder, output_folder, api_key):
    client = OpenAI(api_key=api_key)
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if len(image_files) != 2:
        print(f"Error: Expected 2 images in the input folder, but found {len(image_files)}.")
        return

    image_paths = [os.path.join(input_folder, image_file) for image_file in image_files]
    result = analyze_symmetry(client, image_paths)
    output_file = os.path.join(output_folder, "symmetry_analysis_result.txt")
    with open(output_file, 'w') as f:
        f.write(result)
    
    print(f"Processed: {image_files[0]} and {image_files[1]}")
    print(f"Result: {result}")
    print(f"Result saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze image symmetry using OpenAI API")
    parser.add_argument("--input_folder", required=True, help="Path to the folder containing input images (front and right views)")
    parser.add_argument("--output_folder", required=True, help="Path to the folder where result will be saved")
    parser.add_argument("--api_key", required=True, help="OpenAI API key")
    args = parser.parse_args()

    process_images(args.input_folder, args.output_folder, args.api_key)

if __name__ == "__main__":
    main()