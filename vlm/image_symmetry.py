from openai import OpenAI
import os
import base64
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process images with OpenAI API')
    parser.add_argument('--image_dir', type=str, required=True,
                      help='Directory containing the input images')
    parser.add_argument('--base_image', type=str, required=True,
                      help='Path to the base image with annotated axes')
    parser.add_argument('--api_key', type=str, required=True,
                      help='OpenAI API key')
    return parser.parse_args()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    args = parse_arguments()
    client = OpenAI(api_key=args.api_key)
    image_files = sorted([f for f in os.listdir(args.image_dir) 
                         if f.endswith(".png") and int(f.split('.')[0]) % 30 == 1])
    image_paths = [os.path.join(args.image_dir, f) for f in image_files]
    
    # Insert base image at the beginning
    image_paths.insert(0, args.base_image)
    base64_images = [encode_image(path) for path in image_paths]

    sys_prompt = """I am providing segmented, front-facing images of a 3D object, where each part of the object is marked by a unique color.

Your task is to identify all pairs or groups of symmetric segments based on their arrangement and alignment. A pair or group is symmetric if its members are mirrored across either the y-axis, the x-axis, or both.

For each axis, identify symmetric segments in the format:

y-axis symmetry: parallel_y(colorA, colorB, ...)
x-axis symmetry: parallel_x(colorA, colorB, ...)
Important Instructions:
Each symmetric group should be distinct. Separate each unique pair or group even if colors differ, listing them as separate entries.
If two parts are symmetric across an axis but do not match other pairs, they should be returned independently in their own parallel_x(...) or parallel_y(...) entry.
Include all possible symmetric configurations without merging or skipping unique pairs.
Example Output:

If analyzing an object with 4 drawers:
For symmetry along the x-axis:
If the top drawer is colored red and the bottom drawer is yellow, return: parallel_x(red, yellow)
If the middles drawers are blue and purple, return: parallel_x(blue, purple)
The correct output would be: parallel_x(blue, purple), parallel_x(red, yellow)

If analyzing a chair:

For symmetry along the y-axis:
If the left front leg is yellow and the right front leg is green, return: parallel_y(yellow, green)
If the left back leg is red and the right back leg is black, return: parallel_y(red, black)
The correct output would be: parallel_y(yellow, green), parallel_y(red, black)

Full Output Format:
Output only in the format parallel_x(part1, part2, ...), parallel_y(partA, partB, ...), with each symmetric group explicitly listed as separate entries.
Do not provide additional explanations or details.If nothing is parallel , return None"""

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

if __name__ == "__main__":
    main()