import argparse
from PIL import Image, ImageDraw, ImageFont
import math
import os

def split_image_add_axes(image_path, output_path):
    img = Image.open(image_path)
    width, height = img.size

    # Create a new image with a white background
    new_img = Image.new('RGB', (width, height), color='white')
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)
    draw.line([(0, height//2), (width, height//2)], fill='black', width=2)  # x-axis
    draw.line([(width//2, 0), (width//2, height)], fill='black', width=2)  # y-axis
    try:
        font = ImageFont.load_default()
    except:
        font = None
    def draw_text(position, text, fill='black', font=font):
        if font:
            draw.text(position, text, fill=fill, font=font)
        else:
            draw.text(position, text, fill=fill)
    draw_text((width - 30, height//2 + 10), "x")
    draw_text((width//2 + 10, 20), "y")
    draw_text((width - 30, height//2 - 30), "+x")
    draw_text((20, 20), "-y")
    draw_text((20, height - 40), "-x")
    draw_text((width//2 + 10, height - 40), "+y")
    radius = min(width, height) // 4
    angles = [45, 135, 225, 315]
    for angle in angles:
        x = width//2 + int(radius * math.cos(math.radians(angle)))
        y = height//2 - int(radius * math.sin(math.radians(angle)))
        draw_text((x, y), f"{angle}°")
    draw.line([(0, 0), (width, height)], fill='red', width=2)
    draw.line([(width, 0), (0, height)], fill='red', width=2)
    new_img.save(output_path)

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            split_image_add_axes(input_path, output_path)
            print(f"Processed: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Annotate images with axes and angles")
    parser.add_argument("--input_folder", required=True, help="Path to the folder containing input images")
    parser.add_argument("--output_folder", required=True, help="Path to the folder where annotated images will be saved")
    args = parser.parse_args()

    process_images(args.input_folder, args.output_folder)
    print("All images processed successfully.")

if __name__ == "__main__":
    main()