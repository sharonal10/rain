import os
import argparse
import numpy as np
from PIL import Image
import cv2

parser = argparse.ArgumentParser(description="Process and combine binary masks into images.")
parser.add_argument('--input_dir', type=str, required=True, help="Directory containing various subdirectories with frame-wise binary SAM masks")
parser.add_argument('--output_image_dir', type=str, required=True, help="Directory where the combined masks will be saved")
parser.add_argument('--base_image_dir', type=str, required=True, help="Directory containing the base (ground truth) images per frame")

args = parser.parse_args()

input_dir = args.input_dir
output_image_dir = args.output_image_dir
base_image_dir = args.base_image_dir

colors = {
    '000.png': (255, 0, 0),
    '001.png': (0, 255, 0),
    '002.png': (0, 0, 255),
    '003.png': (255, 255, 0),
}

frame_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
fps = 2

os.makedirs(output_image_dir, exist_ok=True)

for frame_dir in frame_dirs:
    base_image_path = os.path.join(base_image_dir, f'{frame_dir}.png')
    base_image = np.array(Image.open(base_image_path).convert("RGB"))
    frame_path = os.path.join(input_dir, frame_dir)
    
    final_image = base_image.copy()

    for mask_name, color in colors.items():
        mask_path = os.path.join(frame_path, mask_name)
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert("L"))
            mask_height, mask_width = mask.shape
            color_mask = np.zeros_like(final_image, dtype=np.float32)
            if (mask_height, mask_width) != final_image.shape[:2]:
                mask = cv2.resize(mask, (final_image.shape[1], final_image.shape[0]))

            for c in range(3):
                color_mask[:, :, c] = (mask / 255.0) * (color[c] / 255.0)
            alpha = (mask / 255.0 * 0.5).reshape(final_image.shape[0], final_image.shape[1], 1)
            final_image = (1 - alpha) * final_image + alpha * (color_mask * 255)

    final_image = np.clip(final_image, 0, 255).astype(np.uint8)
    
    text = frame_dir
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (0, 0, 255)
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = final_image.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10
    cv2.putText(final_image, text, (text_x, text_y), font, font_scale, color, thickness)

    output_image_path = os.path.join(output_image_dir, f"{frame_dir}.png")
    cv2.imwrite(output_image_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

