import os
import numpy as np
from PIL import Image
import cv2

input_dir = rf""
output_image_dir = rf""
base_image_dir = rf''

colors = {
    '000.png': (255, 0, 0),
    '001.png': (0, 255, 0),
    '002.png': (0, 0, 255),
    '003.png': (255, 255, 0),
}

frame_dirs = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

first_frame_dir = os.path.join(input_dir, frame_dirs[0])
first_mask_path = os.path.join(first_frame_dir, '000.png')
first_mask = np.array(Image.open(first_mask_path))
frame_height, frame_width = first_mask.shape[:2]
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
            if np.any(mask > 0):
                color_mask = np.zeros_like(final_image, dtype=np.float32)
                for c in range(3):
                    color_mask[:, :, c] = (mask / 255.0) * (color[c] / 255.0)
                alpha = (mask / 255.0 * 0.5).reshape(frame_height, frame_width, 1)
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
