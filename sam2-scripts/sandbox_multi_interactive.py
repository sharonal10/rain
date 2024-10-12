import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# just save everything in binary masks and parse through faster
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def process_images(input_dir):
    # assumes 10 cameras. returns the 10 angles that will be used.
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    image_files = sorted(image_files)
    print([idx for idx, c in enumerate(image_files) if idx % 30 == 0])
    image_files = [c for idx, c in enumerate(image_files) if idx % 30 == 0]
    
    return image_files

def save_binary_mask(mask, out_path):
    #print(mask.shape)
    binary_mask = (mask > 0).astype(np.uint8) * 255  # Convert mask to binary (0 or 255)
    binary_mask_img = Image.fromarray(binary_mask, mode="L")  # Convert to a PIL Image in grayscale mode
    binary_mask_img.save(out_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Process all images in an input directory and save the results to an output directory.")
    parser.add_argument('-i', '--input_dir', required=True, help="Directory containing images to process.")
    parser.add_argument('-o', '--output_dir', required=True, help="Directory to save processed images.")
    
    return parser.parse_args()

args = parse_args()
os.makedirs(args.output_dir, exist_ok=True)

sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)

mask_generator = SAM2AutomaticMaskGenerator(sam2)

file_names = process_images(args.input_dir)

for curr_basename in file_names:
    curr_name = os.path.join(args.input_dir,  curr_basename)
    image = Image.open(curr_name)
    image = np.array(image.convert("RGB"))
    masks = mask_generator.generate(image)
    os.makedirs(os.path.join(args.output_dir, os.path.splitext(curr_basename)[0]), exist_ok = True)
    for i, mask in enumerate(masks):
        save_binary_mask(mask['segmentation'], os.path.join(args.output_dir, os.path.splitext(curr_basename)[0], f"{i:03}.png"))