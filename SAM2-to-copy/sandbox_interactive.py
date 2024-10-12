import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return None
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)
    return img

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device='cuda', apply_postprocessing=False)
mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=128,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=0.7,
    crop_n_layers=1,
    box_nms_thresh=0.7,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=2.0,
    use_m2m=True,
)

def process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        print(image_path)
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        
        masks = mask_generator.generate(image)
        
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        new_img = show_anns(masks)
        plt.axis('off')
        
        if new_img is not None:
            output_path = os.path.join(output_dir, image_file)
            plt.imsave(output_path, new_img)
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Process all images in an input directory and save the results to an output directory.")
    parser.add_argument('-i', '--input_dir', required=True, help="Directory containing images to process.")
    parser.add_argument('-o', '--output_dir', required=True, help="Directory to save processed images.")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_images(args.input_dir, args.output_dir)

