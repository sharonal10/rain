import os
from PIL import Image
import numpy as np
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import matplotlib.pyplot as plt
import argparse
import cv2

parser = argparse.ArgumentParser(description="SAM2 Video Segmentation and Mask Generation")
parser.add_argument("--input_dir", type=str, required=True, help="Directory containing video frames")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save binary masks and results")
parser.add_argument("--min_area", type=int, default=2000, help="Minimum mask area")
args = parser.parse_args()

# Setup
video_path = args.input_dir
binary_mask_output_dir = args.output_dir  # Directory to save binary masks
sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
device = "cuda"
min_area = args.min_area

# convert from png to jpg (needed for sam)
def convert_png_to_jpg(directory):
    """
    Converts all .png files in the specified directory to .jpg format.
    
    Args:
        directory (str): The path to the directory where the conversion will happen.
    """
    # Iterate over files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith(".png"):
            # Form full file paths
            png_path = os.path.join(directory, file_name)
            jpg_path = os.path.join(directory, file_name.replace(".png", ".jpg"))

            # Read the .png image
            image = cv2.imread(png_path)

            # Convert to .jpg and save
            cv2.imwrite(jpg_path, image)

            # Optionally, remove the original .png file
            os.remove(png_path)

    print(f"Conversion complete for {directory}.")
convert_png_to_jpg(video_path)

# Create the output directory for binary masks if it doesn't exist
os.makedirs(binary_mask_output_dir, exist_ok=True)

frame_names = [
    p for p in os.listdir(video_path)
    if os.path.splitext(p)[-1] in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Save binary mask as a single-channel image (white mask on black background)
def save_binary_mask(mask, out_path):
    #print(mask.shape)
    binary_mask = (mask > 0).astype(np.uint8) * 255  # Convert mask to binary (0 or 255)
    binary_mask_img = Image.fromarray(binary_mask[0], mode="L")  # Convert to a PIL Image in grayscale mode
    binary_mask_img.save(out_path)

# Auto-masking of first frame (from automatic mask generation notebook)
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
first_frame_path = os.path.join(video_path, frame_names[0])
first_frame = Image.open(first_frame_path)
first_frame = np.array(first_frame.convert("RGB"))

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
    min_mask_region_area=min_area,
    use_m2m=True,
)
auto_masks = mask_generator.generate(first_frame)
# print(x, "Number of auto-masks:", len(auto_masks))

# Add every 'auto-mask' as it's own prompt for video tracking
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
inference_state = predictor.init_state(video_path=video_path)
dtype = next(predictor.parameters()).dtype
lowres_side_length = predictor.image_size // 4
for mask_idx, mask_result in enumerate(auto_masks):

    # Get mask into form expected by the model
    mask_tensor = torch.tensor(mask_result["segmentation"], dtype=dtype, device=device)
    lowres_mask = torch.nn.functional.interpolate(
        mask_tensor.unsqueeze(0).unsqueeze(0),
        size=(lowres_side_length, lowres_side_length),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # Add each mask as it's own 'object' to segment
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=mask_idx,
        mask=lowres_mask,
    )

# Do video segmentation (same as video segmentation notebook)
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
    }

for out_frame_idx in range(len(video_segments)):
    plt.figure(figsize=(6, 4))
    # plt.imshow(Image.open(os.path.join(video_path, frame_names[out_frame_idx])))

    frame_binary_mask_dir = os.path.join(binary_mask_output_dir, f'frame_{out_frame_idx:04d}')
    os.makedirs(frame_binary_mask_dir, exist_ok=True)  # Create a subfolder for each frame

    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

        # Save the binary mask
        binary_mask_path = os.path.join(frame_binary_mask_dir, f'mask_{out_obj_id:02d}.png')
        save_binary_mask(out_mask, binary_mask_path)  # Save binary mask image

    plt.axis('off')
    plt.savefig(f'{frame_binary_mask_dir}/all.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()


# save the video
frame_folders = sorted([f for f in os.listdir(binary_mask_output_dir) if f.startswith("frame_")])

# List to store frame file paths
frame_files = []
for folder in frame_folders:
    image_path = os.path.join(binary_mask_output_dir, folder, "all.jpg")
    if os.path.isfile(image_path):
        frame_files.append(image_path)

# Read the first frame to get dimensions
first_frame = cv2.imread(frame_files[0])
height, width, _ = first_frame.shape

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4 video
video_writer = cv2.VideoWriter(os.path.join(binary_mask_output_dir, f'{binary_mask_output_dir}.mp4'), fourcc, 30, (width, height))

# Write each frame to the video
for frame_file in frame_files:
    frame = cv2.imread(frame_file)
    video_writer.write(frame)

# Release the video writer
video_writer.release()
print(f"Video saved to {os.path.join(binary_mask_output_dir, f'{binary_mask_output_dir}.mp4')}")
