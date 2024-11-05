import os
from PIL import Image
import numpy as np
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import matplotlib.pyplot as plt
import argparse
import cv2

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
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)
    return img

# Parse arguments for input, output, and frame ranges
# python sam2-scripts/sam_video_front_and_back.py --input_dir sugar/imgs/11_04_tablechairs/images --output_dir sam2-scripts/11_04_tablechairs --start 0 --middle 33 --end 122
parser = argparse.ArgumentParser(description="SAM2 Video Segmentation and Mask Generation")
parser.add_argument("--input_dir", type=str, required=True, help="Directory containing video frames")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save binary masks and results")
parser.add_argument("--start", type=int, required=True, help="Frame number to start reverse processing")
parser.add_argument("--middle", type=int, required=True, help="Frame number to start normal processing")
parser.add_argument("--end", type=int, required=True, help="Frame number to end normal processing")
args = parser.parse_args()

# Setup
video_path = args.input_dir
binary_mask_output_dir = args.output_dir
sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda"

# Convert from png to jpg (needed for sam)
def convert_png_to_jpg(directory):
    for file_name in os.listdir(directory):
        if file_name.endswith(".png"):
            png_path = os.path.join(directory, file_name)
            jpg_path = os.path.join(directory, file_name.replace(".png", ".jpg"))
            image = cv2.imread(png_path)
            cv2.imwrite(jpg_path, image)
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

# Save binary mask as a single-channel image
def save_binary_mask(mask, out_path):
    binary_mask = (mask > 0).astype(np.uint8) * 255
    binary_mask_img = Image.fromarray(binary_mask[0], mode="L")
    binary_mask_img.save(out_path)

# Auto-masking of the first frame
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
first_frame_path = os.path.join(video_path, frame_names[args.middle])
first_frame = Image.open(first_frame_path)
first_frame = np.array(first_frame.convert("RGB"))

# mask_generator = SAM2AutomaticMaskGenerator(sam2)
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
        min_mask_region_area=100,
        use_m2m=True,
    )
auto_masks = mask_generator.generate(first_frame)
print("Number of auto-masks:", len(auto_masks))

plt.figure(figsize=(20, 20))
new_img = show_anns(auto_masks)
plt.axis('off')

if new_img is not None:
    output_path = os.path.join(binary_mask_output_dir, 'middle.jpg')
    plt.imsave(output_path, new_img)
plt.close()

print('middle done')

# Initialize video predictor
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
inference_state = predictor.init_state(video_path=video_path)
dtype = next(predictor.parameters()).dtype
lowres_side_length = predictor.image_size // 4
for mask_idx, mask_result in enumerate(auto_masks):
    mask_tensor = torch.tensor(mask_result["segmentation"], dtype=dtype, device=device)
    lowres_mask = torch.nn.functional.interpolate(
        mask_tensor.unsqueeze(0).unsqueeze(0),
        size=(lowres_side_length, lowres_side_length),
        mode="bilinear",
        align_corners=False,
    ).squeeze()
    _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=args.middle,
        obj_id=mask_idx,
        mask=lowres_mask,
    )

# Propagate from middle to start (reverse)
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=args.middle, max_frame_num_to_track=args.middle-args.start, reverse=True):
    video_segments[out_frame_idx] = {}
    zeros = []
    for i, out_obj_id in enumerate(out_obj_ids):
        video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
        if video_segments[out_frame_idx][out_obj_id].sum() == 0:
            zeros.append(out_obj_id)

    print(out_frame_idx, zeros)

# Propagate from middle to end (forward)
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=args.middle, max_frame_num_to_track=args.end-args.middle, reverse=False):
    video_segments[out_frame_idx] = {}
    zeros = []
    for i, out_obj_id in enumerate(out_obj_ids):
        video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
        if video_segments[out_frame_idx][out_obj_id].sum() == 0:
            zeros.append(out_obj_id)

    print(out_frame_idx, zeros)

# Save results
for out_frame_idx in range(len(video_segments)):
    plt.figure(figsize=(6, 4))
    frame_binary_mask_dir = os.path.join(binary_mask_output_dir, f'frame_{out_frame_idx:04d}')
    os.makedirs(frame_binary_mask_dir, exist_ok=True)

    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        binary_mask_path = os.path.join(frame_binary_mask_dir, f'mask_{out_obj_id:02d}.png')
        save_binary_mask(out_mask, binary_mask_path)

    plt.axis('off')
    plt.savefig(f'{frame_binary_mask_dir}/all.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

# Save the video
frame_folders = sorted([f for f in os.listdir(binary_mask_output_dir) if f.startswith("frame_")])
frame_files = []
for folder in frame_folders:
    image_path = os.path.join(binary_mask_output_dir, folder, "all.jpg")
    if os.path.isfile(image_path):
        frame_files.append(image_path)

first_frame = cv2.imread(frame_files[0])
height, width, _ = first_frame.shape
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_path = os.path.join(binary_mask_output_dir, f'{os.path.basename(binary_mask_output_dir)}.mp4')
video_writer = cv2.VideoWriter(video_path, fourcc, 30, (width, height))

for i, frame_file in enumerate(frame_files):
    frame = cv2.imread(frame_file)

    # Define the position where the text will be placed (upper right corner)
    text = f'{i+1}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (0, 0, 255)  # Red color in BGR format
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

    # Position the text near the upper right corner, offset by 10 pixels from the border
    text_x = frame.shape[1] - text_size[0] - 10  # Width minus text width minus some padding
    text_y = text_size[1] + 10  # Offset from top with padding

    # Add the frame number text to the frame
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

    # Write the frame with the text to the video
    video_writer.write(frame)

video_writer.release()
print(f"Video saved to {video_path}")
