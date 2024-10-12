import os
from PIL import Image
import numpy as np
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import matplotlib.pyplot as plt

# Setup
video_path = "videos/bookshelf_sam"
sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
device = "cuda"

frame_names = [
    p for p in os.listdir(video_path)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
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

# Auto-masking of first frame (from automatic mask generation notebook)
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
first_frame_path = os.path.join(video_path, frame_names[0])
first_frame = Image.open(first_frame_path)
first_frame = np.array(first_frame.convert("RGB"))
mask_generator = SAM2AutomaticMaskGenerator(sam2)
auto_masks = mask_generator.generate(first_frame)
print("Number of auto-masks:", len(auto_masks))

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
    plt.imshow(Image.open(os.path.join(video_path, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.axis('off')
    plt.savefig(f'results/bookshelf_{out_frame_idx:04d}.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()
