import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0,0,1,0.4), thickness=1) 

    ax.imshow(img)
    return img

for file_name in os.listdir('cabinets'):
    image_path = os.path.join('cabinets', file_name)
    image = Image.open(image_path)
    width, height = image.size
    center_x = width // 2
    center_y = height // 2
    image = np.array(image.convert("RGB"))

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')

    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)
    input_point = np.array([[center_x, center_y]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # sizes = [np.sum(mask) for mask in masks]
    # mask_idx = sizes.index(max(sizes))
    
    # import pdb; pdb.set_trace()
    for mask_idx in range(len(masks)):
        binary_array = masks[mask_idx]

        # Convert the NumPy array to a PIL image
        image = Image.fromarray(binary_array * 255)  # Multiply by 255 to make it a proper binary image (0 or 255)

        # Save the image as a PNG file
        # image.save('results/sandbox3.png')
        mask_image_name = os.path.splitext(file_name)[0] + f'_mask_image{mask_idx}.png'
        plt.imsave(os.path.join('results', mask_image_name), image)



# sam2 = build_sam2(model_cfg, sam2_checkpoint, device ='cuda', apply_postprocessing=False)

# mask_generator = SAM2AutomaticMaskGenerator(sam2)

# masks = mask_generator.generate(image)

# print(len(masks))
# print(masks[0].keys())

# plt.figure(figsize=(20,20))
# plt.imshow(image)
# new_img = show_anns(masks)
# plt.axis('off')
# plt.imsave('results/label13.jpg', new_img)
# mask_generator_2 = SAM2AutomaticMaskGenerator(
#     model=sam2,
#     points_per_side=64,
#     points_per_batch=128,
#     pred_iou_thresh=0.7,
#     stability_score_thresh=0.92,
#     stability_score_offset=0.7,
#     crop_n_layers=1,
#     box_nms_thresh=0.7,
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=2.0,
#     use_m2m=True,
# )

# masks2 = mask_generator_2.generate(image)
# print(len(masks2))
# print(masks2[0].keys())
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# new_img2 = show_anns(masks2)
# plt.axis('off')
# plt.imsave('results/label13b.jpg', new_img2) 
