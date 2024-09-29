import torch
import os
import torchvision.transforms as transforms

_opj = os.path.join

def save_image(image, filename):
    to_pil = transforms.ToPILImage()
    image_pil = to_pil(image)

    # Save the image
    image_pil.save(filename)

# assume for now that the images are [0, 1]
def get_brightness(tensor_img):
    r, g, b = tensor_img[0, :, :], tensor_img[1, :, :], tensor_img[2, :, :]
    brightness = r + g + b
    print(brightness.shape)
    return brightness

def get_centroid(tensor_img, threshold=1/256, save_idx=0, to_save=False, save_dir=None):
    brightness = get_brightness(tensor_img)
    indices = (brightness > threshold).nonzero(as_tuple=True)
    if len(indices[0]) == 0:  # No pixels above threshold
        ret = (tensor_img.shape[1]//2, tensor_img.shape[2]//2)
    else:
        # Calculate the centroid (mean of x and y indices)
        y = indices[0].float()
        x = indices[1].float()
        ret = (int(torch.mean(x).detach()), int(torch.mean(y).detach()))
    
    if to_save:
        res = (brightness > threshold) * 32/256
        res[ret[1], ret[0]] = 1
        save_image(res.unsqueeze(0).repeat(3, 1, 1), _opj(save_dir, f'brightness_{save_idx}.png'))

    return ret

def align_image(tensor_img, reference_centroid, target_centroid, save_idx=0, to_save=False, save_dir=None):
    offset_x = int(target_centroid[0] - reference_centroid[0])
    offset_y = int(target_centroid[1] - reference_centroid[1])
    
    # Apply the shift using torch.roll for each channel
    aligned_image = torch.roll(tensor_img, shifts=(offset_y, offset_x), dims=(1, 2))

    if to_save:
        save_image(aligned_image, _opj(save_dir, f'aligned_{save_idx}.png'))
    
    return aligned_image

def align_images(images_dict, to_save=False, save_dir=None):
    images = list(images_dict.values())

    reference_centroid = (images[0].shape[1]//2, images[0].shape[2]//2)

    aligned_images = []
    for i, img in enumerate(images):
        assert img.shape == images[0].shape, (i, img.shape, images[0].shape)
        centroid = get_centroid(img, save_idx=i, to_save=to_save, save_dir=save_dir)
        centered = align_image(img, reference_centroid, centroid, save_idx=i, to_save=to_save, save_dir=save_dir)
        aligned_images.append(centered)

    return aligned_images

