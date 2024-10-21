
"""
generate_point_cloud.py

This script generates a 3D point cloud from depth maps using binary masks to filter unwanted regions.
It takes a folder of images, applies Depth-Anything to generate depth maps, and uses Open3D to
combine the depth information into a point cloud. Only unmasked regions contribute to the final point cloud.

Usage:
    python generate_point_cloud.py <img_dir> <mask_dir> \
        --depth_outdir <depth_outdir> \
        --output_point_cloud <output_point_cloud> \
        --encoder <vits | vitb | vitl> \
        --width <image_width> --height <image_height> \
        --fx <focal_length_x> --fy <focal_length_y> \
        --cx <principal_point_x> --cy <principal_point_y> \
        --depth_scale <depth_scale> --depth_trunc <depth_trunc>

Arguments:
    1. <img_dir>: 
       - The path to the folder containing the input PNG images.
       - Example: /viscam/projects/image2Blender/RAIN-GS/sugar/imgs/dresser_5_masks_with_full/images/

    2. <mask_dir>:
       - The path to the folder containing the corresponding binary PNG masks.
       - Each mask should have the same filename as its corresponding image.
       - Example: /viscam/projects/image2Blender/RAIN-GS/sugar/imgs/dresser_5_masks_with_full/masks/0001/

Optional Arguments:
    --depth_outdir:
       - Directory to save the generated depth maps.
       - Default: "output_depth_maps/"
       - Example: /viscam/projects/image2Blender/RAIN-GS/output_depth_maps/

    --output_point_cloud:
       - File path to save the final point cloud (in PLY format).
       - Default: "initialize_point_cloud.ply"
       - Example: /viscam/projects/image2Blender/RAIN-GS/initialize_point_cloud.ply

    --encoder:
       - Encoder type used by Depth-Anything.
       - Options: "vits" (small), "vitb" (base), "vitl" (large).
       - Default: "vitb"
       - Example: --encoder vitb

    --width, --height:
       - The resolution (width and height) of the input images in pixels.
       - Default: 640 (width), 480 (height)
       - Example: --width 640 --height 480

    --fx, --fy:
       - Focal lengths in the x and y directions, in pixels.
       - These values describe how much the camera "magnifies" the scene along each axis.
       - Default: 525.0 (for both fx and fy)
       - Example: --fx 525.0 --fy 525.0

    --cx, --cy:
       - The principal point offsets in the x and y directions, in pixels.
       - These values describe where the camera's optical axis intersects the image plane.
       - Default: cx = 319.5, cy = 239.5 (for a 640x480 image, at the center)
       - Example: --cx 319.5 --cy 239.5

    --depth_scale:
       - A scaling factor to convert depth values from the depth map to real-world units.
       - Default: 1000.0
       - Example: --depth_scale 1000.0

    --depth_trunc:
       - Maximum depth value. Depth values beyond this threshold will be truncated.
       - Useful to ignore noise at very large depths.
       - Default: 3.0
       - Example: --depth_trunc 3.0

Functionality:
    1. **Reads input images and corresponding masks.** 
       - Each mask should be a binary PNG image where pixel values are 0 (masked) or 255 (unmasked).
    
    2. **Runs Depth-Anything to generate grayscale depth maps** from the input images.
       - Depth maps are saved to the specified output directory (`--depth_outdir`).

    3. **Generates a 3D point cloud** from the depth maps using Open3D.
       - Only unmasked points contribute to the final point cloud.

    4. **Saves the generated point cloud** to a PLY file.

Example Command:
    python generate_point_cloud.py \
        /viscam/projects/image2Blender/RAIN-GS/sugar/imgs/dresser_5_masks_with_full/images/ \
        /viscam/projects/image2Blender/RAIN-GS/sugar/imgs/dresser_5_masks_with_full/masks/0001/ \
        --depth_outdir /viscam/projects/image2Blender/RAIN-GS/output_depth_maps/ \
        --output_point_cloud /viscam/projects/image2Blender/RAIN-GS/initialize_point_cloud.ply \
        --encoder vitb \
        --width 640 --height 480 \
        --fx 525.0 --fy 525.0 \
        --cx 319.5 --cy 239.5 \
        --depth_scale 1000.0 --depth_trunc 3.0
"""


import os
import sys
import glob
import subprocess
import open3d as o3d
import argparse
import numpy as np

def main(args):
    # Step 1: Validate input directories
    if not os.path.isdir(args.img_dir):
        print(f"[ERROR] The provided image folder '{args.img_dir}' does not exist.")
        sys.exit(1)
    if not os.path.isdir(args.mask_dir):
        print(f"[ERROR] The provided mask folder '{args.mask_dir}' does not exist.")
        sys.exit(1)

    print(f"[INFO] Processing images from: {args.img_dir}")
    print(f"[INFO] Using masks from: {args.mask_dir}")

    # Step 2: Generate the image paths file
    print("[INFO] Generating image paths...")
    img_paths_file = "img_paths.txt"
    with open(img_paths_file, "w") as f:
        for img_path in glob.glob(os.path.join(args.img_dir, "*.png")):
            f.write(f"{img_path}\n")
    print(f"[INFO] Image paths saved to {img_paths_file}.")

    # Step 3: Run Depth-Anything to generate depth maps
    print("[INFO] Running Depth-Anything...")
    try:
        subprocess.run([
            "python", "depth-anything/run.py",
            "--encoder", args.encoder,
            "--img-path", img_paths_file,
            "--outdir", args.depth_outdir,
            "--pred-only", "--grayscale"
        ], check=True)
        print(f"[INFO] Depth maps saved to: {args.depth_outdir}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Depth-Anything failed with error: {e}")
        sys.exit(1)

    # Step 4: Generate point cloud using Open3D and masks
    print("[INFO] Generating point cloud from depth maps and masks...")
    final_pcd = o3d.geometry.PointCloud()

    for depth_map_path in glob.glob(os.path.join(args.depth_outdir, "*.png")):
        mask_path = os.path.join(args.mask_dir, os.path.basename(depth_map_path))
        if not os.path.isfile(mask_path):
            print(f"[WARNING] Mask not found for {depth_map_path}. Skipping...")
            continue

        print(f"[DEBUG] Processing {depth_map_path} with mask {mask_path}")

        # Load the depth map and mask
        depth_image = o3d.io.read_image(depth_map_path)
        mask_image = o3d.io.read_image(mask_path)
        mask_array = np.asarray(mask_image) / 255.0  # Normalize mask to [0, 1]

        # Generate the point cloud from the depth map
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            args.width, args.height, args.fx, args.fy, args.cx, args.cy
        )
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image, intrinsic, depth_scale=args.depth_scale,
            depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False
        )

        # Apply the mask to filter points
        points = np.asarray(pcd.points)
        valid_indices = mask_array.flatten() > 0  # Keep only unmasked points
        valid_points = points[valid_indices]

        # Add valid points to the final point cloud
        valid_pcd = o3d.geometry.PointCloud()
        valid_pcd.points = o3d.utility.Vector3dVector(valid_points)
        final_pcd += valid_pcd

    if len(final_pcd.points) == 0:
        print("[ERROR] No valid points generated from the depth maps and masks.")
        sys.exit(1)

    # Step 5: Ensure the output directory exists
    output_dir = os.path.dirname(args.output_point_cloud)
    if not os.path.exists(output_dir):
        print(f"[INFO] Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    # Step 6: Save the final point cloud
    o3d.io.write_point_cloud(args.output_point_cloud, final_pcd)
    print(f"[INFO] Final point cloud saved to: {args.output_point_cloud}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a point cloud from depth maps using binary masks.")

    # Arguments
    parser.add_argument("img_dir", type=str, help="Path to the folder containing PNG images.")
    parser.add_argument("mask_dir", type=str, help="Path to the folder containing binary masks.")
    parser.add_argument("--depth_outdir", type=str, default="output_depth_maps",
                        help="Directory to save depth maps.")
    parser.add_argument("--output_point_cloud", type=str, default="initialize_point_cloud.ply",
                        help="Path to save the final point cloud.")
    parser.add_argument("--encoder", type=str, default="vitb",
                        choices=["vits", "vitb", "vitl"], help="Encoder type for Depth-Anything.")
    parser.add_argument("--width", type=int, default=640, help="Image width for intrinsic camera.")
    parser.add_argument("--height", type=int, default=480, help="Image height for intrinsic camera.")
    parser.add_argument("--fx", type=float, default=525.0, help="Focal length in x-axis.")
    parser.add_argument("--fy", type=float, default=525.0, help="Focal length in y-axis.")
    parser.add_argument("--cx", type=float, default=319.5, help="Principal point in x-axis.")
    parser.add_argument("--cy", type=float, default=239.5, help="Principal point in y-axis.")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Depth scale.")
    parser.add_argument("--depth_trunc", type=float, default=3.0, help="Depth truncation value.")

    args = parser.parse_args()
    main(args)
