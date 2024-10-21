
"""
Input/output paths (--img-dir, --depth-outdir, --output-point-cloud).
Depth-Anything parameters (--encoder for encoder type).
Camera intrinsics (--width, --height, --fx, --fy, --cx, --cy).
Depth map processing (--depth-scale, --depth-trunc).
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
