import os
import sys
import glob
import subprocess
import open3d as o3d
import argparse
import numpy as np

def main(args):
    print("[INFO] Generating point cloud from depth maps and masks...")
    final_pcd = o3d.geometry.PointCloud()
    depth_map_path = os.path.join(args.depth_dir, "0001_depth.png")
    mask_path = os.path.join(args.mask_dir, "0001", "004.png") # for now just bottom drawer

    print(f"[DEBUG] Processing {depth_map_path} with mask {mask_path}")

    depth_image = o3d.io.read_image(depth_map_path)
    mask_image = o3d.io.read_image(mask_path)
    mask_array = np.asarray(mask_image) / 255.0
    print("Max value:", np.max(mask_array))
    print("Min value:", np.min(mask_array))

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        args.width, args.height, args.fx, args.fy, args.cx, args.cy
    )
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_image, intrinsic, #depth_scale=args.depth_scale,
        #depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a point cloud from depth maps using binary masks.")

    # Arguments
    parser.add_argument("--mask_dir", type=str, help="Path to the folder containing binary masks.")
    parser.add_argument("--depth_dir", type=str, default="output_depth_maps",
                        help="Directory of depth maps.")
    parser.add_argument("--output_point_cloud", type=str, default="initialize_point_cloud.ply",
                        help="Path to save the final point cloud.")
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