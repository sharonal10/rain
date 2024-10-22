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

    img_path = r'/viscam/projects/image2Blender/RAIN-GS/Depth-Anything/10_21-dresser-no_bg/0001.png'
    color_image = o3d.io.read_image(img_path)
    depth_image = o3d.io.read_image(depth_map_path)
    mask_image = o3d.io.read_image(mask_path)
    mask_array = np.asarray(mask_image) / 255.0
    mask_array = np.where(mask_array >= 0.5, 1, 0)
    color_array = np.asarray(color_image)
    depth_array = np.asarray(depth_image)
    masked_color_array = color_array * mask_array[:, :, np.newaxis]
    masked_depth_array = depth_array * mask_array
    color_image = o3d.geometry.Image(masked_color_array.astype(np.uint8))
    depth_image = o3d.geometry.Image(masked_depth_array.astype(np.uint16))
    # color_image = 

    # import pdb; pdb.set_trace()

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        args.width, args.height, args.fx, args.fy, args.cx, args.cy
    )
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    print(f"Number of points: {len(pcd.points)}")
    o3d.io.write_point_cloud("pc1.ply", pcd, write_ascii=True)
    # pcd = o3d.geometry.PointCloud.create_from_depth_image(
    #     depth_image, intrinsic, depth_scale=args.depth_scale,
    #     depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False
    # )

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