import numpy as np
import os
import argparse
from collections import namedtuple
import struct

# Define CameraModel as a namedtuple
CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])

# Camera models with their corresponding IDs and number of parameters
CAMERA_MODELS = {
    0: CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    1: CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    2: CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    3: CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    4: CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    5: CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    6: CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    7: CameraModel(model_id=7, model_name="FOV", num_params=5),
    8: CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    9: CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    10: CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}

class Camera:
    def __init__(self, id, model, width, height, params):
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_intrinsics_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, np.uint64)[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, np.dtype([("camera_id", np.int32), 
                                                                  ("model_id", np.int32), 
                                                                  ("width", np.uint64), 
                                                                  ("height", np.uint64)]))
            camera_id = camera_properties["camera_id"]
            model_id = camera_properties["model_id"]
            model = CAMERA_MODELS[model_id]
            width = camera_properties["width"]
            height = camera_properties["height"]
            
            # Read the camera parameters based on the model
            num_params = model.num_params
            params = read_next_bytes(fid, 8 * num_params, np.float64)
            
            cameras[camera_id] = Camera(id=camera_id, model=model.model_name, 
                                        width=width, height=height, params=params)
    return cameras

def write_intrinsics_to_file(cameras, output_file):
    with open(output_file, "w") as f:
        for camera_id, intr in cameras.items():
            fx, fy, cx, cy = None, None, None, None
            if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
                fx = fy = intr.params[0]
                cx = intr.params[1]
                cy = intr.params[2]
            elif intr.model == "PINHOLE" or intr.model == "OPENCV" or intr.model == "OPENCV_FISHEYE" or intr.model == "FULL_OPENCV":
                fx = intr.params[0]
                fy = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]
            elif intr.model == "RADIAL" or intr.model == "FOV" or intr.model == "RADIAL_FISHEYE":
                fx = fy = intr.params[0]
                cx = intr.params[1]
                cy = intr.params[2]
            f.write(f"camera_id: {camera_id}\n")
            f.write(f"width: {intr.width}\n")
            f.write(f"height: {intr.height}\n")
            f.write(f"fx: {fx}\n")
            f.write(f"fy: {fy}\n")
            f.write(f"cx: {cx}\n")
            f.write(f"cy: {cy}\n")
            f.write("\n")

def write_full_info_to_file(cameras, output_file):
    with open(output_file, "w") as f:
        for camera_id, intr in cameras.items():
            f.write(f"camera_id: {camera_id}\n")
            f.write(f"model: {intr.model}\n")
            f.write(f"width: {intr.width}\n")
            f.write(f"height: {intr.height}\n")
            f.write(f"params: {intr.params}\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Extract COLMAP camera intrinsics and write to files.")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the COLMAP binary camera file.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory to save the output text files.")
    
    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read camera intrinsics from binary file
    cameras = read_intrinsics_binary(args.input_file)
    
    # Write the formatted intrinsics to the first file
    intrinsics_file = os.path.join(args.output_dir, "camera_intrinsics.txt")
    write_intrinsics_to_file(cameras, intrinsics_file)
    
    # Write the full information to the second file
    full_info_file = os.path.join(args.output_dir, "camera_full_info.txt")
    write_full_info_to_file(cameras, full_info_file)

if __name__ == "__main__":
    main()
