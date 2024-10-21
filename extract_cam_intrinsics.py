import numpy as np
import os
import argparse

# Define the camera model names
CAMERA_MODEL_IDS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV"
}

class Camera:
    def __init__(self, id, model, width, height, params):
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params

def read_next_bytes(fid, num_bytes, format_char_sequence):
    return np.frombuffer(fid.read(num_bytes), dtype=np.dtype(format_char_sequence))

def focal2fov(focal_length, dimension):
    return 2 * np.arctan(dimension / (2 * focal_length)) * 180 / np.pi

# Function to read camera intrinsics from binary file
def read_intrinsics_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS.get(model_id, "UNKNOWN")
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = 0
            if model_name == "SIMPLE_PINHOLE":
                num_params = 3
            elif model_name == "PINHOLE":
                num_params = 4
            elif model_name == "SIMPLE_RADIAL":
                num_params = 4
            elif model_name == "RADIAL":
                num_params = 5
            elif model_name == "OPENCV":
                num_params = 8
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(id=camera_id, model=model_name, width=width, height=height, params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

# Function to write formatted intrinsics to file
def write_intrinsics_to_file(cameras, output_file):
    with open(output_file, "w") as f:
        for camera_id, intr in cameras.items():
            fx, fy, cx, cy = None, None, None, None
            if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
                fx = fy = intr.params[0]
                cx = intr.params[1]
                cy = intr.params[2]
            elif intr.model == "PINHOLE" or intr.model == "OPENCV":
                fx = intr.params[0]
                fy = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]
            elif intr.model == "RADIAL":
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

# Function to write all camera details to a separate file
def write_full_info_to_file(cameras, output_file):
    with open(output_file, "w") as f:
        for camera_id, intr in cameras.items():
            f.write(f"camera_id: {camera_id}\n")
            f.write(f"model: {intr.model}\n")
            f.write(f"width: {intr.width}\n")
            f.write(f"height: {intr.height}\n")
            f.write(f"params: {intr.params}\n")  # Writing all the intrinsic parameters
            f.write("\n")

# Main function with argparse
def main():
    parser = argparse.ArgumentParser(description="Extract COLMAP camera intrinsics and write to files.")
    parser.add_argument("input_file", type=str, help="Path to the COLMAP binary camera file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output text files.")
    
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
