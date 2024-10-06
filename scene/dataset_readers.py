import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, masks_folder, mask_id):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE" or intr.model=="OPENCV" or intr.model=="RADIAL":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        mask_path = os.path.join(masks_folder, os.path.splitext(os.path.basename(extr.name))[0], f"{mask_id:03}.png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path)
        else:
            mask = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              mask=mask)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    colors = np.zeros_like(colors)
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


# Rotation matrix for a single axis (Rx, Ry, Rz)
def rotation_matrix(angles):
    theta_x, theta_y, theta_z = angles
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx


# Transform points to the global frame
def transform_points_local_to_global(points, box_center, box_rotation):
    # Rotate points using rotation matrix
    R = rotation_matrix(box_rotation)
    rotated_points = np.dot(points, R.T)
    
    # Translate points relative to box center
    transformed_points = rotated_points + box_center
    
    return transformed_points

# Generate points inside the box in its local coordinate system
def generate_points_in_box(box_size, num_points):
    # half_size = box_size / 2
    
    # Generate random points inside a box from [-half_size, +half_size]
    points_local = np.random.uniform(-box_size, box_size, (num_points, 3))
    
    return points_local

def read_box(filename):
    # Read the content of the file
    with open(filename, 'r') as file:
        # Read the entire content and split it into a list of numbers
        numbers = list(map(float, file.read().split()))

    # Ensure we have exactly 9 numbers
    if len(numbers) != 10:
        raise ValueError("The file does not contain exactly 10 numbers.")

    # Assign the numbers to respective categories
    box_center = np.array(numbers[0:3])
    box_rotation = np.radians(numbers[3:6])
    box_size = np.array(numbers[6:9])
    num_points = int(numbers[9])

    return box_center, box_rotation, box_size, num_points

def readColmapSceneInfo(path, images, eval, llffhold=8, args_dict=None, mask_id=None, custom_ply_path=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir), masks_folder=os.path.join(path, 'masks'), mask_id=mask_id)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    llffhold = len(cam_infos)/args_dict['num_cams']
    print('args.eval', args_dict['eval'])
    if eval and not args_dict['render_only']:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    print(f"number of cameras: {len(train_cam_infos)}")
    nerf_normalization = getNerfppNorm(train_cam_infos)

    if not args_dict['render_only'] and not args_dict['box_gen']:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")

        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            pcd = fetchPly(ply_path)
        except:
            pcd = None
            
        if args_dict['train_from'] == "noisy_sfm":
            print(f"Adding noise to the point cloud (1.0)...")
            xyz += np.random.normal(0, 1.0, xyz.shape)
            rgb += np.random.normal(0, 1.0, rgb.shape)
            rgb =  np.clip(rgb, 0, 255)
            
        if (args_dict is not None) and (args_dict['paper_random'] or args_dict['ours'] or args_dict['ours_new']):
            if not args_dict['ours'] and args_dict['train_from'] == "reprojection":
                try:
                    xyz, rgb, error = read_points3D_binary(bin_path)
                except:
                    xyz, rgb, error = read_points3D_text(txt_path)
                    
                error_rate = 10
                err_thr = np.percentile(error[:,0], error_rate)
                xyz = xyz[(error[:,0]<err_thr),:]
                rgb = rgb[(error[:,0]<err_thr),:]
                print(f"Train with {len(xyz)} sparse SfM points... (Sparse Type: Reprojection Error Top {error_rate}%)")
                storePly(ply_path, xyz, rgb)
                
            elif not args_dict['ours'] and ((args_dict['train_from'] == "cluster") or (args_dict['train_from'] == "noisy_sfm")):
                from sklearn.cluster import HDBSCAN
                hdbscan = HDBSCAN(min_cluster_size=5, store_centers='both').fit(xyz)
                xyz = hdbscan.centroids_
                shs = np.random.random((len(xyz), 3))
                rgb = SH2RGB(shs) * 255
                print(f"Train with {len(xyz)} sparse SfM points... (Sparse Type: cluster)")
                storePly(ply_path, xyz, rgb)
            
            else:
                num_pts = args_dict["num_gaussians"]
                
                cam_pos = []
                for k in cam_extrinsics.keys():
                    cam_pos.append(cam_extrinsics[k].tvec)
                cam_pos = np.array(cam_pos)
                min_cam_pos = np.min(cam_pos)
                max_cam_pos = np.max(cam_pos)
                mean_cam_pos = (min_cam_pos + max_cam_pos) / 2.0
                cube_mean = (max_cam_pos - min_cam_pos) * 1.5
                
                if args_dict['paper_random']:        
                    xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"] * 3 - nerf_normalization["radius"] * 1.5
                    xyz = xyz + nerf_normalization["translate"]
                    print(f"Generating random point cloud ({num_pts})...")
                else:
                    xyz = np.random.random((num_pts, 3)) * (max_cam_pos - min_cam_pos) * 3 - (cube_mean - mean_cam_pos)
                    print(f"Generating OUR point cloud ({num_pts})...")
            
                shs = np.random.random((num_pts, 3))
                pcd = BasicPointCloud(points=xyz, colors=shs, normals=np.zeros((num_pts, 3)))
                storePly(ply_path, xyz, SH2RGB(shs) * 255)
    elif not args_dict['render_only'] and args_dict['box_gen']:
        print('generating point cloud in a box')
        # box_center = np.array([-0.632, 0.592, 2.72])
        # box_size = np.array([0.318, 0.478, 0.738])
        # box_rotation = np.radians([-41.176, -48.239, -37.687])
        box_path = os.path.join(path, f"sparse/0/{args_dict['box_name']}_{mask_id:03}.txt")
        print(f'loading box from {box_path}')
        box_center, box_rotation, box_size, num_points = read_box(box_path)

        points_local = generate_points_in_box(box_size, num_points)
        xyz = transform_points_local_to_global(points_local, box_center, box_rotation)
        shs = np.random.random((num_points, 3))
        if custom_ply_path is not None:
            ply_path = custom_ply_path
        else:
            ply_path = os.path.join(path, "sparse/0/points3D.ply")
        print(f'storing in {ply_path}')
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    elif not args_dict['render_only'] and args_dict['use_orig']:
        if custom_ply_path is not None:
            ply_path = custom_ply_path
        else:
            ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")

        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            print(f'storing in {ply_path}')
            storePly(ply_path, xyz, rgb)
    else:
        ply_path = os.path.join(path, "result.ply")
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1

            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
