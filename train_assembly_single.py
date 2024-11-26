import os
import torch
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, render_multi
import sys
from scene import Scene, GaussianModel
from scene.gaussian_model import rotation_matrix_to_quaternion, rotate_quaternions, rotate_around_z
from scene.dataset_readers import read_box
from utils.general_utils import safe_state
import uuid
import numpy as np
import wandb
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from lpipsPyTorch import lpips
from plyfile import PlyData, PlyElement
from PIL import Image
from utils.general_utils import PILtoTorch
import cv2
import glob
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
# assembly script
# 1) load required files
# 2) duplicate as needed
# 3) optimize only rotation, scale, and offset, do not otherwise optimize gaussians
def training(dataset, opt, pipe, testing_iterations ,saving_iterations, checkpoint_iterations ,checkpoint, debug_from, args_dict):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, args_dict['output_path'], args_dict['exp_name'], args_dict['project_name'])
    
    if args_dict['ours'] or args_dict['ours_new']:
        divide_ratio = 0.7
    else:
        divide_ratio = 0.8
    print(f"Set divide_ratio to {divide_ratio}")

    # gather centers (hardcoded for now)
    # note: 'centers' and 'offsets' are used interchangably
    # boxes_to_load = [1, 2, 3]
    # raw_centers = []
    # for box_id in boxes_to_load:
    #     box_path = os.path.join(dataset.source_path, f"sparse/0/{args_dict['box_name']}_{box_id:03}.txt")
    #     if os.path.exists(box_path):
    #         print(f'loading box (center only) from {box_path}')
    #         box_center, box_rotation, box_size, num_points = read_box(box_path)
    #         raw_centers.append(box_center)
    #     else:
    #         raw_centers.append(np.array([0, 0, 0]))

    
    gaussians_list = []
    scene_list = []
    assembly_sources = { # hardcode for this experiment
        0: args_dict['input_pcs'][0],
        # 3: args_dict['input_pcs'][1],
    }

    sam_mask_to_load = {
        0: [0],
        # 3: [1, 2, 3],    
    }
    for mask_id in [0]: # 0 = table, 3 = chair, which is duplicated to make 1 & 2
        gaussians = GaussianModel(dataset.sh_degree, divide_ratio, mask_id=mask_id, assembly=True)
        scene = Scene(dataset, gaussians, args_dict=args_dict, mask_id=mask_id, assembly_source=assembly_sources[mask_id], sam_mask_to_load=sam_mask_to_load[mask_id])
        if mask_id == 0:
            gaussians.training_setup(opt, [np.array([0, 0, 0])]) 
        else:
            assert False
            # gaussians.training_setup(opt, raw_centers) 
        gaussians_list.append(gaussians)
        scene_list.append(scene)
    
    if args_dict["warmup_iter"] > 0:
        opt.densify_until_iter += args_dict["warmup_iter"]
        
    if checkpoint:
        assert False
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True)
    first_iter += 1

    scene_order = list(range(len(scene.getTrainCameras())))
    random.shuffle(scene_order)

    for iteration in range(first_iter, opt.iterations + 1):
        with torch.no_grad():
            if iteration % 100 == 0 or iteration < 100:
                display_cam_idx = 150
                display_cam = scene.getTrainCameras().copy()[display_cam_idx]
                display_render_pkg = render_multi(display_cam, gaussians_list, pipe, bg, low_pass = low_pass)
                display_image = display_render_pkg["render"]
                gt_display_image = display_cam.original_image.cuda()

                to_save_image = display_image.detach().permute(1, 2, 0).cpu().numpy()
                to_save_image = Image.fromarray((to_save_image * 255).astype(np.uint8))
                frames_folder = os.path.join(scene.model_path, "display")
                os.makedirs(frames_folder, exist_ok=True)
                to_save_image.save(os.path.join(frames_folder, f'{iteration:04d}.png'))

                to_save_image = gt_display_image.detach().permute(1, 2, 0).cpu().numpy()
                to_save_image = Image.fromarray((to_save_image * 255).astype(np.uint8))
                to_save_image.save(os.path.join(scene.model_path, f'gt_display.png'))
        if not viewpoint_stack:
            random.shuffle(scene_order)
            viewpoint_stack = scene_order.copy()
        viewpoint_idx = viewpoint_stack.pop()
        for sub_iter in range(len(gaussians_list)):
            gaussians = gaussians_list[sub_iter]
            scene = scene_list[sub_iter]
            viewpoint_cam = scene.getTrainCameras().copy()[viewpoint_idx]
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            iter_start.record()

            if args_dict['ours_new']:
                if iteration >= args_dict["warmup_iter"]:    
                    gaussians.update_learning_rate(iteration-args_dict["warmup_iter"])
            else:
                gaussians.update_learning_rate(iteration)

            if args_dict['ours'] or args_dict['ours_new']:
                if iteration >= 5000:
                    if iteration % 1000 == 0:
                        gaussians.oneupSHdegree()
            else:
                if iteration % 1000 == 0:
                    gaussians.oneupSHdegree()
            
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background
            c2f = args_dict['c2f']

            if c2f == True:
                if iteration == 1 or (iteration % args_dict['c2f_every_step'] == 0 and iteration < opt.densify_until_iter) :
                    H = viewpoint_cam.image_height
                    W = viewpoint_cam.image_width
                    N = gaussians.get_xyz.shape[0]
                    low_pass = max (H * W / N / (9 * np.pi), 0.3)
                    if args_dict['c2f_max_lowpass'] > 0:
                        low_pass = min(low_pass, args_dict['c2f_max_lowpass'])
                    print(f"[ITER {iteration}] Low pass filter : {low_pass}")
            else:
                low_pass = 0.3
                
            for center_id in list(range(len(gaussians.centers))):
                if viewpoint_cam.masks[center_id].cuda().sum() < 5:
                    if (iteration in saving_iterations):
                        with torch.no_grad():
                            print("\n[ITER {}] Saving Gaussians".format(iteration))
                            scene.save(iteration, center_id)
                    continue
                render_pkg = render(viewpoint_cam, gaussians, pipe, bg, low_pass = low_pass, center_id=center_id)
                image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]

                gt_image = viewpoint_cam.original_image.cuda()
                mask = viewpoint_cam.masks[center_id].cuda()
                masked_image = image # don't apply mask to rendered image
                # masked_image = image*mask
                masked_gt_image = gt_image*mask

                if iteration % 1000 == 0 or iteration < 4 or iteration == 50:
                    to_save_image = image.detach().permute(1, 2, 0).cpu().numpy()
                    to_save_image = Image.fromarray((to_save_image * 255).astype(np.uint8))
                    to_save_image.save(os.path.join(scene.model_path, f'part_{sub_iter}_{center_id}_{iteration}.png'))

                    to_save_image = masked_gt_image.detach().permute(1, 2, 0).cpu().numpy()
                    to_save_image = Image.fromarray((to_save_image * 255).astype(np.uint8))
                    to_save_image.save(os.path.join(scene.model_path, f'gt_{sub_iter}_{center_id}_{iteration}.png'))

                    to_save_image = mask.expand(3, -1, -1).detach().permute(1, 2, 0).cpu().numpy()
                    to_save_image = Image.fromarray((to_save_image * 255).astype(np.uint8))
                    to_save_image.save(os.path.join(scene.model_path, f'mask_{sub_iter}_{center_id}_{iteration}.png'))
                
                Ll1 = l1_loss(masked_image, masked_gt_image)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(masked_image, masked_gt_image))
                loss.backward()

                iter_end.record()

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "num_gaussians" : f"{gaussians.get_xyz.shape[0]}"})
                    progress_bar.update(5)
                if iteration == opt.iterations:
                    progress_bar.close()

                if iteration < opt.iterations:
                    gaussians.scale_optimizer.step()
                    gaussians.scale_optimizer.zero_grad(set_to_none = True)
                    for c_opt in gaussians.center_optimizers:
                        c_opt.step()
                        c_opt.zero_grad(set_to_none = True)
                    print('xyz.grad', gaussians.get_xyz.grad)
                    for ri, r in enumerate(gaussians.rot_vars):
                        print(ri, r.grad)
                    for r_opt in gaussians.rot_var_optimizers:
                        print('rot')
                        r_opt.step()
                        r_opt.zero_grad(set_to_none = True)
                
                # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration, center_id)

                # as the gaussians are frozen except for rotation, offset, and scale, disable densification and opacity

                # if iteration < opt.densify_until_iter:       
                #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                #         abe_split = True if iteration <= args_dict['warmup_iter'] else False
                        
                #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, N=2, abe_split=abe_split)         
                    
                #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #         gaussians.reset_opacity()

        # also backprop for all, I am assuming bg, pipe etc are fine if kept the same

        # render_pkg = render_multi(viewpoint_cam, gaussians_list, pipe, bg, low_pass = low_pass)
        # image = render_pkg["render"]

        # gt_image = viewpoint_cam.original_image.cuda()

        # mask = Image.open(os.path.join(dataset.source_path, 'full_masks', f'{viewpoint_cam.image_name}.png'))
        # mask = PILtoTorch(mask, (viewpoint_cam.image_width, viewpoint_cam.image_height)).cuda()
        
        # masked_image = image
        # # masked_image = image*mask
        # masked_gt_image = gt_image*mask

        # if iteration % 1000 == 0 or iteration < 4 or iteration == 50:
        #     to_save_image = image.detach().permute(1, 2, 0).cpu().numpy()
        #     to_save_image = Image.fromarray((to_save_image * 255).astype(np.uint8))
        #     to_save_image.save(os.path.join(scene.model_path, f'whole_{iteration}.png'))
            
        #     to_save_image = masked_gt_image.detach().permute(1, 2, 0).cpu().numpy()
        #     to_save_image = Image.fromarray((to_save_image * 255).astype(np.uint8))
        #     to_save_image.save(os.path.join(scene.model_path, f'gt_whole_{iteration}.png'))

        #     to_save_image = mask.expand(3, -1, -1).detach().permute(1, 2, 0).cpu().numpy()
        #     to_save_image = Image.fromarray((to_save_image * 255).astype(np.uint8))
        #     to_save_image.save(os.path.join(scene.model_path, f'mask_whole_{iteration}.png'))

        # Ll1 = l1_loss(masked_image, masked_gt_image)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(masked_image, masked_gt_image))
        # loss.backward()

        # for sub_iter in range(len(gaussians_list)):
        #     gaussians = gaussians_list[sub_iter]
        #     scene = scene_list[sub_iter]
        #     viewpoint_cam = scene.getTrainCameras().copy()[viewpoint_idx]

        #     with torch.no_grad():
        #         ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        #         if iteration % 10 == 0:
        #             progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "num_gaussians" : f"{gaussians.get_xyz.shape[0]}"})
        #             progress_bar.update(5)
        #         if iteration == opt.iterations:
        #             progress_bar.close()
                

        #         if iteration < opt.iterations:
        #             gaussians.scale_optimizer.step()
        #             gaussians.scale_optimizer.zero_grad(set_to_none = True)
        #             for c_opt in gaussians.center_optimizers:
        #                 c_opt.step()
        #                 c_opt.zero_grad(set_to_none = True)
        #             for r_opt in gaussians.rot_var_optimizers:
        #                 print('rot whole')
        #                 r_opt.step()
        #                 r_opt.zero_grad(set_to_none = True)
        #             print('--')
        #             print(sub_iter)
        #             print('scale', gaussians.scale)
        #             print('rot', gaussians.rot_vars)

        #         if (iteration in checkpoint_iterations):
        #             print("\n[ITER {}] Saving Checkpoint".format(iteration))
        #             torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        

        with torch.no_grad():

            if (iteration in saving_iterations):
                # save all together - based on save() function in scene/__init__.py
                point_cloud_path = os.path.join(dataset.model_path, "point_cloud/iteration_{}".format(iteration))
                os.makedirs(point_cloud_path, exist_ok = True)
                for pc in gaussians_list:
                    xyz = None
                    f_dc = None
                    f_rest = None
                    opacities = None
                    scale = None
                    rotation = None
                    first = True
                    for i, center in enumerate(pc.centers):
                        curr = pc.get_xyz + center
                        centroid = curr.mean(dim=0)
                        # center to origin then scale
                        curr, rotation_matrix = rotate_around_z(curr, pc.rot_vars[i], centroid)
                        rotation_quaternion = rotation_matrix_to_quaternion(rotation_matrix)
                        curr = ((curr - centroid) * pc.scale) + centroid
                        if first:
                            first = False
                            xyz = curr.detach().cpu().numpy()
                            f_dc = pc._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
                            f_rest = pc._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
                            opacities = pc._opacity.detach().cpu().numpy()
                            scale = (pc._scaling + torch.log(pc.scale)).detach().cpu().numpy()
                            rotation = rotate_quaternions(pc.get_rotation, rotation_quaternion).detach().cpu().numpy()
                        else:
                            xyz = np.concatenate((xyz, curr.detach().cpu().numpy()), axis=0)
                            f_dc = np.concatenate((f_dc, pc._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()), axis=0)
                            f_rest = np.concatenate((f_rest, pc._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()), axis=0)
                            opacities = np.concatenate((opacities, pc._opacity.detach().cpu().numpy()), axis=0)
                            scale = np.concatenate((scale, (pc._scaling + torch.log(pc.scale)).detach().cpu().numpy()), axis=0)
                            rotation = np.concatenate((rotation, rotate_quaternions(pc.get_rotation, rotation_quaternion).detach().cpu().numpy()), axis=0)

                normals = np.zeros_like(xyz)

                dtype_full = [(attribute, 'f4') for attribute in gaussians_list[0].construct_list_of_attributes()]

                elements = np.empty(xyz.shape[0], dtype=dtype_full)
                attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
                elements[:] = list(map(tuple, attributes))
                el = PlyElement.describe(elements, 'vertex')
                print(os.path.join(point_cloud_path, "point_cloud.ply"))
                PlyData([el]).write(os.path.join(point_cloud_path, "point_cloud.ply"))

    print('saving video')
    video_path = os.path.join(scene.model_path, "output_video.mp4")
    image_files = sorted(glob.glob(os.path.join(frames_folder, "*.png")))
    first_image = cv2.imread(image_files[0])
    frame_height, frame_width, _ = first_image.shape
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    for file in image_files:
        image = cv2.imread(file)
        video_writer.write(image)

    video_writer.release()
                    

def prepare_output_and_logger(args, output_path, exp_name, project_name):
    if (not args.model_path) and (not exp_name):
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    elif (not args.model_path) and exp_name:
        args.model_path = os.path.join("./output", exp_name) 
    
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, 'command_line.txt'), 'w') as file:
        file.write(' '.join(sys.argv))

    tb_writer = None   
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
        print("Logging progress to Tensorboard at {}".format(args.model_path))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                    ssim_test += ssim(image, gt_image)
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                lpips_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS(vgg) {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, lpips_test, ssim_test))
                with open(os.path.join(args.output_path, args.exp_name, 'log_file.txt'), 'a') as file:
                    file.write("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS(vgg) {} SSIM {}\n".format(iteration, config['name'], l1_test, psnr_test, lpips_test, ssim_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--output_path", type=str,default='./output/')
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="gaussian-splatting")
    parser.add_argument("--c2f", action="store_true", default=False)
    parser.add_argument("--c2f_every_step", type=float, default=1000, help="Recompute low pass filter size for every c2f_every_step iterations")
    parser.add_argument("--c2f_max_lowpass", type=float, default= 300, help="Maximum low pass filter size")
    parser.add_argument("--num_gaussians", type=int, default=1000000, help="Number of random initial gaussians to start with (default=1M for random)")
    parser.add_argument('--paper_random', action='store_true', help="Use the initialisation from the paper")
    parser.add_argument("--ours", action="store_true", help="Use our initialisation")
    parser.add_argument("--ours_new", action="store_true", help="Use our initialisation version 2")
    parser.add_argument("--warmup_iter", type=int, default=0)
    parser.add_argument("--train_from", type=str, default="random", choices=["random", "reprojection", "cluster", "noisy_sfm"])
    parser.add_argument('--num_cams', type=int, default=10)
    
    parser.add_argument("--box_gen", action="store_true", help="Use box_gen initialisation")
    parser.add_argument("--box_name", type=str, help="name of the .txt file with box params")
    parser.add_argument("--use_orig", action="store_true", help="Use box_gen initialisation")
    parser.add_argument("--input_pcs", nargs="+", type=str, required=True, help="paths to point clouds to load")
    
    # removed for now, will hardcode
    # parser.add_argument('--num_masks', type=int, required=True)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.white_background = args.white_bg
    args.render_only = False
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    
    args.eval = True
    outdoor_scenes=['bicycle', 'flowers', 'garden', 'stump', 'treehill']
    indoor_scenes=['room', 'counter', 'kitchen', 'bonsai']
    for scene in outdoor_scenes:
        if scene in args.source_path:
            args.images = "images_4"
            print("Using images_4 for outdoor scenes")
    for scene in indoor_scenes:
        if scene in args.source_path:
            args.images = "images_2"
            print("Using images_2 for indoor scenes")
    
    if args.ours or args.ours_new:
        print("========= USING OUR METHOD =========")
        args.c2f = True
        args.c2f_every_step = 1000
        args.c2f_max_lowpass = 300
        args.num_gaussians = 10
    if args.ours_new:
        args.warmup_iter = 10000

    if args.ours and (args.train_from != "random"):
        parser.error("Our initialization version 1 can only be used with --train_from random")
    
    print(f"args: {args}")
    
    while True :
        try:
            network_gui.init(args.ip, args.port)
            print(f"GUI server started at {args.ip}:{args.port}")
            break
        except Exception as e:
            args.port = args.port + 1
            print(f"Failed to start GUI server, retrying with port {args.port}...")
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations ,args.save_iterations, args.checkpoint_iterations ,args.start_checkpoint, args.debug_from, args.__dict__)

    
    print("\nTraining complete.")
