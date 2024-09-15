import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import imageio
import numpy as np
import cv2

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    sample = render(views[0], gaussians, pipeline, background)
    fps = 10
    height, width, _ = sample["depth"].shape
    depth_video = cv2.VideoWriter(os.path.join("videos", f"{os.path.basename(render_path)}_depth.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    height, width, _ = sample["render"].shape
    rgb_video = cv2.VideoWriter(os.path.join("videos", f"{os.path.basename(render_path)}_rgb.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)
        rendered_image, rendered_depth = rendering["render"], rendering["depth"]
        gt = view.original_image[0:3, :, :]
        render_depth = rendered_depth.clone()
        rendered_depth = (rendered_depth-rendered_depth.min()) / (rendered_depth.max() - rendered_depth.min() + 1e-6)
        
        
        render_depth = render_depth.permute(1, 2, 0).squeeze()
        normalizer = mpl.colors.Normalize(vmin=render_depth.min(), vmax=np.percentile(render_depth.cpu().numpy(), 95))
        
        inferno_mapper = cm.ScalarMappable(norm=normalizer,cmap="inferno")
        colormap_inferno = (inferno_mapper.to_rgba(render_depth.cpu().numpy())*255).astype('uint8') 
        depth_video.write(cv2.imread(colormap_inferno))
        rgb_video.write(cv2.imread(rendered_image))
    cv2.destroyAllWindows()
    depth_video.release()
    rgb_video.release()
        # imageio.imwrite(os.path.join(render_path, '{0:05d}'.format(idx) + "_depth_inferno.png"), colormap_inferno)
        
        # torchvision.utils.save_image(rendered_image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(rendered_depth, os.path.join(render_path, '{0:05d}'.format(idx) + "_depth.png"))
        #torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args_dict):
    with torch.no_grad():
        if args_dict['ours'] or args_dict['ours_new']:
            divide_ratio = 0.7
        else:
            divide_ratio = 0.8
        print(f"Set divide_ratio to {divide_ratio}")
        gaussians = GaussianModel(dataset.sh_degree, divide_ratio)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, args_dict=args_dict)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--num_cams", default=10, type=int)
    parser.add_argument("--ours_new", action="store_true", help="Use our initialisation version 2")
    parser.add_argument("--ours", action="store_true", help="Use our initialisation version 2")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    args.render_only = True # NB render_only puts everything in training set
    
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.__dict__)
