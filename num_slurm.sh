#!/bin/bash

# vals=("bouquet" "dozer_nerfgun_waldo" "espresso" "figurines" "donuts" "fruit_aisle" "shoe_rack" "ramen" "teatime")
# for val in "${vals[@]}"; do
#     python -m tu.sbatch.sbatch_sweep --time 2:00:00 \
#     --proj_dir /viscam/projects/image2Blender/garfield --conda_env nerfstudio \
#     --job "${val}" --command "ns-train garfield --data /viscam/projects/image2Blender/nerfstudio_images/${val} --viewer.make-share-url True" $GPU_INFO
# done

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/sugar/gaussian_splatting --conda_env sugar \
#     --job "colmap-bookcase" --command "./multi_run.sh bookcase" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/sugar --conda_env sugar2 \
#     --job "sugar_bookshelf" --command "python gaussian_splatting/train.py -s gaussian_splatting/imgs/bookshelf_8_new/ --iterations 7000 -m results/bookshelf_8_new/ && python train.py -s  gaussian_splatting/imgs/bookshelf_8_new/ -c results/bookshelf_8_new/ -r density
# " --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/sugar --conda_env sugar2 \
#     --job "sugar-worktable_21" --command "python gaussian_splatting/train.py -s gaussian_splatting/imgs/worktable_21/ --iterations 7000 -m results/worktable_21/ && python train.py -s  gaussian_splatting/imgs/worktable_21/ -c results/worktable_21/ -r density
# " --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/sugar --conda_env sugar2 \
#     --job "sugar-fridge_36" --command "python gaussian_splatting/train.py -s gaussian_splatting/imgs/fridge_36/ --iterations 7000 -m results/fridge_36/ && python train.py -s  gaussian_splatting/imgs/fridge_36/ -c results/fridge_36/ -r density
# " --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/sugar --conda_env sugar2 \
#     --job "sugar-bookcase_79" --command "python gaussian_splatting/train.py -s gaussian_splatting/imgs/bookcase_79/ --iterations 7000 -m results/bookcase_79/ && python train.py -s  gaussian_splatting/imgs/bookcase_79/ -c results/bookcase_79/ -r density
# " --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G


# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "mult_brick_250" --command "python train.py -s input_imgs/multi_3/ --exp_name multi_brick_with250 --eval --ours_new --num_cams 250 --save_iterations 0 1000 5000 10000 20000 30000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "mult_brick_200" --command "python train.py -s input_imgs/multi_3/ --exp_name multi_brick_with200 --eval --ours_new --num_cams 200 --save_iterations 0 1000 5000 10000 20000 30000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "mult_brick_normal" --command "python train.py -s input_imgs/multi_3/ --exp_name multi_brick_normal --eval --ours_new --save_iterations 0 1000 5000 10000 20000 30000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "dresser_250" --command "python train.py -s input_imgs/dresser6_72/ --exp_name dresser_with250 --eval --ours_new --num_cams 250 --save_iterations 0 1000 5000 10000 20000 30000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "dresser_200" --command "python train.py -s input_imgs/dresser6_72/ --exp_name dresser_with200 --eval --ours_new --num_cams 200 --save_iterations 0 1000 5000 10000 20000 30000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "dresser_normal" --command "python train.py -s input_imgs/dresser6_72/ --exp_name dresser_normal --eval --ours_new --save_iterations 0 1000 5000 10000 20000 30000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# # --

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "multi_brick_cam10_iter7k" --command "python train.py -s input_imgs/multi_3/ --exp_name multi_brick_cam10_iter7k --eval --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "dresser_brick_cam10_iter7k" --command "python train.py -s input_imgs/dresser3/ --exp_name dresser_brick_cam10_iter7k --eval --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "bookshelf_cam10_iter7k" --command "python train.py -s input_imgs/bookshelf_8/ --exp_name bookshelf_cam10_iter7k --eval --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "worktable_cam10_iter7k" --command "python train.py -s input_imgs/worktable_21/ --exp_name worktable_cam10_iter7k --eval --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "bookcase_cam10_iter7k" --command "python train.py -s input_imgs/bookcase_79/ --exp_name bookcase_cam10_iter7k --eval --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "fridge_cam10_iter7k" --command "python train.py -s input_imgs/fridge_67/ --exp_name fridge_cam10_iter7k --eval --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# # -- sugar, brick bg

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS/sugar --conda_env sugar2 \
#     --job "sugar-dresser3" --command "python train.py -s ../input_imgs/dresser3/ -c ../output/dresser_brick_cam10_iter7k/ -r density --num_cams 10" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS/sugar --conda_env sugar2 \
#     --job "sugar-multi_3" --command "python train.py -s ../input_imgs/multi_3/ -c ../output/multi_brick_cam10_iter7k/ -r density --num_cams 10" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# # --- rain for sugar, rand bg

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "09_08-dresser-cam10_iter7k" --command "python train.py -s sugar/imgs/dresser_2/ --exp_name 09_08-dresser-cam10_iter7k_2 --eval --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "09_08-bookcase-cam10_iter7k" --command "python train.py -s sugar/imgs/bookcase/ --exp_name 09_08-bookcase-cam10_iter7k --eval --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "09_08-bookshelf-cam10_iter7k" --command "python train.py -s sugar/imgs/bookshelf_59/ --exp_name 09_08-bookshelf-cam10_iter7k --eval --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "09_08-sofa-cam10_iter7k" --command "python train.py -s sugar/imgs/sofa_7/ --exp_name 09_08-sofa-cam10_iter7k --eval --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "09_08-worktable-cam10_iter7k" --command "python train.py -s sugar/imgs/worktable_24/ --exp_name 09_08-worktable-cam10_iter7k --eval --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# # --- sugar, rand bg

# # ran this already
# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS/sugar --conda_env sugar2 \
#     --job "09_08-bookcase-sugar" --command "python train.py -s imgs/bookcase/ -c ../output/09_08-bookcase-cam10_iter7k/ -r density --num_cams 10" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS/sugar --conda_env sugar2 \
#     --job "09_08-dresser-sugar" --command "python train.py -s imgs/dresser_2/ -c ../output/09_08-dresser-cam10_iter7k_2/ -r density --num_cams 10" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS/sugar --conda_env sugar2 \
#     --job "09_08-bookshelf-sugar" --command "python train.py -s imgs/bookshelf_59/ -c ../output/09_08-bookshelf-cam10_iter7k/ -r density --num_cams 10" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS/sugar --conda_env sugar2 \
#     --job "09_08-sofa-sugar" --command "python train.py -s imgs/sofa_7/ -c ../output/09_08-sofa-cam10_iter7k/ -r density --num_cams 10" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS/sugar --conda_env sugar2 \
#     --job "09_08-worktable-sugar" --command "python train.py -s imgs/worktable_24/ -c ../output/09_08-worktable-cam10_iter7k/ -r density --num_cams 10" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# # ---

# cp -r sugar/renamed_sugar_results/multi_3/iteration_15000/ output/multi_brick_cam10_iter7k/point_cloud

# python render.py -m output/multi_brick_cam10_iter7k  --ours_new

# python render.py -m output/multi_brick_normal  --ours_new --iteration 30000

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "render_brick1" --command "python render.py -m output/multi_brick_cam10_iter7k  --ours_new" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "render_brick2" --command "python render.py -m output/multi_brick_with10  --ours_new --iteration 30000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "render_brick3" --command "python render.py -m output/multi_brick_with200  --ours_new --iteration 30000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# # --- box gen
# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "09_14-box_gen-dresser-cam10_iter7k" --command "python train.py -s sugar/imgs/dresser_box_gen/ --exp_name 09_14-box_gen-dresser-cam10_iter7k --eval --box_gen --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G #NOTE: has no --bg

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "09_14-orig_gen_bg-dresser-cam10_iter7k" --command "python train.py -s sugar/imgs/dresser_box_gen/ --exp_name 09_14-orig_gen_bg-dresser-cam10_iter7k --eval --use_orig --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000 --bg" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# boxes=("box_big" "box_small" "box_left" "box_right" "box_up" "box_left_big" "box_left_small" "box_right_big" "box_right_small")
# for item in "${boxes[@]}"; do
#     python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#         --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#         --job "09_14-${item}_gen_bg-dresser-cam10_iter7k" --command "python train.py -s sugar/imgs/dresser_box_gen/ --exp_name 09_14-${item}_gen_bg-dresser-cam10_iter7k --eval --box_gen --box_name ${item} --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000 --bg" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G
# done

# boxes=("box" "box_big" "box_small" "box_left" "box_right" "box_up" "box_left_big" "box_left_small" "box_right_big" "box_right_small")
# for item in "${boxes[@]}"; do
#     python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS/sugar --conda_env sugar2 \
#     --job "09_14-${item}_gen_bg-dresser-cam10_iter7k-sugar" --command "python train.py -s imgs/dresser_box_gen/ -c ../output/09_14-${item}_gen_bg-dresser-cam10_iter7k/ -r density --num_cams 10" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G
# done

# boxes=("box" "box_big" "box_small" "box_left" "box_right" "box_up" "box_left_big" "box_left_small" "box_right_big" "box_right_small")
# for item in "${boxes[@]}"; do
#     python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "render-${item}" --command "cp sugar/output/refined_ply/09_14-${item}_gen_bg-dresser-cam10_iter7k/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface115000.ply output/09_14-${item}_gen_bg-dresser-cam10_iter7k/point_cloud/iteration_15000/point_cloud.ply && python render.py -m output/09_14-${item}_gen_bg-dresser-cam10_iter7k  --ours_new --skip_test --iteration 15000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G
# done

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "render" --command "python render.py -m output/09_14-box_gen-dresser-cam10_iter7k  --ours_new --skip_test" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# # -- instantmesh
# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS/sugar --conda_env sugar \
#     --job "colmap" --command "./multi_run.sh instantmesh-dresser" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# # manually made box coords - no guarantee it matches the previous thing
# # since we use different colmap.
# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "09_15-box_gen_bg-instantmesh_dresser-cam10_iter7k" --command "python train.py -s sugar/imgs/instantmesh-dresser_1/ --exp_name 09_15-box_gen_bg-instantmesh_dresser-cam10_iter7k --eval --box_gen --box_name box --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000 --bg" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS/sugar --conda_env sugar2 \
#     --job "09_15-box_gen_bg-instantmesh_dresser-cam10_iter7k-sugar" --command "python train.py -s imgs/instantmesh-dresser_1/ -c ../output/09_15-box_gen_bg-instantmesh_dresser-cam10_iter7k/ -r density --num_cams 10" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "render" --command "python render.py -m output/09_15-box_gen_bg-instantmesh_dresser-cam10_iter7k  --ours_new --skip_test" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "render-" --command "mkdir -p output/09_15-box_gen_bg-instantmesh_dresser-cam10_iter7k/point_cloud/iteration_15000 && cp sugar/output/refined_ply/09_15-box_gen_bg-instantmesh_dresser-cam10_iter7k/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface115000.ply output/09_15-box_gen_bg-instantmesh_dresser-cam10_iter7k/point_cloud/iteration_15000/point_cloud.ply && python render.py -m output/09_15-box_gen_bg-instantmesh_dresser-cam10_iter7k  --ours_new --skip_test --iteration 15000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# cams=(20 30 300)
# for item in "${cams[@]}"; do
#     # python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     # --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     # --job "09_16-box_gen_bg-instantmesh_dresser-cam${item}_iter7k" --command "python train.py -s sugar/imgs/instantmesh-dresser_1/ --exp_name 09_16-box_gen_bg-instantmesh_dresser-cam${item}_iter7k --eval --box_gen --box_name box --ours_new --num_cams ${item} --save_iterations 0 1000 3500 7000 --iterations 7000 --bg" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

#     # python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     # --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     # --job "render-09_16-box_gen_bg-instantmesh_dresser-cam${item}_iter7k" --command "python render.py -m output/09_16-box_gen_bg-instantmesh_dresser-cam${item}_iter7k  --ours_new --skip_test" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

#     # python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     # --proj_dir /viscam/projects/image2Blender/RAIN-GS/sugar --conda_env sugar2 \
#     # --job "09_16-box_gen_bg-instantmesh_dresser-cam${item}_iter7k-sugar" --command "python train.py -s imgs/instantmesh-dresser_1/ -c ../output/09_16-box_gen_bg-instantmesh_dresser-cam${item}_iter7k/ -r density --num_cams 10" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

#     python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "render-${item}" --command "mkdir -p output/09_16-box_gen_bg-instantmesh_dresser-cam${item}_iter7k/point_cloud/iteration_15000 && cp sugar/output/refined_ply/09_16-box_gen_bg-instantmesh_dresser-cam${item}_iter7k/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface115000.ply output/09_16-box_gen_bg-instantmesh_dresser-cam${item}_iter7k/point_cloud/iteration_15000/point_cloud.ply && python render.py -m output/09_16-box_gen_bg-instantmesh_dresser-cam${item}_iter7k  --ours_new --skip_test --iteration 15000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G
# done

# # --- 5 masks
# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "09_22-5_mask_box_gen_bg-dresser-cam10_iter7k" --command "python train.py -s sugar/imgs/dresser_5_masks/ --exp_name 09_22-5_mask_box_gen_bg-dresser-cam10_iter7k --eval --box_gen --box_name box --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000 --bg --num_masks 5" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "render" --command "python render.py -m output/09_22-5_mask_box_gen_bg-dresser-cam10_iter7k  --ours_new --skip_test" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "09_22-manual_5_mask_box_gen_bg-dresser-cam10_iter7k" --command "python train.py -s sugar/imgs/dresser_5_masks/ --exp_name 09_22-manual_5_mask_box_gen_bg-dresser-cam10_iter7k --eval --box_gen --box_name box --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000 --bg --num_masks 5" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "render" --command "python render.py -m output/09_22-manual_5_mask_box_gen_bg-dresser-cam10_iter7k  --ours_new --skip_test" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#     --job "09_23-full_manual_5_mask_box_gen_bg-dresser-cam10_iter7k-2" --command "python train.py -s sugar/imgs/dresser_5_masks_with_full/ --exp_name 09_23-full_manual_5_mask_box_gen_bg-dresser-cam10_iter7k-2 --eval --box_gen --box_name box --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000 --bg --num_masks 5" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

# cams=("point_cloud" "point_cloud_0" "point_cloud_1" "point_cloud_2" "point_cloud_3" "point_cloud_4")
# for item in "${cams[@]}"; do
#     python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#         --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#         --job "render-${item}" --command "python render.py -m output/09_23-full_manual_5_mask_box_gen_bg-dresser-cam10_iter7k-2  --ours_new --skip_test --render_source ${item}" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G
# done

# # with full and compare loss
# lambda=(0.1 0.5 1.0 1.5)
# for la in "${lambda[@]}"; do
#     iters=(0 500 1000 4000)
#     for it in "${iters[@]}"; do
#         python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#             --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#             --job "09_29-full_compare_5_mask_box_gen_bg-dresser-cam10_iter7k-comp${la}-cit${it}" --command "python train.py -s sugar/imgs/dresser_5_masks_with_full/ --exp_name 09_29-full_compare_5_mask_box_gen_bg-dresser-cam10_iter7k-comp${la}-cit${it} --eval --box_gen --box_name box --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 --iterations 7000 --bg --num_masks 5 --lambda_compare ${la} --compare_iter ${it}" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G
#     done
# done

# lambda=(0.1 0.5 1.0 1.5)
# for la in "${lambda[@]}"; do
#     iters=(0 500 1000 4000)
#     for it in "${iters[@]}"; do
#         cams=("point_cloud" "point_cloud_0" "point_cloud_1" "point_cloud_2" "point_cloud_3" "point_cloud_4")
#             for item in "${cams[@]}"; do
#                 python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
#                     --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
#                     --job "render-comp${la}-cit${it}-${item}" --command "python render.py -m output/09_29-full_compare_5_mask_box_gen_bg-dresser-cam10_iter7k-comp${la}-cit${it}  --ours_new --skip_test --render_source ${item}" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G
#             done
#     done
# done

# experiments on applying mask to rendered image during second phase (removed the bg argument, it's redundant) - assumed to be full_compare_5_mask, dresser-cam10_iter7k
lambda=(0.5 1.0 0.1 2.0 10.0)
for la in "${lambda[@]}"; do
    python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
        --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
        --job "10_02-2phase_maskall_nomaskfull-comp${la}" --command "python train.py -s sugar/imgs/dresser_5_masks_with_full/ --exp_name 10_02-2phase_maskall_nomaskfull-comp${la} --eval --box_gen --box_name box --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 8000 10000 12000 14000 16000 --iterations 16000 --num_masks 5 --lambda_compare ${la} --second_phase_begin 7000" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

    python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
        --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
        --job "10_02-2phase_maskcomponly_nomaskfull-comp${la}" --command "python train.py -s sugar/imgs/dresser_5_masks_with_full/ --exp_name 10_02-2phase_maskcomponly_nomaskfull-comp${la} --eval --box_gen --box_name box --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 8000 10000 12000 14000 16000 --iterations 16000 --num_masks 5 --lambda_compare ${la} --second_phase_begin 7000 --mask_on_compare_only" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

    python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
        --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
        --job "10_02-2phase_maskall_maskfull-comp${la}" --command "python train.py -s sugar/imgs/dresser_5_masks_with_full/ --exp_name 10_02-2phase_maskall_maskfull-comp${la} --eval --box_gen --box_name box --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 8000 10000 12000 14000 16000 --iterations 16000 --num_masks 5 --lambda_compare ${la} --second_phase_begin 7000 --mask_to_full" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

    python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
        --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
        --job "10_02-2phase_maskcomponly_maskfull-comp${la}" --command "python train.py -s sugar/imgs/dresser_5_masks_with_full/ --exp_name 10_02-2phase_maskcomponly_maskfull-comp${la} --eval --box_gen --box_name box --ours_new --num_cams 10 --save_iterations 0 1000 3500 7000 8000 10000 12000 14000 16000 --iterations 16000 --num_masks 5 --lambda_compare ${la} --second_phase_begin 7000 --mask_on_compare_only --mask_to_full" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G
done

lambda=(0.5 1.0 0.1 2.0 10.0)
for la in "${lambda[@]}"; do
    iters=(7000 8000 10000 12000 14000 16000)
    for it in "${iters[@]}"; do
        cams=("point_cloud" "point_cloud_0" "point_cloud_1" "point_cloud_2" "point_cloud_3" "point_cloud_4")
        for item in "${cams[@]}"; do
            python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
                --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
                --job "render-10_02-2phase_maskall_nomaskfull-comp${la}-iter${it}" --command "python render.py -m output/10_02-2phase_maskall_nomaskfull-comp${la}  --ours_new --skip_test --render_source ${item} --iteration ${it}" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

            python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
                --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
                --job "render-10_02-2phase_maskcomponly_nomaskfull-comp${la}-iter${it}" --command "python render.py -m output/10_02-2phase_maskcomponly_nomaskfull-comp${la}  --ours_new --skip_test --render_source ${item} --iteration ${it}" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

            python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
                --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
                --job "render-10_02-2phase_maskall_maskfull-comp${la}-iter${it}" --command "python render.py -m output/10_02-2phase_maskall_maskfull-comp${la}  --ours_new --skip_test --render_source ${item} --iteration ${it}" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

            python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
                --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
                --job "render-10_02-2phase_maskcomponly_maskfull-comp${la}-iter${it}" --command "python render.py -m output/10_02-2phase_maskcomponly_maskfull-comp${la}  --ours_new --skip_test --render_source ${item} --iteration ${it}" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G
        done
    done
done