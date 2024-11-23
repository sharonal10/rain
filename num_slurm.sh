#!/bin/bash

python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
    --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
    --job "11_20_phase_2_assembly_synth2" --command "python train_assembly.py -s sugar/imgs/11_20_synth2/ --exp_name 11_20_phase_2_assembly_synth2 --eval --box_gen --box_name box --ours_new --num_cams 300 --save_iterations 1 1000 3500 7000 --iterations 7000 --input_pcs output/11_17-phase_1_scene_table/point_cloud_0/iteration_7000/point_cloud.ply output/11_20_phase_1_synth2_chair/point_cloud_0/iteration_7000/point_cloud.ply" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G