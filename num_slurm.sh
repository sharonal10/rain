#!/bin/bash

python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
    --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
    --job "11_28_phase_2_assembly_synth2" --command "python train_assembly.py -s sugar/imgs/11_20_synth2/ --exp_name 11_28_phase_2_assembly_synth --eval --box_gen --box_name box --ours_new --num_cams 300 --save_iterations 1 1000 3500 7000 --iterations 7000 --input_pcs output/11_17-phase_1_scene_table/point_cloud_0/iteration_7000/point_cloud.ply output/11_20_phase_1_synth2_chair/point_cloud_0/iteration_7000/point_cloud.ply" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

    python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
    --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
    --job "11_28_sanity_check_synth2_chair" --command "python train_assembly_single.py -s sugar/imgs/11_20_synth2_chair/ --exp_name 11_28_sanity_check_synth2_chair2 --eval --box_gen --box_name box --ours_new --num_cams 300 --save_iterations 1 1000 3500 7000 --iterations 7000 --input_pcs output/11_20_phase_1_synth2_chair/point_cloud_0/iteration_7000/point_cloud.ply" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G

pairs=(
    "3000 6000"
    "3000 9000"
    "3000 12000"
    "6000 9000"
    "6000 12000"
    "9000 12000"
)
for pair in "${pairs[@]}"; do
    a=$(echo $pair | cut -d' ' -f1)
    b=$(echo $pair | cut -d' ' -f2)
    underscore_pair="${a}_${b}"
    python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
    --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
    --job "11_28_sanity_check_synth2_chair_${underscore_pair}" --command "python train_assembly_single.py -s sugar/imgs/11_20_synth2_chair/ --exp_name 11_28_sanity_check_synth2_chair_${underscore_pair} --eval --box_gen --box_name box --ours_new --num_cams 300 --save_iterations 1 15000 --iterations 15000 --input_pcs output/11_20_phase_1_synth2_chair/point_cloud_0/iteration_7000/point_cloud.ply --lt_translation ${a} --gt_translation ${b}" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G
done

    python -m tu.sbatch.sbatch_sweep --time 24:00:00 \
    --proj_dir /viscam/projects/image2Blender/RAIN-GS --conda_env rain \
    --job "11_24_train_but_render" --command "python train_but_render.py -s sugar/imgs/11_20_synth2/ --exp_name 11_26_train_but_render --eval --box_gen --box_name box --ours_new --num_cams 300 --save_iterations 1 1000 3500 7000 --iterations 7000 --input_pcs output/11_17-phase_1_scene_table/point_cloud_0/iteration_7000/point_cloud.ply output/11_20_phase_1_synth2_chair/point_cloud_0/iteration_7000/point_cloud.ply" --partition viscam --account viscam --gpu_type a6000 --cpus_per_task 8 --num_gpus 1 --mem 64G