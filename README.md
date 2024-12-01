Activating conda env: 
`source /viscam/u/sharonal/miniconda3/bin/activate /viscam/u/sharonal/miniconda3/envs/rain`

Run code from `/viscam/projects/image2Blender/RAIN-GS`

Running the sanity check (single chair), results can be found in `/viscam/projects/image2Blender/RAIN-GS/output/<exp_name>`:
`python train_assembly_single.py -s sugar/imgs/11_20_synth2_chair/ --exp_name <exp_name> --eval --box_gen --box_name box --ours_new --num_cams 300 --save_iterations 1 3000 --iterations 3000 --input_pcs output/11_20_phase_1_synth2_chair/point_cloud_0/iteration_7000/point_cloud.ply --min_translation 1000 --max_translation 2000`

Running on whole scene:
`python train_assembly.py -s sugar/imgs/11_20_synth2/ --exp_name <exp_name> --eval --box_gen --box_name box --ours_new --num_cams 300 --save_iterations 1 3000 --iterations 3000 --input_pcs output/11_17-phase_1_scene_table/point_cloud_0/iteration_7000/point_cloud.ply output/11_20_phase_1_synth2_chair/point_cloud_0/iteration_7000/point_cloud.ply --min_translation 1000 --max_translation 2000  --translation_lr 0.0005`

Key files:
- sanity check code: train_assembly_single.py
- run with whole scene code: train_assembly.py
- render code: gaussian_renderer\__init__.py
- initialization of parameters and optimizers: scene\gaussian_model.py (training_setup function)
