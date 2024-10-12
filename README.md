When using our cluster, different GPUs have different CUDA versions. A6000s have the CUDA version we require, so we will use an interactive session for setup:
```
srun -A viscam -p viscam-interactive --gres gpu:a6000:1 --cpus-per-task 1 --mem 30G --time 1:00:00 --pty bash
conda env create --file environment.yml
conda activate gaussian_splatting
```
