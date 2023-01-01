GPU_ID=7
CONFIG="experiments/ptfn_v4_b16/config_test.json"
noise_levels="10 20 30 40 50"
CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval_codes.generate_images_davis -nl $noise_levels -c $CONFIG
python -m eval_codes.evaluation -nl $noise_levels -c $CONFIG
CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval_codes.generate_images_set8 -nl $noise_levels -c $CONFIG
python -m eval_codes.evaluation -nl $noise_levels --set8 -c $CONFIG