GPU_ID=6
CONFIG="experiments/ptfn-L_blind/config_ptfn2.json"
noise_levels="10 20 30 40 50"
CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval_codes.generate_images_davis_blind -nl $noise_levels -c $CONFIG #--not_generae_inter_img
python -m eval_codes.evaluation -nl $noise_levels -c $CONFIG
CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval_codes.generate_images_set8_blind -nl $noise_levels -c $CONFIG #--not_generae_inter_img
python -m eval_codes.evaluation -nl $noise_levels --set8 -c $CONFIG