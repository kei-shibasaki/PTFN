GPU_ID=4
CONFIG="experiments/ptfn_ahead_k7_resume/config_ptfn.json"
noise_levels="50"
CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval_codes.generate_images_davis2 -nl $noise_levels -c $CONFIG #--not_generae_inter_img
python -m eval_codes.evaluation -nl $noise_levels -c $CONFIG
#CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval_codes.generate_images_set82 -nl $noise_levels -c $CONFIG #--not_generae_inter_img
#python -m eval_codes.evaluation -nl $noise_levels --set8 -c $CONFIG