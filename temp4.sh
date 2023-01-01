GPU_ID=0
CONFIG="UNUSED/config_wnet.json"
#CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval_codes.generate_images_bsvd -c $CONFIG
python -m eval_codes.evaluation_notinter -nl 10 20 30 40 50 -c $CONFIG
#CUDA_VISIBLE_DEVICES=$GPU_ID python -m eval_codes.generate_images_bsvd_set8 -c $CONFIG
#python -m eval_codes.evaluation -nl 10 20 30 40 50 --set8 -c $CONFIG