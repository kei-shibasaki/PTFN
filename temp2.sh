CUDA_VISIBLE_DEVICES=7 python -m eval_codes.generate_images_davis -nl 50 -c experiments/ptfn_fullsca_b8/config_test.json
python -m eval_codes.evaluation -nl 50 -c experiments/ptfn_fullsca_b8/config_test.json
CUDA_VISIBLE_DEVICES=7 python -m eval_codes.generate_images_set8 -nl 50 -c experiments/ptfn_fullsca_b8/config_test.json
python -m eval_codes.evaluation -nl 50 --set8 -c experiments/ptfn_fullsca_b8/config_test.json