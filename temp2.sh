CUDA_VISIBLE_DEVICES=3 python -m eval_codes.generate_images_naf -c experiments/naf_multi/config_test.json
#CUDA_VISIBLE_DEVICES=3 python -m eval_codes.generate_images_naf2 -c experiments/naf_multi/config_test.json
python -m eval_codes.evaluation -c experiments/naf_multi/config_test.json
#python -m eval_codes.evaluation -c experiments/extreme2/config_Ex.json
