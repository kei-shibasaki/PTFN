CUDA_VISIBLE_DEVICES=7 python -m eval_codes.generate_images_ex1 -c experiments/extreme/config_Ex.json
CUDA_VISIBLE_DEVICES=7 python -m eval_codes.generate_images_ex2 -c experiments/extreme2/config_Ex.json
python -m eval_codes.evaluation -c experiments/extreme/config_Ex.json
python -m eval_codes.evaluation -c experiments/extreme2/config_Ex.json
