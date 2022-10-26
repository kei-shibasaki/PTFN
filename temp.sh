CUDA_VISIBLE_DEVICES=4 python temp.py
CUDA_VISIBLE_DEVICES=5 python temp.py
CUDA_VISIBLE_DEVICES=6 python temp.py
CUDA_VISIBLE_DEVICES=7 python temp.py
CUDA_VISIBLE_DEVICES=3 python temp.py

CUDA_VISIBLE_DEVICES=4 python -m train_codes.train -c config/config_unet.json
CUDA_VISIBLE_DEVICES=5 python -m train_codes.train -c config/config_unet.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.trainA -c config/config_A.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train2 -c config/config_test.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.train3 -c config/config_test2.json
CUDA_VISIBLE_DEVICES=6 python -m train_codes.train2 -c config/config_test.json
CUDA_VISIBLE_DEVICES=7 python -m train_codes.trainEx -c config/config_Ex.json
CUDA_VISIBLE_DEVICES=6 python -m train_codes.trainEx2 -c config/config_Ex.json

CUDA_VISIBLE_DEVICES=6 python -m eval_codes.generate_images -c experiments/fastdvd2/config_test3.json
CUDA_VISIBLE_DEVICES=7 python -m eval_codes.generate_images -c experiments/fastdvdM/config_M.json
CUDA_VISIBLE_DEVICES=1 python -m eval_codes.gen_temp -c experiments/fastdvd_level5/config_test2.json


python -m eval_codes.visualize_losses -c experiments/extreme/config_Ex.json
python -m eval_codes.visualize_losses -c experiments/fastdvd2/config_test3.json
python -m eval_codes.evaluation -c experiments/fastdvd2/config_test3.json
python -m eval_codes.evaluation -c experiments/fastdvdM/config_M.json


CUDA_VISIBLE_DEVICES=3 python calc_speed.py