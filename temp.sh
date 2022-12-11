CUDA_VISIBLE_DEVICES=4 python temp.py
CUDA_VISIBLE_DEVICES=5 python temp.py
CUDA_VISIBLE_DEVICES=6 python temp.py
CUDA_VISIBLE_DEVICES=7 python temp.py
CUDA_VISIBLE_DEVICES=3 python temp.py


CUDA_VISIBLE_DEVICES=6,7 python -m train_codes.train_mimo_dp -c config/config_test3.json
CUDA_VISIBLE_DEVICES=3 python -m train_codes.train_mimo -c config/config_test3.json
CUDA_VISIBLE_DEVICES=5 python -m train_codes.train_mimo2 -c config/config_test.json
CUDA_VISIBLE_DEVICES=3 python -m train_codes.train_mimo3 -c 
CUDA_VISIBLE_DEVICES=3,2 python -m train_codes.train_mimo_dp2 -c config/config_test2.json
CUDA_VISIBLE_DEVICES=6,7 python -m train_codes.train_mimo_dp -c config/config_test_dp.json
CUDA_VISIBLE_DEVICES=2,3 python -m train_codes.train_wnet -c config/config_wnet.json
CUDA_VISIBLE_DEVICES=5 python -m train_codes.train -c config/config_test.json
CUDA_VISIBLE_DEVICES=2,3 python -m train_codes.train_dp -c config/config_test_dp.json


CUDA_VISIBLE_DEVICES=5 python -m eval_codes.generate_images_davis_multi -c experiments/naf_tsm_multi/config_test.json
CUDA_VISIBLE_DEVICES=4 python -m eval_codes.generate_images_davis -c experiments/naf_tsm_b16/config_test_dp.json
CUDA_VISIBLE_DEVICES=5 python -m eval_codes.generate_images_set8 -c experiments/naf_tsm_b16/config_test_dp.json
CUDA_VISIBLE_DEVICES=5 python -m eval_codes.generate_images_bsvd -c experiments/naf_tsm_pt_b8/config_test_dp.json
CUDA_VISIBLE_DEVICES=5 python -m eval_codes.generate_images_bsvd_set8 -c experiments/naf_tsm_pt_b8/config_test_dp.json
CUDA_VISIBLE_DEVICES=2,3 python -m eval_codes.generate_images_bsvd_set8 -c experiments/naf_tsm_pt_b8/config_test_dp.json



CUDA_VISIBLE_DEVICES=7 python -m eval_codes.generate_images_ex1 -c experiments/wiener/config_test.json


python -m eval_codes.visualize_losses_sigma50 -c experiments/naf_tsm3/config_test.json
python -m eval_codes.visualize_losses_sigma50 -c experiments/naf_tsm_dp/config_test3.json
python -m eval_codes.evaluation_sigma50 -c experiments/naf_tsm_b16/config_test_dp.json
python -m eval_codes.evaluation_sigma50_set8 -c experiments/naf_tsm_b16/config_test_dp.json
python -m eval_codes.evaluation_sigma50_bsvd -c experiments/wnet/config_wnet.json

python -m eval_codes.visualize_losses -c experiments/fastdvd2/config_test3.json
python -m eval_codes.evaluation -c experiments/fastdvd2/config_test3.json
python -m eval_codes.evaluation -c experiments/fastdvdM/config_M.json


CUDA_VISIBLE_DEVICES=3 python calc_speed.py
CUDA_VISIBLE_DEVICES=4 python calc_speed_bsvd.py
CUDA_VISIBLE_DEVICES=5 python calc_speed_remo.py