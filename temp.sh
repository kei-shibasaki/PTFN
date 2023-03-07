CUDA_VISIBLE_DEVICES=1 python temp.py
CUDA_VISIBLE_DEVICES=5 python temp.py
CUDA_VISIBLE_DEVICES=6 python temp.py
CUDA_VISIBLE_DEVICES=7 python temp.py
CUDA_VISIBLE_DEVICES=0 python temp.py

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train_codes.train_dp -c config/config_ptfn.json
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train_codes.train_dp -c config/config_ptfn4.json
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train_codes.train_blind_dp -c config/config_ptfn2.json
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train_codes.finetune_dp -c config/config_finetune.json
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train_codes.finetune_dp -c config/config_finetune2.json
CUDA_VISIBLE_DEVICES=2,3 python -m train_codes.finetune_dp -c config/config_finetune2.json
CUDA_VISIBLE_DEVICES=4 python -m train_codes.finetune -c config/config_finetune.json
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train_codes.train_dp -c config/config_ptfn3.json
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m train_codes.train_progressive_dp -c config/config_ptfn_pro.json
CUDA_VISIBLE_DEVICES=0,1 python -m train_codes.train_progressive_dp -c config/config_ptfn_pro.json
CUDA_VISIBLE_DEVICES=5 python -m train_codes.train -c config/config_ptfn.json


CUDA_VISIBLE_DEVICES=6 python -m eval_codes.generate_images_davis -nl 50 -c experiments/ptfn_b8/config_test.json 
CUDA_VISIBLE_DEVICES=5 python -m eval_codes.generate_images_set8 -c experiments/naf_tsm_b16/config_test_dp.json
CUDA_VISIBLE_DEVICES=0 python -m eval_codes.generate_images_bsvd -c UNUSED/config_wnet.json
CUDA_VISIBLE_DEVICES=1 python -m eval_codes.generate_images_bsvd_set8 -c UNUSED/config_wnet.json
CUDA_VISIBLE_DEVICES=2,3 python -m eval_codes.generate_images_bsvd_set8 -c experiments/naf_tsm_pt_b8/config_test_dp.json

CUDA_VISIBLE_DEVICES=7 python -m eval_codes.generate_images_ex1 -c experiments/wiener/config_test.json


python -m eval_codes.visualize_losses_sigma50 -c experiments/naf_tsm3/config_test.json
python -m eval_codes.visualize_losses_sigma50 -c experiments/naf_tsm_dp/config_test3.json
python -m eval_codes.evaluation_sigma50_set8 -c experiments/naf_tsm_b16/config_test_dp.json
python -m eval_codes.evaluation_sigma50_bsvd -c experiments/wnet/config_wnet.json

python -m eval_codes.visualize_losses -c experiments/fastdvd2/config_test3.json
python -m eval_codes.evaluation -c experiments/fastdvd2/config_test3.json
python -m eval_codes.evaluation -c experiments/fastdvdM/config_M.json


CUDA_VISIBLE_DEVICES=3 python calc_speed.py
CUDA_VISIBLE_DEVICES=1 python calc_speed_bsvd.py
CUDA_VISIBLE_DEVICES=5 python calc_speed_bsvd.py
CUDA_VISIBLE_DEVICES=7 python calc_speed_remo.py
CUDA_VISIBLE_DEVICES=3 python calc_speed_vnlb.py
