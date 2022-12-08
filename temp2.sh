#CUDA_VISIBLE_DEVICES=5 python -m eval_codes.generate_images_davis_multi -c experiments/naf_tsm_multi/config_test.json
#CUDA_VISIBLE_DEVICES=5 python -m eval_codes.generate_images_set8_multi -c experiments/naf_tsm_multi/config_test.json

#python -m eval_codes.evaluation_sigma50_set8_multi -c experiments/naf_tsm_multi/config_test.json
#python -m eval_codes.evaluation_sigma50_multi -c experiments/naf_tsm_multi/config_test.json

CUDA_VISIBLE_DEVICES=5 python -m eval_codes.generate_images_davis -c experiments/naf_tsm_pt_b8/config_test_dp.json
python -m eval_codes.evaluation_sigma50 -c experiments/naf_tsm_pt_b8/config_test_dp.json

CUDA_VISIBLE_DEVICES=5 python -m eval_codes.generate_images_set8 -c experiments/naf_tsm_pt_b8/config_test_dp.json
python -m eval_codes.evaluation_sigma50_set8 -c experiments/naf_tsm_pt_b8/config_test_dp.json