set -e
name='transformer'
gpu_ids=0
# test_checkpoints="
# transformer/baseline_valence_affectnet-wav2vec_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1;
# transformer/baseline_valence_affectnet-wav2vec_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2;
# transformer/baseline_valence_affectnet-compare_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1;
# transformer/baseline_valence_affectnet-compare_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2
# "

# test_checkpoints="
# results_3_14/save_model/transformer_both_affectnet-vggish-egemaps-wav2vec_bs16_lr2e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_['batch_ccc']_[1.0];
# results_3_14/save_model/transformer_both_affectnet-wav2vec_bs16_lr2e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_['batch_ccc']_[1.0]
# "

test_checkpoints="
results_3_14/save_model/transformer_both_affectnet-vggish-egemaps-wav2vec_bs16_lr2e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_1.0;
results_3_14/save_model/transformer_both_affectnet-vggish-egemaps-wav2vec_bs16_lr2e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_1.0
"

prefix_list="
seed_100_12;
seed_101_8
"

test_target='valence'


cmd="python val_smooth_lyc.py --dataset_mode=seq --model=transformer --test_log_dir=./val_lyc_logs
--checkpoints_dir=/data13/lyc/ABAW_VA_2022_lrc/code/checkpoints/
--test_checkpoints='$test_checkpoints' --prefix_list='$prefix_list' --test_target=$test_target --gpu_ids=$gpu_ids
--name=$name"

# test_checkpoints中不同的路径之间通过;分隔，送入的路径应该是从./checkpoints之后的下一级目录开始的
# 加载的数据集以train的为准（但这里也需要写上，否则会找base_opts.py中设置的默认的dataset）

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh
exit



# bash scripts/val_smooth_lyc.sh
