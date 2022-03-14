set -e
name='transformer'
gpu_ids=0
test_checkpoints="
transformer/baseline_valence_affectnet-wav2vec_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1;
transformer/baseline_valence_affectnet-wav2vec_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2;
transformer/baseline_valence_affectnet-compare_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1;
transformer/baseline_valence_affectnet-compare_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2
"

# test_checkpoints="
# transformer/baseline_arousal_affectnet-compare_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_mse_run1;
# transformer/baseline_arousal_affectnet-compare_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_mse_run2;
# transformer/baseline_arousal_affectnet-wav2vec_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_mse_run1;
# transformer/baseline_arousal_affectnet-wav2vec_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_mse_run2;
# transformer/baseline_valence_affectnet-compare_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1;
# transformer/baseline_valence_affectnet-compare_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2;
# transformer/baseline_valence_affectnet-wav2vec_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1;
# transformer/baseline_valence_affectnet-wav2vec_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2
# "

cmd="python val_smooth.py --dataset_mode=seq --model=transformer --test_log_dir=./val_logs
--checkpoints_dir=./checkpoints
--test_checkpoints='$test_checkpoints' --gpu_ids=$gpu_ids
--name=$name"

# test_checkpoints中不同的路径之间通过;分隔，送入的路径应该是从./checkpoints之后的下一级目录开始的
# 加载的数据集以train的为准（但这里也需要写上，否则会找base_opts.py中设置的默认的dataset）

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh
exit



# bash scripts/val_smooth.sh
