set -e
name='transformer'
gpu_ids=7
test_checkpoints="
resplit/transformer/transformer_both_affectnet-egemaps-wav2vec_cv1_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1/10;
resplit/transformer/transformer_both_affectnet-egemaps-wav2vec_cv2_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1/13;
resplit/transformer/transformer_both_affectnet-egemaps-wav2vec_cv3_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1/1;
resplit/transformer/transformer_both_affectnet-egemaps-wav2vec_cv4_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1/5;
resplit/transformer/transformer_both_affectnet-egemaps-wav2vec_cv5_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1/5
"
test_target='valence'

cmd="python test.py --test_log_dir=./test_results
--checkpoints_dir=./checkpoints
--test_target=$test_target --test_checkpoints='$test_checkpoints' --gpu_ids=$gpu_ids
--name=$name"

# test_checkpoints中不同的路径之间通过;分隔，送入的路径应该是从./checkpoints之后的下一级目录开始的
# 注意，输入的checkpoints路径要带前缀。比如
# 3-16/transformer_both_affectnet-vggish-wav2vec_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc/13_net_reg.pth
# 就应该写成.../13

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh
exit

# bash scripts/test_valence.sh
