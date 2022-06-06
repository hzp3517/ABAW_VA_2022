set -e
name='transformer'
gpu_ids=3
# test_checkpoints="
# resplit/transformer/transformer_both_affectnet-FAU_situ-wav2vec_cv1_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1/5;
# resplit/transformer/transformer_both_affectnet-FAU_situ-wav2vec_cv2_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1/20;
# resplit/transformer/transformer_both_affectnet-FAU_situ-wav2vec_cv3_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1/10;
# resplit/transformer/transformer_both_affectnet-FAU_situ-wav2vec_cv4_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1/12;
# resplit/transformer/transformer_both_affectnet-FAU_situ-wav2vec_cv5_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1/11
# "

# test_checkpoints="
# resplit/fclstm-xl/fclstm-xl_both_affectnet-FAU_situ-wav2vec_cv1_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2/11;
# resplit/fclstm-xl/fclstm-xl_both_affectnet-FAU_situ-wav2vec_cv2_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2/24;
# resplit/fclstm-xl/fclstm-xl_both_affectnet-FAU_situ-wav2vec_cv3_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2/11;
# resplit/fclstm-xl/fclstm-xl_both_affectnet-FAU_situ-wav2vec_cv4_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2/17;
# resplit/fclstm-xl/fclstm-xl_both_affectnet-FAU_situ-wav2vec_cv5_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2/11;
# "

# test_checkpoints="
# resplit/telstm/transformer_lstm_both_affectnet-FAU_situ-wav2vec_cv1_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1/25;
# resplit/telstm/transformer_lstm_both_affectnet-FAU_situ-wav2vec_cv2_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1/26;
# resplit/telstm/transformer_lstm_both_affectnet-FAU_situ-wav2vec_cv3_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1/14;
# resplit/telstm/transformer_lstm_both_affectnet-FAU_situ-wav2vec_cv4_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1/17;
# resplit/telstm/transformer_lstm_both_affectnet-FAU_situ-wav2vec_cv5_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1/13
# "

test_checkpoints="
resplit/transformer/ori_both_affectnet-FAU_situ-wav2vec_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run2/14;
resplit/fclstm-xl/ori_both_affectnet-FAU_situ-wav2vec_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2/13;
resplit/telstm/ori_both_affectnet-FAU_situ-wav2vec_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1/10
"


test_target='arousal'

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

# bash scripts/test_arousal.sh
