import json
import numpy as np
import os
from tqdm import tqdm
from utils.metrics import evaluate_regression

target_name = 'arousal'

json_list = [
    '/data2/hzp/ABAW_VA_2022/code/test_results/3-16/transformer_lstm/transformer_lstm_both_affectnet-vggish-wav2vec_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run3',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-vggish-wav2vec_cv1_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-vggish-wav2vec_cv2_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-vggish-wav2vec_cv3_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-vggish-wav2vec_cv4_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-vggish-wav2vec_cv5_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2'
]

# json_list = [
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/ori_both_affectnet-FAU_situ-wav2vec_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run2',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-FAU_situ-wav2vec_cv1_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run2',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-FAU_situ-wav2vec_cv2_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run2',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-FAU_situ-wav2vec_cv3_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run2',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-FAU_situ-wav2vec_cv4_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run2',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-FAU_situ-wav2vec_cv5_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run2'
# ]

# json_list = [
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/fclstm-xl/ori_both_affectnet-FAU_situ-wav2vec_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/fclstm-xl/fclstm-xl_both_affectnet-FAU_situ-wav2vec_cv1_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/fclstm-xl/fclstm-xl_both_affectnet-FAU_situ-wav2vec_cv2_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/fclstm-xl/fclstm-xl_both_affectnet-FAU_situ-wav2vec_cv3_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/fclstm-xl/fclstm-xl_both_affectnet-FAU_situ-wav2vec_cv4_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/fclstm-xl/fclstm-xl_both_affectnet-FAU_situ-wav2vec_cv5_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2'
# ]

# json_list = [
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/ori_both_affectnet-FAU_situ-wav2vec_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-FAU_situ-wav2vec_cv1_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-FAU_situ-wav2vec_cv2_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-FAU_situ-wav2vec_cv3_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-FAU_situ-wav2vec_cv4_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-FAU_situ-wav2vec_cv5_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1'
# ]



if target_name == 'arousal':
    json_file_name = 'val_pred_arousal_nosmooth.json'
else:
    json_file_name = 'val_pred_valence_smooth.json'

total_preds = []
total_labels = []

for json_dir in tqdm(json_list):
    json_f = os.path.join(json_dir, json_file_name)
    with open(json_f, 'r') as f:
        json_dict = json.load(f)
    for video in json_dict.keys():
        preds = json_dict[video]['pred']
        labels = json_dict[video]['label']
        total_preds += preds
        total_labels += labels

total_preds = np.array(total_preds)
total_labels = np.array(total_labels)
mse, rmse, pcc, ccc = evaluate_regression(total_labels, total_preds)

print('total ccc: {}'.format(ccc))

