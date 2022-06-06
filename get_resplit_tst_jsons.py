import json
import numpy as np
import os
from tqdm import tqdm

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

target = 'arousal'
save_root = './resplit_tst_jsons'

json_file_lst = [
    '/data2/hzp/ABAW_VA_2022/code/test_results/3-16/transformer_lstm/transformer_lstm_both_affectnet-vggish-wav2vec_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run3',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-vggish-wav2vec_cv1_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-vggish-wav2vec_cv2_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-vggish-wav2vec_cv3_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-vggish-wav2vec_cv4_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-vggish-wav2vec_cv5_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2',

    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/ori_both_affectnet-FAU_situ-wav2vec_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-FAU_situ-wav2vec_cv1_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-FAU_situ-wav2vec_cv2_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-FAU_situ-wav2vec_cv3_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-FAU_situ-wav2vec_cv4_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-FAU_situ-wav2vec_cv5_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run2',

    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/fclstm-xl/ori_both_affectnet-FAU_situ-wav2vec_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/fclstm-xl/fclstm-xl_both_affectnet-FAU_situ-wav2vec_cv1_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/fclstm-xl/fclstm-xl_both_affectnet-FAU_situ-wav2vec_cv2_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/fclstm-xl/fclstm-xl_both_affectnet-FAU_situ-wav2vec_cv3_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/fclstm-xl/fclstm-xl_both_affectnet-FAU_situ-wav2vec_cv4_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/fclstm-xl/fclstm-xl_both_affectnet-FAU_situ-wav2vec_cv5_bs16_lr3e-05_dp0.3_seq100_reg-512-256_hidden512_batch_ccc_run2',

    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/ori_both_affectnet-FAU_situ-wav2vec_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-FAU_situ-wav2vec_cv1_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-FAU_situ-wav2vec_cv2_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-FAU_situ-wav2vec_cv3_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-FAU_situ-wav2vec_cv4_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1',
    '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/telstm/transformer_lstm_both_affectnet-FAU_situ-wav2vec_cv5_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden512_layers4_ffn1024_nhead4_batch_ccc_run1'
]

# json_file_lst = [
#     '/data2/hzp/ABAW_VA_2022/code/test_results/leo/transformer_both_affectnet-compare-wav2vec_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_1.0/seed_102',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/leo/transformer_both_affectnet-vggish-egemaps-wav2vec_bs16_lr2e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_1.0/seed_100',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/leo/transformer_both_affectnet-egemaps-wav2vec_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_1.0/seed_100',
    
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-compare-wav2vec_cv1_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-vggish-egemaps-wav2vec_cv1_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-egemaps-wav2vec_cv1_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1',

#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-compare-wav2vec_cv2_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-vggish-egemaps-wav2vec_cv2_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-egemaps-wav2vec_cv2_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1',

#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-compare-wav2vec_cv3_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-vggish-egemaps-wav2vec_cv3_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-egemaps-wav2vec_cv3_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1',

#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-compare-wav2vec_cv4_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-vggish-egemaps-wav2vec_cv4_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-egemaps-wav2vec_cv4_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1',

#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-compare-wav2vec_cv5_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-vggish-egemaps-wav2vec_cv5_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1',
#     '/data2/hzp/ABAW_VA_2022/code/test_results/resplit/transformer/transformer_both_affectnet-egemaps-wav2vec_cv5_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1'
# ]

mkdir(save_root)
for json_dir in tqdm(json_file_lst):
    if json_dir.split('/')[-1][:4] == 'seed':
        name = json_dir.split('/')[-2]
    else:
        name = json_dir.split('/')[-1]
    
    # save_dir = os.path.join(save_root, name)
    # mkdir(save_dir)
    json_file_name = 'tst_pred_arousal_nosmooth.json' if target == 'arousal' else 'tst_pred_valence_smooth.json'
    json_path = os.path.join(json_dir, json_file_name)
    # _cmd = 'cp {} {}'.format(json_path, save_dir)
    # os.system(_cmd)

    with open(json_path, 'r') as f:
        json_dict = json.load(f)

    new_json_dict = {}
    for video in sorted(list(json_dict.keys())):
        new_json_dict[video] = json_dict[video]['pred']

    new_json_path = os.path.join(save_root, name + '.json')
    json.dump(new_json_dict, open(new_json_path, 'w'))
    


