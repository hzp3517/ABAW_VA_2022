import numpy as np
import torch
import os

paths = ['/data8/hzp/ABAW_VA_2022/code/checkpoints/transformer/baseline_arousal_affectnet-wav2vec_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_mse_run1/11_net_reg.pth', '/data8/hzp/ABAW_VA_2022/code/checkpoints/transformer/baseline_arousal_affectnet-wav2vec_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_mse_run2/6_net_reg.pth', '/data8/hzp/ABAW_VA_2022/code/checkpoints/transformer/baseline_valence_affectnet-wav2vec_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run1/14_net_reg.pth', '/data8/hzp/ABAW_VA_2022/code/checkpoints/transformer/baseline_valence_affectnet-wav2vec_transformer_sincos_bs32_lr3e-05_dp0.3_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run2/7_net_reg.pth']

save_dir = '/data8/hzp/tmp/'

original_dict = torch.load(paths[3])
new_dict = {}
for key in original_dict.keys():
    if key == 'module.1.weight':
        new_dict['module.0.weight'] = original_dict['module.1.weight']
    elif key == 'module.1.bias':
        new_dict['module.0.bias'] = original_dict['module.1.bias']
    elif key == 'module.4.weight':
        new_dict['module.3.weight'] = original_dict['module.4.weight']
    elif key == 'module.4.bias':
        new_dict['module.3.bias'] = original_dict['module.4.bias']
    else:
        new_dict[key] = original_dict[key]

save_path = os.path.join(save_dir, '7_net_reg.pth')
torch.save(new_dict, save_path)


# path = '/data8/hzp/tmp/11_net_reg.pth'
# a = torch.load(path)
# print(a.keys())