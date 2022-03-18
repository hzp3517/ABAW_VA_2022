set -e
name=transformer_lstm
batch_size=16
lr=2e-5
dropout=0.3
regress_layers=256,256
max_seq_len=250
hidden_size=256
num_layers=4
ffn_dim=1024
nhead=4

loss_weights=1
loss_type=batch_ccc
log_dir=./logs/3-16/transformer_lstm_a_6406
checkpoints_dir=./checkpoints/3-16/transformer_lstm_a_6406

pth_path="/data2/hzp/ABAW_VA_2022/code/checkpoints/3-16/transformer_both_affectnet-vggish-wav2vec_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc/13_net_seq.pth"

target=$1
feature=$2
norm_features=$3
residual=$4
run_idx=$5
gpu_ids=$6


cmd="python train_lyc_seed.py --dataset_mode=seq --model=transformer_lstm --gpu_ids=$gpu_ids
--log_dir=$log_dir --checkpoints_dir=$checkpoints_dir --print_freq=2
--hidden_size=$hidden_size --regress_layers=$regress_layers --max_seq_len=$max_seq_len
--num_layers=$num_layers --ffn_dim=$ffn_dim --nhead=$nhead
--feature_set=$feature --target=$target --loss_type=$loss_type --loss_weights=$loss_weights --use_pe
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=10 --niter_decay=20
--num_threads=0 --norm_features=$norm_features --norm_method=trn
--transformer_pretrained --pth_path=$pth_path --residual=$residual
--name=$name --encoder_type=transformer
--suffix={target}_{feature_set}_res-{residual}_bs{batch_size}_lr{lr}_dp{dropout_rate}_seq{max_seq_len}_reg-{regress_layers}_hidden{hidden_size}_layers{num_layers}_ffn{ffn_dim}_nhead{nhead}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# log_dir=./logs/3-16/transformer_lstm_a_6406
# checkpoints_dir=./checkpoints/3-16/transformer_lstm_a_6406


# 跑之前注意修改：
#   ① log_dir和checkpoints_dir，若采用训过的transformer，则要加上[target]_[ccc]的后缀
#   ② pth_path，预训练过的transformer的pth路径
#   ③ --transformer_pretrained，--pth_path 视情况是否加上


# bash scripts/train_transformer_lstm_1.sh both affectnet,vggish,wav2vec None y 1 0
# bash scripts/train_transformer_lstm_1.sh both affectnet,vggish,wav2vec None y 2 0
# bash scripts/train_transformer_lstm_1.sh both affectnet,vggish,wav2vec None y 3 0

# bash scripts/train_transformer_lstm_1.sh both affectnet,vggish,wav2vec None n 1 0
# bash scripts/train_transformer_lstm_1.sh both affectnet,vggish,wav2vec None n 2 0
# bash scripts/train_transformer_lstm_1.sh both affectnet,vggish,wav2vec None n 3 0