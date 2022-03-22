set -e
name=transformer
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
log_dir=./logs/3-22/transformer_late
checkpoints_dir=./checkpoints/3-22/transformer_late

target=$1
v_features=$2
a_features=$3
norm_features=$4
affine_type=$5
affine_ks=$6
run_idx=$7
gpu_ids=$8


cmd="python train_lyc_seed.py --dataset_mode=seq_late --model=transformer_late --gpu_ids=$gpu_ids
--log_dir=$log_dir --checkpoints_dir=$checkpoints_dir --print_freq=2
--hidden_size=$hidden_size --regress_layers=$regress_layers --max_seq_len=$max_seq_len
--num_layers=$num_layers --ffn_dim=$ffn_dim --nhead=$nhead
--a_features=$a_features --v_features=$v_features --target=$target --loss_type=$loss_type --loss_weights=$loss_weights --use_pe
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=10 --niter_decay=20
--num_threads=0 --norm_features=$norm_features --norm_method=trn
--name=$name --encoder_type=transformer --affine_type=$affine_type --affine_ks=$affine_ks
--suffix={target}_{v_features}_{a_features}_affine-{affine_type}_ks{affine_ks}_bs{batch_size}_lr{lr}_dp{dropout_rate}_seq{max_seq_len}_reg-{regress_layers}_hidden{hidden_size}_layers{num_layers}_ffn{ffn_dim}_nhead{nhead}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# log_dir=./logs/3-22/transformer_late
# checkpoints_dir=./checkpoints/3-22/transformer_late

# 当affine_type设为fc时，affine_ks以及suffix中对应的部分应当被去掉

# bash scripts/train_transformer_late.sh both affectnet vggish,wav2vec None cnn 3 1 3
# bash scripts/train_transformer_late.sh both affectnet vggish,wav2vec None cnn 3 2 3
# bash scripts/train_transformer_late.sh both affectnet vggish,wav2vec None cnn 3 3 3

# bash scripts/train_transformer_late.sh both affectnet vggish,wav2vec None cnn 5 1 3
# bash scripts/train_transformer_late.sh both affectnet vggish,wav2vec None cnn 5 2 3
# bash scripts/train_transformer_late.sh both affectnet vggish,wav2vec None cnn 5 3 3

# bash scripts/train_transformer_late.sh both affectnet compare,wav2vec compare cnn 3 1 4
# bash scripts/train_transformer_late.sh both affectnet compare,wav2vec compare cnn 3 2 4
# bash scripts/train_transformer_late.sh both affectnet compare,wav2vec compare cnn 3 3 4

# bash scripts/train_transformer_late.sh both affectnet compare,wav2vec compare cnn 5 1 4
# bash scripts/train_transformer_late.sh both affectnet compare,wav2vec compare cnn 5 2 4
# bash scripts/train_transformer_late.sh both affectnet compare,wav2vec compare cnn 5 3 4



