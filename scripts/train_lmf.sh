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
log_dir=./logs/3-22/lmf
checkpoints_dir=./checkpoints/3-22/lmf

target=$1
v_features=$2
a_features=$3
norm_features=$4
affine_type=$5
fusion_type=$6
lmf_rank=$7
run_idx=$8
gpu_ids=$9


cmd="python train_lyc_seed.py --dataset_mode=seq_late --model=transformer_mf --gpu_ids=$gpu_ids
--log_dir=$log_dir --checkpoints_dir=$checkpoints_dir --print_freq=2
--hidden_size=$hidden_size --regress_layers=$regress_layers --max_seq_len=$max_seq_len
--num_layers=$num_layers --ffn_dim=$ffn_dim --nhead=$nhead
--a_features=$a_features --v_features=$v_features --target=$target --loss_type=$loss_type --loss_weights=$loss_weights --use_pe
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=10 --niter_decay=20
--num_threads=0 --norm_features=$norm_features --norm_method=trn
--name=$name --encoder_type=transformer --affine_type=$affine_type --fusion_type=$fusion_type --lmf_rank=$lmf_rank
--suffix={target}_{v_features}_{a_features}_affine-{affine_type}_{fusion_type}_lmf-rank{lmf_rank}_bs{batch_size}_lr{lr}_dp{dropout_rate}_seq{max_seq_len}_reg-{regress_layers}_hidden{hidden_size}_layers{num_layers}_ffn{ffn_dim}_nhead{nhead}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# 当affine_type设为fc时，affine_ks以及suffix中对应的部分应当被去掉

# bash scripts/train_lmf.sh both affectnet vggish,wav2vec None fc lmf 4 1 0
# bash scripts/train_lmf.sh both affectnet vggish,wav2vec None fc lmf 4 2 0
# bash scripts/train_lmf.sh both affectnet vggish,wav2vec None fc lmf 4 3 0

# bash scripts/train_lmf.sh both affectnet compare,wav2vec compare fc lmf 4 1 1
# bash scripts/train_lmf.sh both affectnet compare,wav2vec compare fc lmf 4 2 1
# bash scripts/train_lmf.sh both affectnet compare,wav2vec compare fc lmf 4 3 1