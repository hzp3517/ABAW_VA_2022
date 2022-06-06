set -e
name=transformer
batch_size=16
lr=2e-5
dropout=0.3
regress_layers=256,256
max_seq_len=250
hidden_size=512
num_layers=4
ffn_dim=1024
nhead=4

loss_weights=1
loss_type=batch_ccc
log_dir=./logs/resplit/transformer
checkpoints_dir=./checkpoints/resplit/transformer

target=$1
feature=$2
norm_features=$3
cv=$4
run_idx=$5
gpu_ids=$6


cmd="python train_lyc_seed.py --dataset_mode=seq_resplit --model=transformer --gpu_ids=$gpu_ids
--log_dir=$log_dir --checkpoints_dir=$checkpoints_dir --print_freq=2
--hidden_size=$hidden_size --regress_layers=$regress_layers --max_seq_len=$max_seq_len
--num_layers=$num_layers --ffn_dim=$ffn_dim --nhead=$nhead
--feature_set=$feature --target=$target --loss_type=$loss_type --loss_weights=$loss_weights --use_pe
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=10 --niter_decay=20
--num_threads=0 --norm_features=$norm_features --norm_method=trn --cv=$cv
--name=$name --encoder_type=transformer
--suffix={target}_{feature_set}_cv{cv}_bs{batch_size}_lr{lr}_dp{dropout_rate}_seq{max_seq_len}_reg-{regress_layers}_hidden{hidden_size}_layers{num_layers}_ffn{ffn_dim}_nhead{nhead}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# bash scripts/train_resplit_transformer_512.sh both affectnet,FAU_situ,wav2vec None 1 1 0
# bash scripts/train_resplit_transformer_512.sh both affectnet,FAU_situ,wav2vec None 1 2 0
# bash scripts/train_resplit_transformer_512.sh both affectnet,FAU_situ,wav2vec None 2 1 0
# bash scripts/train_resplit_transformer_512.sh both affectnet,FAU_situ,wav2vec None 2 2 0

