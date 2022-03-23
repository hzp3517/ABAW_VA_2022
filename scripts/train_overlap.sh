set -e
name=transformer
batch_size=16
lr=5e-5
dropout=0.3
regress_layers=256,256
max_seq_len=250
hidden_size=256
num_layers=4
ffn_dim=1024
nhead=4

loss_weights=1
loss_type=batch_ccc
log_dir=./logs/3-24/overlap
checkpoints_dir=./checkpoints/3-24/overlap

target=$1
feature=$2
norm_features=$3
hop_len=$4
run_idx=$5
gpu_ids=$6


cmd="python train_lyc_seed.py --dataset_mode=seq --model=transformer_overlap --gpu_ids=$gpu_ids
--log_dir=$log_dir --checkpoints_dir=$checkpoints_dir --print_freq=2
--hidden_size=$hidden_size --regress_layers=$regress_layers --max_seq_len=$max_seq_len
--num_layers=$num_layers --ffn_dim=$ffn_dim --nhead=$nhead
--feature_set=$feature --target=$target --loss_type=$loss_type --loss_weights=$loss_weights --use_pe
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=10 --niter_decay=20
--num_threads=0 --norm_features=$norm_features --norm_method=trn
--hop_len=$hop_len
--name=$name --encoder_type=transformer
--suffix={target}_{feature_set}_hop{hop_len}_bs{batch_size}_lr{lr}_dp{dropout_rate}_seq{max_seq_len}_reg-{regress_layers}_hidden{hidden_size}_layers{num_layers}_ffn{ffn_dim}_nhead{nhead}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# bash scripts/train_overlap.sh both affectnet,vggish,wav2vec None 30 1 3
# bash scripts/train_overlap.sh both affectnet,vggish,wav2vec None 30 2 3
# bash scripts/train_overlap.sh both affectnet,vggish,wav2vec None 30 3 3

# bash scripts/train_overlap.sh both affectnet,compare,wav2vec compare 30 1 3
# bash scripts/train_overlap.sh both affectnet,compare,wav2vec compare 30 2 3
# bash scripts/train_overlap.sh both affectnet,compare,wav2vec compare 30 3 3



