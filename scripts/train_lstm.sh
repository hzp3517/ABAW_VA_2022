set -e
name=ori
batch_size=16
lr=3e-5
dropout=0.3
hidden_size=512
regress_layers=512,256
max_seq_len=100
loss_weights=1
loss_type=batch_ccc
log_dir=./logs/resplit/fclstm-xl
checkpoints_dir=./checkpoints/resplit/fclstm-xl

target=$1
feature=$2
norm_features=$3
run_idx=$4
gpu_ids=$5

cmd="python train_lyc_seed.py --dataset_mode=seq --model=fclstm_xl --gpu_ids=$gpu_ids
--log_dir=$log_dir --checkpoints_dir=$checkpoints_dir --print_freq=5
--max_seq_len=$max_seq_len --regress_layers=$regress_layers --hidden_size=$hidden_size
--feature_set=$feature --target=$target --loss_weights=$loss_weights --loss_type=$loss_type
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=3 --niter_decay=27 
--num_threads=0 --norm_features=$norm_features --norm_method=trn
--name=$name 
--suffix={target}_{feature_set}_bs{batch_size}_lr{lr}_dp{dropout_rate}_seq{max_seq_len}_reg-{regress_layers}_hidden{hidden_size}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# bash scripts/train_lstm.sh both affectnet,FAU_situ,wav2vec None 1 1
# bash scripts/train_lstm.sh both affectnet,FAU_situ,wav2vec None 2 3