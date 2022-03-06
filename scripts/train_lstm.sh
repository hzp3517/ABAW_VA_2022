set -e
name=$1
target=$2
feature=$3
norm_features=$4
batch_size=$5
lr=$6
dropout=$7
hidden_size=$8
regress_layers=$9
max_seq_len=${10}
loss_type=${11}
run_idx=${12}
gpu_ids=${13}


cmd="python train.py --dataset_mode=seq --model=lstm_baseline --gpu_ids=$gpu_ids
--log_dir=./logs/lstm --checkpoints_dir=./checkpoints/lstm --print_freq=2
--max_seq_len=$max_seq_len --regress_layers=$regress_layers --hidden_size=$hidden_size
--feature_set=$feature --target=$target --loss_type=$loss_type
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=30 --niter_decay=40 --num_threads=0 --norm_features=$norm_features --norm_method=trn
--name=$name --suffix={target}_{feature_set}_bs{batch_size}_lr{lr}_dp{dropout_rate}_reg-{regress_layers}_hidden{hidden_size}_seq{max_seq_len}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh



# bash scripts/train_lstm.sh lstm valence denseface None 32 1e-4 0.3 128 256,256 100 mse 1 6
# bash scripts/train_lstm.sh lstm arousal denseface None 32 1e-4 0.3 128 256,256 100 mse 1 7