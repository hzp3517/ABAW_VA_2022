set -e
name=$1
target=$2
feature=$3
norm_features=$4
batch_size=$5
lr=$6
dropout=$7
regress_layers=$8
tcn_dropout=$9
tcn_channels=${10}
kernel_size=${11}
max_seq_len=${12}
loss_type=${13}
run_idx=${14}
gpu_ids=${15}


cmd="python train.py --dataset_mode=seq --model=tcn --gpu_ids=$gpu_ids
--log_dir=./logs/tcn --checkpoints_dir=./checkpoints/tcn --print_freq=2
--max_seq_len=$max_seq_len --regress_layers=$regress_layers
--feature_set=$feature --target=$target --loss_type=$loss_type
--tcn_dropout=$tcn_dropout --tcn_channels=$tcn_channels --kernel_size=$kernel_size
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=30 --niter_decay=40 --num_threads=0 --norm_features=$norm_features --norm_method=trn
--name=$name --suffix={target}_{feature_set}_bs{batch_size}_lr{lr}_dp{dropout_rate}_reg-{regress_layers}_tcn-dp{tcn_dropout}_tcn-ch{tcn_channels}_ks{kernel_size}_seq{max_seq_len}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# bash scripts/train_tcn.sh tcn valence denseface None 32 1e-4 0.3 256,256 0.2 256 3 100 mse 1 1
# bash scripts/train_tcn.sh tcn arousal denseface None 32 1e-4 0.3 256,256 0.2 256 3 100 mse 1 2

# bash scripts/train_tcn.sh tcn valence denseface None 32 1e-4 0.3 256,256 0.2 256 3 100 mse 2 4
# bash scripts/train_tcn.sh tcn arousal denseface None 32 1e-4 0.3 256,256 0.2 256 3 100 mse 2 5

# bash scripts/train_tcn.sh tcn valence denseface None 32 1e-4 0.3 256,256 0.2 256,256,256 3 100 mse 1 4
# bash scripts/train_tcn.sh tcn arousal denseface None 32 1e-4 0.3 256,256 0.2 256,256,256 3 100 mse 1 5