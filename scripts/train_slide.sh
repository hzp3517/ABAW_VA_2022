set -e
name=$1
target=$2
feature=$3
norm_features=$4
batch_size=$5
lr=$6
dropout=$7
regress_layers=$8
hidden_size=$9
num_layers=${10}
ffn_dim=${11}
nhead=${12}
loss_type=${13}
run_idx=${14}
gpu_ids=${15}


cmd="python train_slide.py --dataset_mode=seq_slide --model=transformer_slide --gpu_ids=$gpu_ids
--log_dir=./logs/debug --checkpoints_dir=./checkpoints/debug --print_freq=100
--hidden_size=$hidden_size --regress_layers=$regress_layers --num_layers=$num_layers
--ffn_dim=$ffn_dim --nhead=$nhead
--feature_set=$feature --target=$target --loss_type=$loss_type
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=30 --niter_decay=40 --num_threads=0 --norm_features=$norm_features --norm_method=trn
--name=$name 
--suffix={target}_{feature_set}_bs{batch_size}_lr{lr}_dp{dropout_rate}_reg-{regress_layers}_hidden{hidden_size}_layers{num_layers}_ffn{ffn_dim}_nhead{nhead}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# bash scripts/train_slide.sh transformer valence denseface None 32 5e-5 0.3 256,256 256 4 1024 4 mse 1 1
# bash scripts/train_slide.sh transformer arousal denseface None 32 5e-5 0.3 256,256 256 4 1024 4 mse 1 1
