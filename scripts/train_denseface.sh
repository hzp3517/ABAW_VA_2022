set -e
name=$1
task=$2
read_length=$3
batch_size=$4
lr=$5
dropout=$6
regress_layers=$7
loss_type=$8
run_idx=$9
gpu_ids=${10}


cmd="python train_frame_saveallcp.py --dataset_mode=frame --model=denseface --gpu_ids=$gpu_ids
--log_dir=./logs/denseface --checkpoints_dir=./checkpoints/denseface --print_freq=100
--img_type=gray --read_length=$read_length --norm_type=norm
--pretrain=y --frozen_dense_blocks=0 --regress_layers=$regress_layers
--feature_set=None --loss_type=$loss_type
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=5 --niter_decay=5 --num_threads=0 --task=$task
--name=$name --suffix=bs{batch_size}_task-{task}_readlen{read_length}_lr{lr}_dp{dropout_rate}_reg-{regress_layers}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# bash scripts/train_denseface.sh denseface v+a 200000 512 1e-4 0.3 256 mse 1 0