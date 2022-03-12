set -e
name=$1
batch_size=$2
lr=$3
dropout=$4
arch_type=$5
regress_layers=$6
loss_type=$7
run_idx=$8
gpu_ids=$9


cmd="python train_frame_saveallcp.py --dataset_mode=frame --model=resnet --gpu_ids=$gpu_ids
--log_dir=./logs/resnet --checkpoints_dir=./checkpoints/resnet --print_freq=10
--img_type=color --read_length=20000 --norm_type=reduce_mean
--arch_type=$arch_type --regress_layers=$regress_layers
--feature_set=None --loss_type=$loss_type
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=5 --niter_decay=5 --num_threads=0
--name=$name --suffix=bs{batch_size}_lr{lr}_dp{dropout_rate}_pretrain-{arch_type}_reg-{regress_layers}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# bash scripts/train_resnet.sh resnet 64 1e-4 0.3 resnet50_ft 256 mse 1 6
# bash scripts/train_resnet.sh resnet 64 1e-4 0.3 resnet50_ft 256 mse 2 7