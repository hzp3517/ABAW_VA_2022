set -e
name=$1
task=$2
max_read_num=$3
max_clip_length=$4
batch_size=$5
lr=$6
dropout=$7
regress_layers=$8
loss_type=$9
run_idx=${10}
gpu_ids=${11}


cmd="python train_wav2vec_saveallcp.py --dataset_mode=audio_clip --model=wav2vec --gpu_ids=$gpu_ids
--log_dir=./logs/wav2vec --checkpoints_dir=./checkpoints/wav2vec --print_freq=1
--max_read_num=$max_read_num --max_clip_length=$max_clip_length
--regress_layers=$regress_layers --feature_set=None --loss_type=$loss_type
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=5 --niter_decay=5 --num_threads=0 --task=$task
--name=$name --suffix=bs{batch_size}_task-{task}_readnum{max_read_num}_cliplen{max_clip_length}_lr{lr}_dp{dropout_rate}_reg-{regress_layers}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# bash scripts/train_wav2vec.sh wav2vec v+a 10 10 2 1e-4 0.3 256 mse 1 3
# bash scripts/train_wav2vec.sh wav2vec v 10 10 2 1e-4 0.3 256 mse 1 4