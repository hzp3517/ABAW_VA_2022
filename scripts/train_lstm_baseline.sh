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
run_idx=${11}
gpu_ids=${12}


cmd="python train.py --dataset_mode=seq --model=lstm_baseline --gpu_ids=$gpu_ids
--log_dir=./logs/3-5 --checkpoints_dir=./checkpoints/3-5 --print_freq=2
--max_seq_len=$max_seq_len --regress_layers=$regress_layers --hidden_size=$hidden_size
--feature_set=$feature --target=$target
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=30 --niter_decay=40 --num_threads=0 --norm_features=$norm_features --norm_method=trn
--name=$name --suffix={target}_{feature_set}_bs{batch_size}_lr{lr}_dp{dropout_rate}_hidden{hidden_size}_reg-{regress_layers}_seq{max_seq_len}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# bash scripts/train_lstm_baseline.sh lstm valence denseface None 64 1e-4 0.3 128 256,256 100 1 3
# bash scripts/train_lstm_baseline.sh lstm arousal denseface None 64 1e-4 0.3 128 256,256 100 1 4

# bash scripts/train_lstm_baseline.sh lstm valence denseface None 64 1e-4 0.3 128 256,256 100 2 1
# bash scripts/train_lstm_baseline.sh lstm arousal denseface None 64 1e-4 0.3 128 256,256 100 2 5

# bash scripts/train_lstm_baseline.sh lstm valence vggface2 vggface2 64 1e-4 0.3 128 256,256 100 1 1
# bash scripts/train_lstm_baseline.sh lstm arousal vggface2 vggface2 64 1e-4 0.3 128 256,256 100 1 2

# bash scripts/train_lstm_baseline.sh lstm valence vggface2 None 32 1e-4 0.3 128 256,256 100 1 3
# bash scripts/train_lstm_baseline.sh lstm arousal vggface2 None 64 1e-4 0.3 128 256,256 100 1 4


# bash scripts/train_lstm_baseline.sh lstm arousal denseface None 8 1e-4 0.3 128 256,256 100 1 1
# bash scripts/train_lstm_baseline.sh lstm valence denseface None 8 1e-4 0.3 128 256,256 100 1 3


# bash scripts/train_lstm_baseline.sh lstm arousal vggish,denseface None 32 1e-4 0.3 128 256,256 100 1 2
# bash scripts/train_lstm_baseline.sh lstm valence vggish,denseface None 32 1e-4 0.3 128 256,256 100 1 4


# bash scripts/train_lstm_baseline.sh lstm arousal vggish,denseface None 32 1e-4 0.3 128 256,256 100 1 7
# bash scripts/train_lstm_baseline.sh lstm valence vggish,denseface None 32 1e-4 0.3 128 256,256 100 1 7

# bash scripts/train_lstm_baseline.sh lstm valence denseface_backup None 64 1e-4 0.3 128 256,256 100 1 1
# bash scripts/train_lstm_baseline.sh lstm valence denseface_backup None 64 1e-4 0.3 128 256,256 100 1 2