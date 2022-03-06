set -e
name=$1
target=$2
feature=$3
norm_features=$4
batch_size=$5
lr=$6
dropout=$7
regress_layers=$8
max_seq_len=$9
hidden_size=${10}
num_layers=${11}
ffn_dim=${12}
nhead=${13}
loss_type=${14}
run_idx=${15}
gpu_ids=${16}


cmd="python train_csv.py --dataset_mode=seq --model=transformer --gpu_ids=$gpu_ids
--log_dir=./logs/transformer --checkpoints_dir=./checkpoints/transformer --print_freq=2
--hidden_size=$hidden_size --regress_layers=$regress_layers --max_seq_len=$max_seq_len
--num_layers=$num_layers --ffn_dim=$ffn_dim --nhead=$nhead
--feature_set=$feature --target=$target --loss_type=$loss_type
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=30 --niter_decay=40
--num_threads=0 --norm_features=$norm_features --norm_method=trn
--name=$name 
--suffix={target}_{feature_set}_bs{batch_size}_lr{lr}_dp{dropout_rate}_reg-{regress_layers}_hidden{hidden_size}_layers{num_layers}_ffn{ffn_dim}_nhead{nhead}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# --lr_policy=linear_with_warmup

# 5e-5


# bash scripts/train_transformer_auto.sh baseline arousal vggface2 vggface2 32 5e-5 0.3 256,256 100 256 4 1024 4 mse 1 7

