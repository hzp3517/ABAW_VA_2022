set -e
name=baseline
target=valence
feature=$1
norm_features=$2
batch_size=32
lr=$3
dropout=0.3
regress_layers=256,256
max_seq_len=250
hidden_size=256
num_layers=4
ffn_dim=1024
nhead=4
loss_type=$4
encoder_type=$5
pe_type=$6
run_idx=$7
gpu_ids=$8


cmd="python train.py --dataset_mode=seq --model=transformer --gpu_ids=$gpu_ids
--log_dir=./logs/transformer --checkpoints_dir=./checkpoints/transformer --print_freq=2
--hidden_size=$hidden_size --regress_layers=$regress_layers 
--max_seq_len=$max_seq_len
--num_layers=$num_layers --ffn_dim=$ffn_dim --nhead=$nhead
--feature_set=$feature --target=$target --loss_type=$loss_type
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=10 --niter_decay=20 
--encoder_type=$encoder_type --pe_type=$pe_type
--num_threads=0 --norm_features=$norm_features --norm_method=trn
--name=$name 
--suffix={target}_{feature_set}_{encoder_type}_{pe_type}_bs{batch_size}_lr{lr}_dp{dropout_rate}_reg-{regress_layers}_hidden{hidden_size}_layers{num_layers}_ffn{ffn_dim}_nhead{nhead}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# --lr_policy=linear_with_warmup

# bash scripts/train_transformer_valence.sh affectnet,compare compare 3e-5 batch_ccc transformer sincos 1 1
# bash scripts/train_transformer_valence.sh affectnet,compare compare 3e-5 batch_ccc transformer sincos 2 2