set -e
name=downsample
target=valence
feature=affectnet,compare
norm_features=compare
batch_size=32
lr=$1
dropout=0.3
regress_layers=256,256
downsample_rate=$2
max_seq_len=250
hidden_size=256
num_layers=4
ffn_dim=1024
nhead=4
loss_type=$3
encoder_type=$4
pe_type=$5
run_idx=$6
gpu_ids=$7


cmd="python train.py --dataset_mode=seq_downsample --model=transformer --gpu_ids=$gpu_ids
--log_dir=./logs/transformer --checkpoints_dir=./checkpoints/transformer --print_freq=2
--hidden_size=$hidden_size --regress_layers=$regress_layers 
--max_seq_len=$max_seq_len --downsample_rate=$downsample_rate
--num_layers=$num_layers --ffn_dim=$ffn_dim --nhead=$nhead
--feature_set=$feature --target=$target --loss_type=$loss_type
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=10 --niter_decay=20 
--encoder_type=$encoder_type --pe_type=$pe_type
--num_threads=0 --norm_features=$norm_features --norm_method=trn
--name=$name 
--suffix={target}_{feature_set}_downsample{downsample_rate}_{encoder_type}_{pe_type}_bs{batch_size}_lr{lr}_dp{dropout_rate}_reg-{regress_layers}_hidden{hidden_size}_layers{num_layers}_ffn{ffn_dim}_nhead{nhead}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# --lr_policy=linear_with_warmup

# bash scripts/train_transformer_valence_downsample.sh 3e-5 1 batch_ccc transformer sincos 1 5

# bash scripts/train_transformer_valence_downsample.sh 3e-5 3 batch_ccc transformer sincos 1 6