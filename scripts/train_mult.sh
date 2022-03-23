set -e
name=mult
batch_size=16
lr=2e-5
dropout=0.3
regress_layers=256,256
max_seq_len=250
hidden_size=256
num_layers=4
nhead=4

loss_weights=1
loss_type=batch_ccc
log_dir=./logs/3-22/mult
checkpoints_dir=./checkpoints/3-22/mult

target=$1
v_features=$2
a_features=$3
norm_features=$4
use_selfattn=$5
run_idx=$6
gpu_ids=$7


cmd="python train_lyc_seed.py --dataset_mode=seq_late --model=mult --gpu_ids=$gpu_ids
--log_dir=$log_dir --checkpoints_dir=$checkpoints_dir --print_freq=2
--hidden_size=$hidden_size --regress_layers=$regress_layers --max_seq_len=$max_seq_len
--num_layers=$num_layers --nhead=$nhead
--a_features=$a_features --v_features=$v_features --target=$target --loss_type=$loss_type --loss_weights=$loss_weights
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=10 --niter_decay=20
--num_threads=0 --norm_features=$norm_features --norm_method=trn
--name=$name --use_selfattn=$use_selfattn
--suffix={target}_{v_features}_{a_features}_selfattn-{use_selfattn}_bs{batch_size}_lr{lr}_dp{dropout_rate}_seq{max_seq_len}_reg-{regress_layers}_hidden{hidden_size}_layers{num_layers}_nhead{nhead}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# bash scripts/train_mult.sh both affectnet vggish,wav2vec None n 1 0
# bash scripts/train_mult.sh both affectnet vggish,wav2vec None n 2 0
# bash scripts/train_mult.sh both affectnet vggish,wav2vec None n 3 0

# bash scripts/train_mult.sh both affectnet vggish,wav2vec None y 1 0
# bash scripts/train_mult.sh both affectnet vggish,wav2vec None y 2 0
# bash scripts/train_mult.sh both affectnet vggish,wav2vec None y 3 0


# bash scripts/train_mult.sh both affectnet compare,wav2vec compare n 1 1
# bash scripts/train_mult.sh both affectnet compare,wav2vec compare n 2 1
# bash scripts/train_mult.sh both affectnet compare,wav2vec compare n 3 1

# bash scripts/train_mult.sh both affectnet compare,wav2vec compare y 1 1
# bash scripts/train_mult.sh both affectnet compare,wav2vec compare y 2 1
# bash scripts/train_mult.sh both affectnet compare,wav2vec compare y 3 1



