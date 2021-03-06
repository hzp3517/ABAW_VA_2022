set -e
name=transformer_lstm
batch_size=16
lr=2e-5
dropout=0.3
regress_layers=256,256
max_seq_len=250
hidden_size=256
num_layers=4
ffn_dim=1024
nhead=4

loss_weights=1
loss_type=batch_ccc
log_dir=./logs/resplit/telstm
checkpoints_dir=./checkpoints/resplit/telstm

target=$1
feature=$2
norm_features=$3
cv=$4
residual=$5
run_idx=$6
gpu_ids=$7


cmd="python train_lyc_seed.py --dataset_mode=seq_resplit --model=transformer_lstm --gpu_ids=$gpu_ids
--log_dir=$log_dir --checkpoints_dir=$checkpoints_dir --print_freq=2
--hidden_size=$hidden_size --regress_layers=$regress_layers --max_seq_len=$max_seq_len
--num_layers=$num_layers --ffn_dim=$ffn_dim --nhead=$nhead
--feature_set=$feature --target=$target --loss_type=$loss_type --loss_weights=$loss_weights --use_pe
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=10 --niter_decay=20
--num_threads=0 --norm_features=$norm_features --norm_method=trn
--pth_path=$pth_path --residual=$residual --cv=$cv
--name=$name --encoder_type=transformer
--suffix={target}_{feature_set}_cv{cv}_res-{residual}_bs{batch_size}_lr{lr}_dp{dropout_rate}_seq{max_seq_len}_reg-{regress_layers}_hidden{hidden_size}_layers{num_layers}_ffn{ffn_dim}_nhead{nhead}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh

# bash scripts/train_resplit_telstm.sh both affectnet,vggish,wav2vec None 1 n 1 0
# bash scripts/train_resplit_telstm.sh both affectnet,vggish,wav2vec None 1 n 2 0
# bash scripts/train_resplit_telstm.sh both affectnet,vggish,wav2vec None 2 n 1 0
# bash scripts/train_resplit_telstm.sh both affectnet,vggish,wav2vec None 2 n 2 0
# bash scripts/train_resplit_telstm.sh both affectnet,vggish,wav2vec None 3 n 1 0
# bash scripts/train_resplit_telstm.sh both affectnet,vggish,wav2vec None 3 n 2 3
# bash scripts/train_resplit_telstm.sh both affectnet,vggish,wav2vec None 4 n 1 1
# bash scripts/train_resplit_telstm.sh both affectnet,vggish,wav2vec None 4 n 2 1
# bash scripts/train_resplit_telstm.sh both affectnet,vggish,wav2vec None 5 n 1 1
# bash scripts/train_resplit_telstm.sh both affectnet,vggish,wav2vec None 5 n 2 1





# bash scripts/train_resplit_telstm.sh both affectnet,compare,wav2vec compare 1 n 1 3
# bash scripts/train_resplit_telstm.sh both affectnet,compare,wav2vec compare 1 n 2 4