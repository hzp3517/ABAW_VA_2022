set -e
name=transformer
batch_size=16
lr=2e-5
dropout=0.3
regress_layers=256,256
win_len=250
hop_len=50
hidden_size=256
num_layers=4
ffn_dim=1024
nhead=4

loss_weights=1
loss_type=batch_ccc
log_dir=./logs/3-21/slide
checkpoints_dir=./checkpoints/3-21/slide

target=$1
feature=$2
norm_features=$3
run_idx=$4
gpu_ids=$5


cmd="python train_slide_seed.py --dataset_mode=seq_slide --model=transformer_slide --gpu_ids=$gpu_ids
--log_dir=$log_dir --checkpoints_dir=$checkpoints_dir --print_freq=100
--hidden_size=$hidden_size --regress_layers=$regress_layers --win_len=$win_len --hop_len=$hop_len
--num_layers=$num_layers --ffn_dim=$ffn_dim --nhead=$nhead
--feature_set=$feature --target=$target --loss_type=$loss_type --loss_weights=$loss_weights --use_pe
--batch_size=$batch_size --lr=$lr --dropout_rate=$dropout --run_idx=$run_idx --verbose
--niter=10 --niter_decay=20
--num_threads=0 --norm_features=$norm_features --norm_method=trn
--name=$name --encoder_type=transformer
--suffix={target}_{feature_set}_bs{batch_size}_lr{lr}_dp{dropout_rate}_win{win_len}_hop{hop_len}_reg-{regress_layers}_hidden{hidden_size}_layers{num_layers}_ffn{ffn_dim}_nhead{nhead}_{loss_type}_run{run_idx}"

echo "-------------------------------------------------------------------------------------"
echo $cmd | sh


# bash scripts/train_transformer_slide.sh both affectnet,vggish,wav2vec None 1 0
# bash scripts/train_transformer_slide.sh both affectnet,vggish,wav2vec None 2 1
# bash scripts/train_transformer_slide.sh both affectnet,vggish,wav2vec None 3 3
# bash scripts/train_transformer_slide.sh both affectnet,compare,wav2vec compare 1 4
# bash scripts/train_transformer_slide.sh both affectnet,compare,wav2vec compare 2 5
# bash scripts/train_transformer_slide.sh both affectnet,compare,wav2vec compare 3 6
