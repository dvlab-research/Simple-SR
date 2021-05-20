python3 -m torch.distributed.launch --nproc_per_node=$1 --master_port=$2 train.py
