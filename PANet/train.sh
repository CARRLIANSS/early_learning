log_name=`date +%Y%m%d-%H:%M:%S`

python -u train_single_gpu.py 2>&1 | tee ./logs/train-"$log_name".log