log_name=`date +%Y%m%d-%H:%M:%S`

python -u test.py 2>&1 | tee ./logs/test-"$log_name".log