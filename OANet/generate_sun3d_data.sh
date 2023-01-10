cd ./dump_match

python extract_feature.py --input_path=/home/OANet/author/raw_data/sun3d_train # Extract features from 3dsun_train dataset and offline
python extract_feature.py --input_path=/home/OANet/author/raw_data/sun3d_test # Extract features from 3dsun_test dataset and offline

python sun3d.py # Generate the final dataset file