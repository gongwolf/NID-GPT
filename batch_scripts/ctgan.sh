python scripts/ctgan_train_generation.py --input ./data/UNSW/training_all_classes.csv --output ./data/UNSW/ctgan_synthetic_data_all.csv --meta ./data/UNSW/ctgan_data_all_metadata.json --model ./data/UNSW/all_ctgan_synthesizer.pkl --num_rows 254005 --epochs 5 --use_cuda --load_meta_only

python scripts/ctgan_train_generation.py --input ./data/CICIDS2017/train_all_classes.csv --output ./data/CICIDS2017/ctgan_synthetic_data_all.csv --meta ./data/CICIDS2017/ctgan_data_all_metadata.json --model ./data/CICIDS2017/all_ctgan_synthesizer.pkl --num_rows 282583 --epochs 5 --use_cuda --load_meta_only

python scripts/ctgan_train_generation.py --input ./data/CICDDOS2019/training.csv --output ./data/CICDDOS2019/ctgan_synthetic_data_all.csv --meta ./data/CICDDOS2019/ctgan_data_all_metadata.json --model ./data/CICDDOS2019/all_ctgan_synthesizer.pkl --num_rows 200000 --epochs 5 --use_cuda --load_meta_only
