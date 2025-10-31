# Training BE_GREAT (LLM model) and Sampling Synthetic Flow Data on the UNSW-NB15 Dataset
python GReaT/Train_and_generate.py --epochs 5 --batch_size 2 --n_samples 50000 --device cuda --data_path data/UNSW/training_all_classes_sample.csv

# Training BE_GREAT (LLM model) and Sampling Synthetic Flow Data on the CICIDS2017 Dataset
python GReaT/Train_and_generate.py --epochs 5 --batch_size 2 --n_samples 50000 --device cuda --data_path data/CICIDS2017/train_all_classes.csv

# Training BE_GREAT (LLM model) and Sampling Synthetic Flow Data on the CICDDOS2019 Dataset
python GReaT/Train_and_generate.py --epochs 5 --batch_size 2 --n_samples 50000 --device cuda --data_path data/CICIDDOS2019/training.csv