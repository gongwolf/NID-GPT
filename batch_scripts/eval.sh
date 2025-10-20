python scripts/evaluation.py --train data/UNSW/ctgan_synthetic_data_all.csv --test data/UNSW/testing_all_classes.csv --model mlp --dataset unsw_all
python scripts/evaluation.py --train data/UNSW/ddpm_synthetic_data_all.csv --test data/UNSW/testing_all_classes.csv --model mlp --dataset unsw_all


python scripts/evaluation.py --train data/CICIDS2017/ctgan_synthetic_data_all.csv --test data/CICIDS2017/test_all_classes.csv --model mlp --dataset cicids2017_all
python scripts/evaluation.py --train data/CICIDS2017/ddpm_synthetic_data_all.csv --test data/CICIDS2017/test_all_classes.csv --model mlp --dataset cicids2017_all

python scripts/evaluation.py --train data/CICDDOS2019/ctgan_synthetic_data_all.csv --test data/CICDDOS2019/testing.csv --model mlp --dataset cicddos2019_all
python scripts/evaluation.py --train data/CICDDOS2019/ddpm_synthetic_data_all.csv --test data/CICDDOS2019/testing.csv --model mlp --dataset cicddos2019_all