# GReaT Framework

![LLM data generation](GReaT_Evaluation.jpg)

---
## Dataset
You can download the processed dataset from [here](https://eltnmsu-my.sharepoint.com/:f:/g/personal/hcao_nmsu_edu/Etuw1nXMxgZAixSU405NdEkBsNo8AVsR2X41lfv1gDD4yA?e=Aec8dD).

---

## Pre-trained Large Language Model

You can download the original pre-trained LLM from [GPT-Neo-125m](https://huggingface.co/EleutherAI/gpt-neo-125m/tree/main), you can find more pre-trained model from [HuggingFace](https://huggingface.co/).

You also can download the fine-tuned LLM from our [OneDrive](https://eltnmsu-my.sharepoint.com/:f:/g/personal/hcao_nmsu_edu/EhTZWB27vSJLi1zSXNeQDlcBXBusi7XVo41Rjo3SC0brVQ?e=MPnxrB).

---

### Install be_great:
```aiignore
pip install be-great
```
Requires a `Python version >= 3.9`. If you have problem to set up environment for conda from the linux server, you can get help from [here](https://github.com/JiefeiLiu/Federated_learning_env_set_up). 

You can find the original GReaT framework from [here](https://github.com/tabularis-ai/be_great).

## Quick start
1. Download the dataset and LLM model, and put then into the current directory. 
2. (optional) The sample [code](https://github.com/gongwolf/NID-GPT/blob/main/data_process/CICIDS2017_change_column_names1.ipynb) to replace the feature names to full explanation of the CICIDS2017 dataset.
3. Change the loading paths for data, llm, and saving path in [llm_gpt_neo.ipynb](https://github.com/gongwolf/NID-GPT/blob/main/GReaT/llm_gpt_neo.ipynb). Example shows below:

```aiignore
data = pd.read_csv('<Tabular_data.csv>')
model = GReaT(llm='<path_to_downloaded_llm_model>',experiment_dir="<folder_to_save_check_point>", batch_size=4, epochs=5, save_steps=5000)
```

- Tabular_data.csv: download the processed .csv file, put it into the same directory with the python script. 
- path_to_downloaded_llm: download the folder of LLM, put it into the same directory with the python script. 
- folder_to_save_check_point: Create a folder for the checkpoint LLM model saving. 
- batch_size: A subset of the training dataset used in one iteration of training, if GPU memory is small, suggest use smaller batch size like 1 or 2. 
- epochs: Total epochs to fine-tune the LLM, each epoch takes more than a month to train. 
- save_steps: Saving the checkpoints, if the training process get killed or server reboot, you can load the checkpoint and keep training instead of re-train. 