import os
import torch
import numpy as np, random
import re
from pathlib import Path


def mount_drive():
    from google.colab import drive
    drive.mount('/content/drive')

def install_and_upgrade_kaggle():
    os.system('pip install --upgrade --force-reinstall --no-deps  kaggle > /dev/null')
    os.system('mkdir ~/.kaggle')
    os.system('cp "/content/drive/My Drive/Kaggle/kaggle.json" ~/.kaggle/')
    os.system('chmod 600 ~/.kaggle/kaggle.json')


def download_comp_dataset(comp_data_name, save_root, override=False, unzip=True):
    save_root = Path(save_root)
    folder = save_root / comp_data_name
    if folder.exists() and not override:
        raise ValueError(f"The file '{folder}' already exists")

    os.system(f'kaggle competitions download -c {comp_data_name}')

    if unzip:
        os.system(f'mkdir -p "{folder}"')
        os.system(f'unzip {comp_data_name}.zip -d "{folder}"')


def download_user_datasets():

    DATA_URLs = [
        "julian3833/jigsaw-toxic-comment-classification-challenge",
    ]

    for url in DATA_URLs:
        name = url.split("/")[-1]
        filepath = f'/content/{name}.zip' 
        folder = f"/content/datasets/{name}"
        # folder = "/content"
        if os.path.exists(folder):
          continue
          
        os.system('kaggle datasets download -o -d {url}')
        os.system('unzip -o {filepath} -d {folder}')
        os.system('rm {filepath}')


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def slugify(s):
    return re.sub(r"[^\w\-_]", "_", s)


def get_config_as_param(configs):
    config_ok_types = (
        int,
        float,
        dict,
        str,
        tuple,
        list,
        np.ndarray,
        Path,
        torch.device
    )

    config_dict = {
        key: getattr(configs, key) for key in configs.__dir__()
    }
    
    config_dict = {
        key : val for key, val in config_dict.items() if isinstance(val, config_ok_types) and not key.startswith("__")
    }
    
    return config_dict