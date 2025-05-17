import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *  # Ensure this includes your VAE models
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from custom_data import VAEDataset  # Your custom VAEDataset using MyDataset
from pytorch_lightning.strategies import DDPStrategy
import torch

# Argument parsing
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help='Path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print("Error loading YAML config:", exc)
        exit(1)

# Logger setup
tb_logger = TensorBoardLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['model_params']['name']
)

# Reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

# Load model and experiment
model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model, config['exp_params'])

# ==========================
# ✅ Custom dataset setup
# ==========================
data = VAEDataset(
    data_path="chest_xray_data/chest_xray",
    train_batch_size=config["data_params"].get("train_batch_size", 32),
    val_batch_size=config["data_params"].get("val_batch_size", 32),
    patch_size=config["data_params"].get("patch_size", (224, 224)),
    num_workers=config["data_params"].get("num_workers", 4),
    pin_memory=torch.cuda.is_available()
)

# ==========================
# ✅ Safe Trainer setup
# ==========================
trainer_params = config['trainer_params']
if 'gpus' in trainer_params and (not torch.cuda.is_available()):
    print("⚠️  CUDA not available, overriding trainer config to use CPU")
    trainer_params['accelerator'] = 'cpu'
    trainer_params.pop('gpus', None)

# Prepare trainer arguments
trainer_args = {
    "logger": tb_logger,
    "callbacks": [
        LearningRateMonitor(),
        ModelCheckpoint(
            save_top_k=2,
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            monitor="val_loss",
            save_last=True,
        ),
    ],
    **trainer_params,
}

# Only add strategy if using more than 1 GPU
if torch.cuda.device_count() > 1:
    trainer_args["strategy"] = DDPStrategy(find_unused_parameters=False)

# Create the Trainer instance
runner = Trainer(**trainer_args)

# Output folders
Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

# Start training
print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)
