import os
import yaml
import argparse
from pathlib import Path
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from models import vae_models
from experiment import VAEXperiment
from custom_data import VAEDataset
from metrics import calculate_fid, calculate_inception_score
from torchvision.utils import save_image

# -------------------------------
# Argument Parsing
# -------------------------------
parser = argparse.ArgumentParser(description='Testing a trained VAE model')
parser.add_argument('--config', '-c', dest="filename", metavar='FILE', help='Path to the config file',
                    default='configs/vae.yaml')
args = parser.parse_args()

with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print("Error loading YAML config:", exc)
        exit(1)

# -------------------------------
# Logger & Reproducibility
# -------------------------------
tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'])
seed_everything(config['exp_params']['manual_seed'], True)

# -------------------------------
# Model & Experiment
# -------------------------------
# -------------------------------
# Model & Experiment
# -------------------------------
#model = vae_models[config['model_params']['name']](**config['model_params'])

# # Load checkpoint if specified in config and exists
# model_class = vae_models[config['model_params']['name']]
# model = model_class(**config['model_params'])
# checkpoint_path = config.get('checkpoint_path', None)

# checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# state_dict = checkpoint['state_dict']

# # Strip 'model.' prefix from keys if present
# new_state_dict = {}
# for k, v in state_dict.items():
#     if k.startswith('model.'):
#         new_key = k[len('model.'):]  # remove 'model.' prefix
#     else:
#         new_key = k
#     new_state_dict[new_key] = v

# model.load_state_dict(new_state_dict)



# if checkpoint_path and os.path.isfile(checkpoint_path):
#     print(f"Loading model weights from checkpoint: {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
#     model.load_state_dict(checkpoint['state_dict'])  # or checkpoint directly depending on how saved
# else:
#     print("No checkpoint found, using untrained model")

# -------------------------------
# Load Trained VAE Model
# -------------------------------
model_class = vae_models[config['model_params']['name']]
model = model_class(**config['model_params'])

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load checkpoint
checkpoint_path = config.get("checkpoint_path", "/content/drive/MyDrive/285/PyTorch-VAE/last.ckpt")
if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

print(f"✅ Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)

# Handle cases where checkpoint contains a state_dict key
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

# Remove 'model.' prefix if present in keys
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("model.", "") if k.startswith("model.") else k
    new_state_dict[new_key] = v

# Load state dict into model
missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

# Log any key issues
if missing_keys:
    print(f"⚠️ Missing keys when loading model: {missing_keys}")
if unexpected_keys:
    print(f"⚠️ Unexpected keys in checkpoint: {unexpected_keys}")
print("✅ Model loaded successfully.")

# Wrap in experiment
experiment = VAEXperiment(model, config['exp_params'])


# -------------------------------
# Dataset
# -------------------------------
data = VAEDataset(
    data_path="chest_xray_data/chest_xray",
    train_batch_size=config["data_params"].get("train_batch_size", 32),
    val_batch_size=config["data_params"].get("val_batch_size", 32),
    test_batch_size=config["data_params"].get("test_batch_size", 32),
    patch_size=config["data_params"].get("patch_size", (224, 224)),
    num_workers=config["data_params"].get("num_workers", 4),
    pin_memory=torch.cuda.is_available()
)

# -------------------------------
# Trainer Configuration
# -------------------------------
trainer_params = config['trainer_params']
if 'gpus' in trainer_params and not torch.cuda.is_available():
    print("⚠️ CUDA not available, forcing CPU.")
    trainer_params['accelerator'] = 'cpu'
    trainer_params.pop('gpus', None)

trainer = Trainer(
    logger=tb_logger,
    callbacks=[
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            save_top_k=2,
            monitor="val_loss",
            dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
            save_last=True
        )
    ],
    **trainer_params
)

# -------------------------------
# Run Test
# -------------------------------
print(f"\n======= Testing {config['model_params']['name']} =======")
results = trainer.test(experiment, datamodule=data)
print("Test Loss Metrics:", results)

# -------------------------------
# Compute FID & Inception Score
# -------------------------------
print("✅ Computing Inception Score & FID...")

# Prepare real and generated images
dataloader = data.test_dataloader()
real_imgs, _ = next(iter(dataloader))
real_imgs = real_imgs.to(next(model.parameters()).device)

batch_size = real_imgs.size(0)
# with torch.no_grad():
#     device = next(model.parameters()).device

#     batch_size = 64  # or however many images you want to generate
#     latent_dim = model.latent_dim  # use the correct latent dim from your model

#     z = torch.randn(batch_size, latent_dim).to(model.device)  # ✅ Correct shape
#     fake_imgs = model.decode(z)  # ✅ This will now work

#     #z = model.sample(batch_size, current_device=device)  # ✅ Correct

#     #z = model.sample(len(real_imgs)).to(model.device)
#     #fake_imgs = model.decode(z)


with torch.no_grad():
    device = next(model.parameters()).device

    batch_size = 64  # or however many images you want to generate
    latent_dim = model.latent_dim  # get latent dim

    z = torch.randn(batch_size, latent_dim).to(device)
    fake_imgs = model.decode(z)

    # Save generated images
    save_dir = os.path.join(tb_logger.log_dir, "generated_images")
    os.makedirs(save_dir, exist_ok=True)

    # Rescale from [-1, 1] to [0, 1]
    fake_imgs = (fake_imgs + 1) / 2
    fake_imgs = torch.clamp(fake_imgs, 0, 1)

    save_image(fake_imgs, os.path.join(save_dir, "fake_images_grid.png"), nrow=8)
    for i, img in enumerate(fake_imgs):
        save_image(img, os.path.join(save_dir, f"fake_img_{i}.png"))


fid = calculate_fid(real_imgs, fake_imgs, device=device)
inception_score = calculate_inception_score(fake_imgs, device=device)

# Save Metrics
metrics = {
    'FID': float(fid),
    'Inception_Score': float(inception_score)
}
print("✅ Test Metrics:", metrics)

with open(os.path.join(tb_logger.log_dir, "test_metrics.yaml"), "w") as f:
    yaml.dump(metrics, f)

print(f"📄 Saved metrics to {tb_logger.log_dir}/test_metrics.yaml")
