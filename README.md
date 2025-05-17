## Implementation Details


```bash
!git clone repo
%cd Vanilla_VAE
!pip install -r requirements.txt

!pip install pytorch-lightning==1.5.0 (Use if required !pip install --upgrade pytorch-lightning)

Download the data (keep it outside the main repo)


To train:
!python PyTorch-VAE/run.py -c PyTorch-VAE/configs/vae.yaml

For inference:
!python PyTorch-VAE/run_test.py -c PyTorch-VAE/configs/vae.yaml

```








