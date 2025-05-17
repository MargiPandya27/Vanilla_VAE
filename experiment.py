import os
import csv
import torch
from torch import optim
import pytorch_lightning as pl
from torchvision import utils as vutils


class VAEXperiment(pl.LightningModule):
    def __init__(self, vae_model, params):
        super(VAEXperiment, self).__init__()
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = params.get('retain_first_backpass', False)
        self.automatic_optimization = False

        # Log file setup
        self.loss_log_path = os.path.join(self.params['log_dir'], 'epoch_losses.csv')
        os.makedirs(self.params['log_dir'], exist_ok=True)
        with open(self.loss_log_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'train_loss', 'val_loss'])

        # Track epoch-wise losses
        self.train_loss_epoch = None
        self.val_loss_epoch = None

    def forward(self, input, **kwargs):
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        self.curr_device = x.device
        recons, input, mu, log_var = self.forward(x, labels=labels)
        loss_dict = self.model.loss_function(recons, input, mu, log_var, M_N=self.params['kld_weight'])
        loss = loss_dict['loss']

        # Manual optimization
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        self.log_dict({f"train_{k}": v.item() for k, v in loss_dict.items()}, sync_dist=True)
        self.train_loss_epoch = loss.item()  # For logging in on_train_epoch_end
        return loss

    def on_train_epoch_end(self):
        pass  # Logging happens in on_validation_epoch_end after val loss is known

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        self.curr_device = x.device
        recons, input, mu, log_var = self.forward(x, labels=labels)
        loss_dict = self.model.loss_function(recons, input, mu, log_var, M_N=1.0)
        self.log_dict({f"val_{k}": v.item() for k, v in loss_dict.items()}, sync_dist=True)
        self.val_loss_epoch = loss_dict['loss'].item()

    def on_validation_epoch_end(self):
        # Log only after 10th epoch
        if self.current_epoch >= 1:
            with open(self.loss_log_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([self.current_epoch, self.train_loss_epoch, self.val_loss_epoch])
        self.sample_images()

    def sample_images(self):
        dataloader = self.trainer.datamodule.test_dataloader()
        x, labels = next(iter(dataloader))
        x, labels = x.to(self.curr_device), labels.to(self.curr_device)

        recons = self.model.generate(x, labels=labels)
        vutils.save_image(
            recons.data,
            os.path.join(self.logger.log_dir, "Reconstructions", f"recons_Epoch_{self.current_epoch}.png"),
            normalize=True,
            nrow=12,
        )

        try:
            samples = self.model.sample(144, self.curr_device, labels=labels)
            vutils.save_image(
                samples.cpu().data,
                os.path.join(self.logger.log_dir, "Samples", f"samples_Epoch_{self.current_epoch}.png"),
                normalize=True,
                nrow=12,
            )
        except Exception as e:
            print(f"Sampling error: {e}")

    def configure_optimizers(self):
        optims = []
        scheds = []

        opt1 = optim.Adam(self.model.parameters(), lr=self.params['LR'], weight_decay=self.params['weight_decay'])
        optims.append(opt1)

        if self.params.get('LR_2') and self.params.get('submodel'):
            submodel_params = getattr(self.model, self.params['submodel']).parameters()
            opt2 = optim.Adam(submodel_params, lr=self.params['LR_2'])
            optims.append(opt2)

        if self.params.get('scheduler_gamma'):
            sched1 = optim.lr_scheduler.ExponentialLR(optims[0], gamma=self.params['scheduler_gamma'])
            scheds.append(sched1)

        if len(optims) > 1 and self.params.get('scheduler_gamma_2'):
            sched2 = optim.lr_scheduler.ExponentialLR(optims[1], gamma=self.params['scheduler_gamma_2'])
            scheds.append(sched2)

        return (optims, scheds) if scheds else optims
