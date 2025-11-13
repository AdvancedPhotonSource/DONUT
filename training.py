# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com/AdvancedPhotonSource/DONUT/blob/main/LICENSE

import os
import torch
import numpy as np
from math import *
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import Normalize
import matplotlib.cm as cm

class Trainer(object):
    def __init__(self, model, ngpus, device, model_save_path):
        super().__init__()
        self.model = model
        self.ngpus = ngpus
        self.device = device
        self.model_save_path = model_save_path
        
    def update_saved_model(self, name, run_num):
        """Update saved model if validation loss is minimum"""
        if not os.path.isdir(self.model_save_path):
            os.mkdir(self.model_save_path)
        #for f in os.listdir(self.model_save_path):
            #os.remove(os.path.join(self.model_save_path, f)) # This removes all previously save models
        run_path = os.path.join(self.model_save_path, 'run' + str(run_num))
        if not os.path.isdir(run_path):
            os.mkdir(run_path)
        if self.ngpus > 1:
            torch.save(self.model.module.state_dict(), os.path.join(run_path, name + '.pth'))
            torch.save(self.model.module.encoder.state_dict(), os.path.join(run_path, name + '_encoder.pth'))
        else:
            torch.save(self.model.state_dict(), os.path.join(run_path, name + '.pth'))
            torch.save(self.model.encoder.state_dict(), os.path.join(run_path, name + '_encoder.pth'))

    def plot_while_training(self, gt_im_arr, output_im_arr, recon_im_arr):
        """Plot some input, output, and decoder reconstructed images during training"""
        vmin = 1
        vmax = 7
        normalizer = colors.LogNorm(vmin, vmax)
        im = cm.ScalarMappable(norm=normalizer, cmap='jet')

        f, ax = plt.subplots(figsize=(11, 6.5), nrows=3, ncols=4)

        ax[0, 0].imshow(gt_im_arr[0], interpolation='none', cmap='jet', norm=normalizer)
        ax[0, 0].set_xticks([])
        ax[0, 0].set_yticks([])
        ax[0, 0].set_title('Input 0')

        ax[0, 1].imshow(gt_im_arr[4], interpolation='none', cmap='jet', norm=normalizer)
        ax[0, 1].set_xticks([])
        ax[0, 1].set_yticks([])
        ax[0, 1].set_title('Input 1')

        ax[0, 2].imshow(gt_im_arr[8], interpolation='none', cmap='jet', norm=normalizer)
        ax[0, 2].set_xticks([])
        ax[0, 2].set_yticks([])
        ax[0, 2].set_title('Input 2')

        ax[0, 3].imshow(gt_im_arr[12], interpolation='none', cmap='jet', norm=normalizer)
        ax[0, 3].set_xticks([])
        ax[0, 3].set_yticks([])
        ax[0, 3].set_title('Input 3')

        ax[1, 0].imshow(output_im_arr[0], interpolation='none', cmap='jet', norm=normalizer)
        ax[1, 0].set_xticks([])
        ax[1, 0].set_yticks([])
        ax[1, 0].set_title('Pred. 0')

        ax[1, 1].imshow(output_im_arr[4], interpolation='none', cmap='jet', norm=normalizer)
        ax[1, 1].set_xticks([])
        ax[1, 1].set_yticks([])
        ax[1, 1].set_title('Pred. 1')

        ax[1, 2].imshow(output_im_arr[8], interpolation='none', cmap='jet', norm=normalizer)
        ax[1, 2].set_xticks([])
        ax[1, 2].set_yticks([])
        ax[1, 2].set_title('Pred. 2')

        ax[1, 3].imshow(output_im_arr[12], interpolation='none', cmap='jet', norm=normalizer)
        ax[1, 3].set_xticks([])
        ax[1, 3].set_yticks([])
        ax[1, 3].set_title('Pred. 3')

        ax[2, 0].imshow(recon_im_arr[0], interpolation='none', cmap='jet', norm=normalizer)
        ax[2, 0].set_xticks([])
        ax[2, 0].set_yticks([])
        ax[2, 0].set_title('Pred. 0')

        ax[2, 1].imshow(recon_im_arr[4], interpolation='none', cmap='jet', norm=normalizer)
        ax[2, 1].set_xticks([])
        ax[2, 1].set_yticks([])
        ax[2, 1].set_title('Pred. 1')

        ax[2, 2].imshow(recon_im_arr[8], interpolation='none', cmap='jet', norm=normalizer)
        ax[2, 2].set_xticks([])
        ax[2, 2].set_yticks([])
        ax[2, 2].set_title('Pred. 2')

        ax[2, 3].imshow(recon_im_arr[12], interpolation='none', cmap='jet', norm=normalizer)
        ax[2, 3].set_xticks([])
        ax[2, 3].set_yticks([])
        ax[2, 3].set_title('Pred. 3')

        f.subplots_adjust(hspace=0.15, wspace=0.05, right=0.9)
        cbar_ax = f.add_axes([0.90, 0.12, 0.02, 0.75])
        f.colorbar(im, cax=cbar_ax)

        return f

    def generate_state_dict(self, epoch_num, metrics, optimizer, scheduler=None):
            """
            Returns a dictionary of the state_dicts of all states but not the model.
            """
            state = {
                'current_epoch': epoch_num + 1,
                'loss_tracker': metrics,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            }
            return state

    def save_model_and_states_checkpoint(self, run_num, epoch_num, metrics, optimizer, scheduler=None):
        """Save a checkpoint state that can be loaded to continue training."""
    #     if not self.gatekeeper.should_proceed(gate_kept=True):
    #         return
        state_dict = self.generate_state_dict(epoch_num, metrics, optimizer, scheduler=None)
        state_path = os.path.join(self.model_save_path, 'run' + str(run_num))
        self.update_saved_model('checkpoint_model', run_num)
        torch.save(state_dict, os.path.join(state_path, 'checkpoint.state'))

    def load_state_checkpoint(self, run_num, optimizer, scheduler=None):
        """Load everything but the model."""
    #     if self.configs.checkpoint_dir is None:
    #         return
        checkpoint_path = os.path.join(self.model_save_path, 'run' + str(run_num))
        checkpoint_fname = os.path.join(checkpoint_path, 'checkpoint.state')
        try:
            os.path.exists(checkpoint_fname)
        except:
            print('Checkpoint not found in {}.'.format(checkpoint_fname))
        state_dict = torch.load(checkpoint_fname)
        current_epoch = state_dict['current_epoch']
        metrics = state_dict['loss_tracker']
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        if state_dict['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        return current_epoch, metrics, optimizer, scheduler

    def train(self, trainloader, criterion, optimizer, weights, metrics, plot):
        """Training loop"""
        running_loss = 0.0
        running_decoder_loss = 0.0
        running_sim_loss = 0.0

        for i, data in enumerate(trainloader):
            inputs = data['image'].to(self.device)
          
            recon_img, sim_img, lattice = self.model(inputs)

            loss_decoder = criterion(recon_img, inputs)
            loss_sim = criterion(sim_img, inputs)
            loss = (weights[0] * loss_decoder) + (weights[1] * loss_sim)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().item()
            running_decoder_loss += loss_decoder.detach().item()
            running_sim_loss += loss_sim.detach().item()
            
            if plot==True:
                if i == len(trainloader)//2:
                    target = inputs.squeeze().detach().cpu().numpy()
                    output = sim_img.squeeze().detach().cpu().numpy()
                    recon_output = recon_img.squeeze().detach().cpu().numpy()
                    self.plot_while_training(target, output, recon_output).show()

        metrics['losses'].append(running_loss/i)
        metrics['decoder_loss'].append(running_decoder_loss/i)
        metrics['sim_loss'].append(running_sim_loss/i)

    def validate(self, validloader, criterion, optimizer, weights, metrics, run_num, scheduler=None):
        """Validation loop"""
        tot_val_loss = 0.0

        for j, sample in enumerate(validloader):
            images = sample['image'].to(self.device)
            
            pred_recon, pred_sim, pred_lattice = self.model(images)

            val_loss_decoder = criterion(pred_recon, images)
            val_loss_sim = criterion(pred_sim, images)
            val_loss = (weights[0] * val_loss_decoder) + (weights[1] * val_loss_sim)
            tot_val_loss += val_loss.detach().item()

        metrics['val_losses'].append(tot_val_loss/j)
        if scheduler:
            scheduler.step(tot_val_loss/j)
            metrics['enc_lrs'].append(optimizer.param_groups[0]['lr'])
            metrics['dec_lrs'].append(optimizer.param_groups[1]['lr'])
            
        if (tot_val_loss/j < metrics['best_val_loss']):
            print("Saving improved model after Val. Loss improved from %.5f to %.5f" 
                  % (metrics['best_val_loss'], tot_val_loss/j))
            metrics['best_val_loss'] = tot_val_loss/j
            self.update_saved_model('best_model', run_num)