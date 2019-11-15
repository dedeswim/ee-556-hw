import uuid
import tempfile
import os
import torch
import math

from tqdm import tqdm
from . import plots

temp = tempfile.gettempdir()


class GanTrainer():
    """
    Trains a WGAN with different discriminator regularization strategies

    Args:
        batch_size (int): batch size
        data (callable): real data distribution.
        noise (callable): noise distribution.
        make_gif (bool): Whether to save a 2D snapshot of the samples per
            iteration and compose an animated gif.
        checkpoints (int): number of images to save as pdf
    """

    def __init__(
            self, batch_size, data, noise, make_gif=False):
        self.data = data
        self.noise = noise
        self.id = uuid.uuid4()
        self.snapshots = []
        self.checkpoints = []
        self.batch_size = batch_size
        self.make_gif = make_gif
        self.fixed_real_sample = data.sample((1000,)).cpu().numpy()

    def _snapshot(self, g, ckpt=False):
        """Save an image of the current generated samples"""
        with torch.no_grad():
            gen_sample = g(self.noise.sample((self.batch_size,))).cpu().numpy()
        file_png = os.path.join(
            temp, str(self.id) + '_' + str(len(self.snapshots)) + '.png')
        filename = [file_png]
        if ckpt:
            file_pdf = os.path.join(
                str(self.id) + '_' + str(len(self.checkpoints)) + '.pdf')
            filename.append(file_pdf)
        plots.compare_samples_2D(
            self.fixed_real_sample, gen_sample, filename)
        self.snapshots.append(filename[0])
        if ckpt:
            self.checkpoints.append(filename[1])

    def render_gif(self, output_file, duration):
        """
        Render animated gif based on current snapshots

        Args:
            output_file (str): output_file
            duration (float): output video duration in seconds
        """
        plots.animate(self.snapshots, output_file, duration)

    @staticmethod
    def objective(f, g, data_sample, noise_sample):
        """
        Minimax objective of the GANs

        Args:
            data_sample (torch.tensor): sample from the true distribution
            noise_sample (torch.tensor): sample from the noise distribution
        """

        total_objective = f(data_sample) - f(g(noise_sample))

        return total_objective.mean()

    def simultaneous_update(self, f, g, f_optim, g_optim):
        """
        Update dual variable and generator at the same time
        """

        # Reset gradients
        g_optim.zero_grad()
        f_optim.zero_grad()

        # Get data samples
        data_sample = self.data.sample((self.batch_size,))
        noise_sample = self.noise.sample((self.batch_size,))
        
        # Compute objective and its gradient
        objective = self.objective(f, g, data_sample, noise_sample)
        objective.backward()

        # Optimize functions
        g_optim.step()
        f_optim.step()

    def alternating_update(self, f, g, f_optim, g_optim):
        """
        Update dual variable, then update generator
        """
        # Update f
        # Reset f gradient
        f_optim.zero_grad()

        # Get data samples
        f_data_sample = self.data.sample((self.batch_size,))
        f_noise_sample = self.noise.sample((self.batch_size,))

        # Compute objective and its gradient with old f and g
        objective = self.objective(f, g, f_data_sample, f_noise_sample)
        objective.backward()

        # Optimize f
        f_optim.step()

        # Update g
        # Reset gradient
        g_optim.zero_grad()

        # Get new data samples
        g_data_sample = self.data.sample((self.batch_size,))
        g_noise_sample = self.noise.sample((self.batch_size,))

        # Compute objective and new gradient using new f
        objective = self.objective(f, g, g_data_sample, g_noise_sample)
        objective.backward()

        # Optimize g
        g_optim.step()

    def simultaneous(self, n_iter, f, g, f_optim, g_optim, n_checkpoints):
        """
        Update generator and discriminator a number of iterations
        """
        ckpts = math.floor(n_iter / n_checkpoints)

        for _ in tqdm(range(n_iter)):
            self.simultaneous_update(f, g, f_optim, g_optim)
            f.enforce_lipschitz()

            if self.make_gif:
                if _ % ckpts == 0:
                    self._snapshot(g, ckpt=True)
                else:
                    self._snapshot(g, ckpt=False)

    def alternating(self, n_iter, f, g, f_optim, g_optim, n_checkpoints):
        """
        Update generator and discriminator a number of iterations via
        alternating gradient descent/ascent.

        Args:
            n_iter (int):
            f (nn.Module):
            g (nn.Module):
            f_optim (optim.Optimizer):
            g_optim (optim.Optimizer):
            n_checkpoints (int):
        """
        ckpts = math.floor(n_iter / n_checkpoints)

        for _ in tqdm(range(n_iter)):
            self.alternating_update(f, g, f_optim, g_optim)
            f.enforce_lipschitz()

            if self.make_gif:
                if _ % ckpts == 0:
                    self._snapshot(g, ckpt=True)
                else:
                    self._snapshot(g, ckpt=False)
