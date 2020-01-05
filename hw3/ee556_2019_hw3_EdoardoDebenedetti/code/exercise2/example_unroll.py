import sys
sys.path.append('../')
import time
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from skimage.measure import compare_ssim as ssim

from unrolled_network import ResNetDC
from utils import apply_random_mask, psnr, load_image

##############################
# Load image and sample mask #
##############################
model = ResNetDC(2, unroll_depth=5)
model.load_state_dict(torch.load('data/unrolled_nn.pt'))

params = {
    'shape': (256, 256),
    'rate': 0.4,
}

image = load_image('data/gandalf.jpg', params['shape'])

im_us, mask = apply_random_mask(image, params['rate'])

image_torch = torch.tensor(image).view(1, 1, params['shape'][0], params['shape'][1]).float()
mask_torch = torch.tensor(mask).view(1, 1, params['shape'][0], params['shape'][1]).float()
im_us_torch = image_torch * mask_torch

##############################
#      Run the model         #
##############################
with torch.no_grad():
    t_start = time.time()
    reconstruction_nn = model(im_us_torch, mask_torch)
    t_nn = time.time() - t_start
    reconstruction_nn = reconstruction_nn[0, 0, :, :].cpu().numpy()
    psnr_nn = psnr(image, reconstruction_nn)
    ssim_nn = ssim(image, reconstruction_nn)

####################################
# Visualize the results            #
####################################
fig, ax = plt.subplots(1, 4, figsize=(15, 5))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(im_us, cmap='gray')
ax[1].set_title('Original with missing pixels')
ax[2].imshow(reconstruction_nn, cmap="gray")
ax[2].set_title('NN - PSNR = {:.2f}\n SSIM  = {:.2f} - Time: {:.2f}s'.format(psnr_nn, ssim_nn, t_nn))
im = ax[3].imshow(abs(image - reconstruction_nn), cmap="inferno", vmax=.05)
# Plot the colorbar
divider = make_axes_locatable(ax[3])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical');

[axi.set_axis_off() for axi in ax.flatten()]

plt.show()
