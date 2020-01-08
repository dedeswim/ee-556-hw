import numpy as np
import time
import os
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from utils import apply_random_mask, psnr
from collections import OrderedDict




# ============================================================================
# DEFINING THE MODEL
# ============================================================================

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
        residual block of the resnet, without max pooling
    """
    def __init__(self, inplanes, outplanes):
        super(BasicBlock, self).__init__()
        
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, outplanes)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outplanes, outplanes)
        self.bn2 = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


    
class ResNetDC(nn.Module):
    """
        ResNetDC implements a modified resnet for regression. 
        It is based on the architecture of 
            Mardani, Morteza, et al. "Neural proximal gradient descent for compressive imaging." 
            Advances in Neural Information Processing Systems. 2018.
        where the tensorflow code available at https://github.com/MortezaMardani/Neural-PGD was
        implemented to pytorch.
    """
    def __init__(self, nblocks, max_channel=64, unroll_depth = 2, width_per_group=64):    
        super(ResNetDC, self).__init__()
        self.convblocklist = nn.ModuleList([])
        self.num_blocks = nblocks
        self.convblocklist.append(BasicBlock(1, max_channel))
                                  
        for i in range(nblocks-1):
            self.convblocklist.append(BasicBlock(max_channel, max_channel))
        
        self.conv1 = conv1x1(max_channel, max_channel, stride = 1)                              
        self.relu =  nn.ReLU(inplace=True)
        self.conv2 = conv1x1(max_channel, max_channel, stride = 1)         
        self.conv3 = conv1x1(max_channel, 1, stride = 1)          
        
        self.T = unroll_depth
                                  
    def forward(self, x, mask):
                                  
        x0 = x*mask
        for t in range(self.T):
            for i in range(self.num_blocks):
                x = self.convblocklist[i](x)
            x = self.relu(self.conv1(x))                           
            x = self.relu(self.conv2(x))
            x = self.conv3(x)

            x = (1.-mask)*x + x0
            #print((1.-mask).sum(), mask.sum())
        return x
    
    def prepare(self):
        # original saved file with DataParallel
        state_dict = torch.load('data/nn_weights.pt')['model']
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        self.load_state_dict(new_state_dict)

# ============================================================================
# TRAINING FUNCTION
# ============================================================================
    
def train_epoch(args, epoch, model, data_loader, optimizer):
    model.train()
    avg_loss = 0.
    start_epoch = start_iter = time.time()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):
        target, _ = data
        
        input, mask = apply_random_mask(target,args['rate'])
        target = torch.tensor(target).to(args['device'])
        input = torch.tensor(input).to(args['device'])
        mask = torch.tensor(mask).to(args['device'])
        output = model(input,mask)#.squeeze(1)

        loss = F.mse_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #if iter > 10:
        #    break
        if iter % args['report_interval'] == 0:
            print(  'Epoch = [{:3d}/{:3d}] '.format(epoch,args['num_epochs'])+
                'Iter = [{:4d}/{:4d}] '.format(iter,len(data_loader))+
                'Loss = {:.4g} '.format(loss.item())+
                'Time = {:.4f}s'.format(time.time() - start_iter)
            )
        start_iter = time.time()
    return avg_loss, time.time() - start_epoch

# ============================================================================
# TESTING FUNCTION
# ============================================================================


def evaluate(args, epoch, model, data_loader):
    model.eval()
    losses = []
    start_epoch = time.time()
    psnr_tot = []
    with torch.no_grad():
        
        for iter, data in enumerate(data_loader):
            target, _ = data
            input, mask = apply_random_mask(target,args['rate'])
            target = torch.tensor(target).to(args['device'])
            input = torch.tensor(input).to(args['device'])
            mask = torch.tensor(mask).to(args['device'])
            output = model(input,mask)#.squeeze(1)

            loss = F.mse_loss(output, target , reduction='sum')
            losses.append(loss.item())
            psnr_tot.append(np.mean( [psnr(t.cpu().numpy(),o.cpu().numpy()) for t,o in zip(target,output)]))
            #if iter > 10:
            #    break

    return np.mean(losses), np.mean(psnr_tot), time.time() - start_epoch


# ============================================================================
# HELPER
# ============================================================================


def save_model(args, exp_dir, epoch, model, optimizer):
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    torch.save({'epoch': epoch,'args': args,'model': model.state_dict(), 
                'optimizer': optimizer.state_dict(),'exp_dir': exp_dir},
                f=exp_dir + '/model_{}.pt'.format(epoch))




  


