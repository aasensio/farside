import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import argparse
import model_unet as model
import scipy.io as io
import h5py
from astropy.io import fits

def dice_coeff(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

class deep_farside(object):
    def __init__(self, parameters):

        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.device = "cpu"
    
        torch.backends.cudnn.benchmark = True

        self.input_file = parameters['input']
        self.output_file = parameters['output']
        self.max_batch = parameters['maxbatch']

        self.format = self.input_file.split('.')[-1]
        self.format_out = self.output_file.split('.')[-1]

        self.verbose = parameters['verbose']

        if (self.verbose):
            print("Input format is {0}".format(self.format))
                        
    def init_model(self, checkpoint=None, n_hidden=8, loss_type='dice'):

        self.loss_type = loss_type
        
        self.checkpoint = checkpoint

        if (self.loss_type == 'bce'):
            self.model = model.UNet(n_channels=11, n_classes=1, n_hidden=n_hidden).to(self.device)

        if (self.loss_type == 'dice'):
            self.model = model.UNet(n_channels=11, n_classes=1, n_hidden=n_hidden).to(self.device)

        if (self.verbose):
            print("=> loading checkpoint {0}.pth".format(self.checkpoint))
        if (self.cuda):
            checkpoint = torch.load('{0}.pth'.format(self.checkpoint))
        else:
            checkpoint = torch.load('{0}.pth'.format(self.checkpoint), map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])        
        
        if (self.verbose):
            print("=> loaded checkpoint {0}.pth".format(self.checkpoint))

    def gen_test_file(self):
        tmp = io.readsav('/scratch1/deepLearning/farside/farside_new.sav')

        phases = np.zeros((20,11,144,120))

        for i in range(20):
            phases[i,:,:,:] = tmp['data_out'][i:i+11,1:145,0:120]

        f = h5py.File('test.h5', 'w')
        db = f.create_dataset('phases', shape=phases.shape)
        db[:] = phases
        f.close()        

    def forward(self):

        if (self.verbose):
            print("Reading input file with the phases...")
        if (self.format == 'sav'):
            phase = io.readsav(self.input_file)['phases']

        if (self.format == 'h5'):
            f = h5py.File(self.input_file, 'r')
            phase = f['phases'][:]
            f.close()

        n_cases, n_phases, nx, ny = phase.shape

        #assert (nx == 144), "x dimension is not 140"
        #assert (ny == 120), "y dimension is not 120"
        assert (n_phases == 11), "n. phases is not 11"

        if (self.verbose):
            print("Normalizing data...")
        phase = np.nan_to_num(phase)

        phase -= np.mean(phase)
        phase /= np.std(phase)

        phase[phase>0] = 0.0

        self.model.eval()

        n_batches = n_cases // self.max_batch
        n_remaining = n_cases % self.max_batch

        if (self.verbose):
            print(" - Total number of maps : {0}".format(n_cases))
            print(" - Total number of batches/remainder : {0}/{1}".format(n_batches, n_remaining))
        
        magnetograms = np.zeros((n_cases,nx,ny))

        left = 0

        if (self.verbose):
            print("Predicting magnetograms...")

        with torch.no_grad():

            for i in range(n_batches):                
                right = left + self.max_batch
                phases = torch.from_numpy(phase[left:right,:,:,:].astype('float32')).to(self.device)                
                output = self.model(phases)

                magnetograms[left:right,:,:] = output.cpu().numpy()[:,0,:,:]

                left += self.max_batch

            if (n_remaining != 0):
                right = left + n_remaining
                phases = torch.from_numpy(phase[left:right,:,:,:].astype('float32')).to(self.device)                
                output = self.model(phases)
                magnetograms[left:right,:,:] = output.cpu().numpy()[:,0,:,:]
            

        if (self.verbose):
            print("Saving output file...")

        if (self.format_out == 'h5'):
            f = h5py.File(self.output_file, 'w')
            db = f.create_dataset('magnetogram', shape=magnetograms.shape)
            db[:] = magnetograms
            f.close()

        if (self.format_out == 'fits'):
            hdu = fits.PrimaryHDU(magnetograms)
            hdu.writeto(self.output_file)

                 
if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='''
     Predict a farside magnetogram from the computed phases.
     The input phases needs to be in a file (IDL save file or HDF5) and should contain
     a single dataset with name `phases` of size [n_cases,11,144,120]
     ''')
    parser.add_argument('-i','--input', help='Input file', required=True)
    parser.add_argument('-o','--output', help='Output file', required=True)
    parser.add_argument('-b','--maxbatch', help='Maximum batch size', default=10)
    parser.add_argument('-v','--verbose', help='Verbose', default=False)
    parsed = vars(parser.parse_args())
    
    deep_farside_network = deep_farside(parsed)
    
    # Best so far with BCE
    deep_farside_network.init_model(checkpoint='2019-04-02-11:27:48_hid_16_lr_0.0003_wd_0.0', n_hidden=16, loss_type='bce')

    # deep_farside_network.gen_test_file()
    
    deep_farside_network.forward()
