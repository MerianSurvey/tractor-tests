#===============================#
# Tractor on Merian using mpi4Py 
#===============================#

# import functions 
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import ion

import astropy.units as u
from astropy import wcs
from astropy.io import fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
from astropy.visualization import make_lupton_rgb
from astropy.utils.data import download_file, clear_download_cache

import kuaizi
from kuaizi.display import display_single, SEG_CMAP
from kuaizi.download import download_decals_cutout
from kuaizi import DECaLS_pixel_scale, DECaLS_zeropoint
from kuaizi import HSC_zeropoint, HSC_pixel_scale
from kuaizi.detection import Data
from kuaizi.utils import padding_PSF
from kuaizi.tractor.utils import tractor_hsc_sep_blob_by_blob

import copy

from mpi4py import MPI

import pickle

# from IPython.display import clear_output

import pandas as pd

import os

import sys 

import time 

ion()

sys.path.append("/home/dcblanco/astrometry.net-0.85/")

# new obj_cat
obj_cat = Table.read(
    f'/home/dcblanco/Research/Data/broadcut_GAMA09H_cosmos_match.fits'
    ) # broadcut catalog
galaxies_number = len(obj_cat)

channels = 'grizy'
ref_filt = 'i'
forced_channels = [filt for filt in channels if filt != ref_filt]

def run_tractor_model(galaxy_ID): 
    print('start run for galaxy No.', galaxy_ID)
    time_start = time.time()
 
    # object ID
    obj = obj_cat[galaxy_ID]
    obj_id = obj['object_id']
    
    # folder 
    folder = obj['prefix'][40:-49]

    os.chdir(f'/data/users/dcblanco/Merian/')
    if not os.path.isdir(f'/data/users/dcblanco/Merian/{folder}/{obj_id}/tractor'): 
        os.mkdir(f'/data/users/dcblanco/Merian/{folder}/{obj_id}/tractor')
    os.chdir(f'{folder}/{obj_id}/tractor')
    
    coord = SkyCoord(obj['ra'], obj['dec'], frame = 'icrs', unit = 'deg')
    
    cutout = [
        fits.open(
            f'/data/groups/leauthaud/Merian/poststamps/cosmos_broad/{folder}/{obj_id}/hsc/cosmos_{obj_id}_{filt}.fits'
        ) for filt in channels
    ]
    
    psf_list = [
        fits.open(
            f'/data/groups/leauthaud/Merian/poststamps/cosmos_broad/{folder}/{obj_id}/hsc/cosmos_{obj_id}_{filt}_psf.fits'
        ) for filt in channels 
    ]
    
    # reconstruct data 
    images = np.array([hdu[1].data for hdu in cutout])
    
    w = wcs.WCS(cutout[0][1].header) # note: all bands share the same WCS here
    
    filters = list(channels)
    
    weights = 1 / np.array([hdu[3].data for hdu in cutout])
    
    psf_pad = padding_PSF(psf_list) # padding PSF cutouts from HSC
    
    hsc_data = Data(images = images, weights = weights, wcs = w, psfs = psf_pad, channels = channels)
    
    merian_hdu = fits.open(f'/data/users/dcblanco/Merian/{folder}/{obj_id}/merian/cosmos_{obj_id}_N708.fits')
    merian_data = Data(images = merian_hdu[1].data[np.newaxis, :, :], 
                      weights = merian_hdu[3].data[np.newaxis, :, :], 
                      wcs = wcs.WCS(merian_hdu[1].header), 
                      psfs = [merian_hdu[4].data], channels = 'N')
    
    # start fitting
    model_dict = {}
    
    # fitting in the i-band first: then pass the i-band parameters of target galaxie to other bands
    model_dict[ref_filt] = tractor_hsc_sep_blob_by_blob(
        obj, ref_filt, hsc_data.channels, hsc_data, 
    freeze_dict = {'pos': False, 'shape': False, 'sersicindex': False}, # don't fix shape/sersic
    verbose = False)
    
    for filt in forced_channels: 
        model_dict[filt] = tractor_hsc_sep_blob_by_blob(
        obj, filt, hsc_data.channels, hsc_data, 
        ref_source = model_dict[ref_filt].catalog[model_dict[ref_filt].target_ind], 
        freeze_dict = {'pos': True, 'shape': True, 'sersicindex': True}, # fix shape/sersic, 
        verbose = False)
    
    # fitting for Merian image, translate HSC coordinates into Merian coordinates
    ref_source = copy.deepcopy(model_dict[ref_filt].catalog[model_dict[ref_filt].target_ind]) 
    # makes a copy of the model dict of the ref filt of the target ind
    x, y = merian_data.wcs.wcs_world2pix(*hsc_data.wcs.wcs_pix2world(ref_source.pos.x, ref_source.pos.y, 0), 0) 
    ref_source.pos.x = float(x) 
    ref_source.pos.y = float(y)
    
    try: 
        ref_source.shape.re *= 0.168 / 0.27 # to Merian pixel 
    except: 
        print('Not a sersic!') 
        pass

    try: 
        model_dict['N'] = tractor_hsc_sep_blob_by_blob(
            obj, 'N', merian_data.channels, merian_data, 
            ref_source = ref_source, 
            freeze_dict = {'pos': False, 'shape.ab': True, 'shape.phi': True, 'sersicindex': True}, 
            # since coord system of Merian is different from HSC, don't fix position
            verbose = False)
    except: 
        print('Not a sersic?') 
        pass

    with open(f'cosmos_{obj_id}_sep_tractor.pkl', 'wb') as f: 
        pickle.dump(model_dict, f)

    time_end = time.time()
    print(time_end)
    print(f'Photometry for object {obj_id} complete.')

    printout = str(galaxy_ID) + ' ' + str(time_start) + ' ' + str(time_end) + '\n'
    return printout

def run_mpi(galaxy_ID): 
   
    printout = run_tractor_model(galaxy_ID)
    f = open('/home/dcblanco/Merian_Prl/output_merian_broadcut_tractor.txt', 'a') 
    f.write(printout) 
   

task_list = range(galaxies_number) 

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

for i,task in enumerate(task_list):
    if i%size!=rank: continue
    print("Task number %d (%d) being done by processor %d of %d" % (i, task, rank, size))
    run_mpi(task)
