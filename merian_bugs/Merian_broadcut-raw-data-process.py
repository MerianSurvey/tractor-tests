#===================================#
# Create Merian cutouts using mpi4Py
#===================================#

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
from kuaizi.utils import save_to_fits

from mpi4py import MPI

import pickle

import sep 

import time 

ion()

# from IPython.display import clear_output

obj_cat = Table.read(
    '/home/dcblanco/Research/Data/broadcut_GAMA09H_cosmos_match.fits'
    ) # now ladybug catalog
obj = obj_cat[0]
galaxies_number = len(obj_cat)
print(f'There are {galaxies_number} galaxies.') 

### 
# Image and Weights Map 
###

print('sucker1')
# from fits file
merian_hdu = fits.open('/data/groups/leauthaud/Merian/merian/cosmos_stack/c4d_210307_021334_osj_N708_wide.fits.fz')

print('sucker2')
# wcs 
w_img = wcs.WCS(merian_hdu[1].header)

print('sucker3')
# image
img = merian_hdu[1].data

print('sucker4')
# the zero point
#print('ZP =', merian_hdu[0].header['MAGZERO'])

print('sucker5')
# need this to match HSC zero point (27.0) ! 
img *= 10**((27.0 - merian_hdu[0].header['MAGZERO']) / 2.5)  

print('sucker6')
# representation of spatially variable image background and noise
# ie, just the background!
bkg = sep.Background(img, bw = 256, bh = 256) # bw, bh: size of background boxes in pixels

# display_single(bkg.back()); # evaluate background as 2D array, for display purposes

print('sucker7')
# load weights
weight_hdu = fits.open('/data/groups/leauthaud/Merian/merian/cosmos_stack/c4d_210307_021334_osw_N708_wide.fits.fz')
print('sucker7.1') 
#weight = weight_hdu[1].data
print('sucker7.2') 
w_weight = wcs.WCS(weight_hdu[1].header)

print('sucker8')
# load PSF 
import psfex
pex = psfex.PSFEx('/home/dcblanco/Research/Data/cosmos_deep_init.psf')

###
# Create Cutouts
###

print('sucker8')
from kuaizi.utils import img_cutout
import os
filt = 'N708'
 
def run_create_cutout(galaxy_ID):

    print('Start run for galaxy No.', galaxy_ID) 
    time_start = time.time()

    # object ID
    obj = obj_cat[galaxy_ID]
    obj_id = obj['object_id']    
    
    #print('Prefix =' + obj['prefix'])
    # folder 
    folder = obj['prefix'][40:-49] 
   
    x, y = w_img.wcs_world2pix(obj['ra'], obj['dec'], 0) # coverts coordinates into pixels
    
    # test number = 1 (arbitrarily chosen, primarily to follow the format of prev) 
    if not os.path.isdir(f'/data/users/dcblanco/Merian/{folder}/{obj_id}/merian'): 
        os.makedirs(f'/data/users/dcblanco/Merian/{folder}/{obj_id}/merian')
        
    img_cut, [cen_pos, dx, dy], img_cut_header = img_cutout(
        (img-bkg.globalback), # subtract global background level
        w_img, obj['ra'], obj['dec'], 
        size = [obj['radius'] * 2, obj['radius'] * 2], # NOTE: dynamic cutout size
        pixel_scale = 0.27, 
        save = False,)
    
    weight = weight_hdu[1].data   
    weight_cut, [cen_pos, dx, dy], weight_cut_header = img_cutout(
        weight, w_weight, obj['ra'], obj['dec'], 
        size = [obj['radius'] * 2, obj['radius'] * 2 ], 
        pixel_scale = 0.27, 
        save = False,)
    
    psf_array = pex.get_rec(y, x)
    
    hdu1 = fits.HDUList([
        fits.PrimaryHDU(header = merian_hdu[0].header), # header
        fits.ImageHDU(data = img_cut.data, header = img_cut_header, 
                     name = 'IMAGE'), # image
        fits.ImageHDU(data = None, header = None, 
                     name = 'MASK'), # here, mask is none
        fits.ImageHDU(data = weight_cut.data, header = weight_cut_header, 
                     name = 'WEIGHT'), # weight
        fits.ImageHDU(data = psf_array, name = 'PSF'), # PSF
    ])
    
    fits_file = f'cosmos_{obj_id}_{filt}' + '.fits'
    fits_file = os.path.join(f'/data/users/dcblanco/Merian/{folder}/{obj_id}/merian/', fits_file)
    
    hdu1.writeto(fits_file, overwrite = True)

    time_end = time.time()
    print(f'Photometry for object {obj_id} complete.') 

    printout = str(galaxy_ID) + ' ' + str(time_start) + ' ' + str(time_end) + '\n'

    return printout



def run_mpi(galaxy_ID): 
    print('running mpi') 
    printout = run_create_cutout(galaxy_ID)
    f = open('/home/dcblanco/Merian_Prl/output_merian_broadcut_raw_process.txt', 'a')
    f.write(printout)


task_list = range(galaxies_number)
print(task_list) 

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

for i,task in enumerate(task_list):
    if i%size!=rank: continue
    print("Task number %d (%d) being done by processor %d of %d" % (i, task, rank, size))
    run_mpi(task)
 
