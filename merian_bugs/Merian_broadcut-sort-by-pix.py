#==================================#
# Sorting Tractor Output by PixPos 
#    [HSC g,r,i,z,y and  N708]
#==================================#

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, Column, QTable
import astropy.units as u

import numpy as np

import os

import pickle

import sys


# astrometry personally hates my laptop, so have to do this every time:
sys.path.append("/home/dcblanco/astrometry.net-0.85/")

# since this is the first time I am ever doing this, let's start with
# one ladybug sample only, and then upgrade to all three

# broadcut obj_cat
obj_cat = Table.read(
    f'/home/dcblanco/Research/Data/broadcut_GAMA09H_cosmos_match.fits'
    ) # broadcut catalog
galaxies_number = len(obj_cat)

#=============================
channel = 'i'
channels = 'grizy'


# to hold indices of all target sources
target_sources_test = [] # not sure why I called it this way

for obj in obj_cat: # one galaxy only
     
    # object ID 
    obj_id = obj['object_id']

    # folder containing galaxy 
    folder = obj['prefix'][40:-49]

    # change directory to tractor output
    os.chdir(f'/data/users/dcblanco/Merian/{folder}/{obj_id}/tractor')

    # update coordinate system
    coord = SkyCoord(obj['ra'], obj['dec'], frame = 'icrs', unit = 'deg')

    # open the cutouts from HSC and Merian
    cutout = [
        fits.open(
            f'/data/groups/leauthaud/Merian/poststamps/cosmos_broad/{folder}/{obj_id}/hsc/cosmos_{obj_id}_{filt}.fits'
        ) for filt in channel
    ]
    
    merian_cutout = [
        fits.open(
            f'/data/users/dcblanco/Merian/{folder}/{obj_id}/merian/cosmos_{obj_id}_N708.fits')
    ]

    hsc_center1 = (cutout[0][1].header['NAXIS1'])/2
    hsc_center2 = (cutout[0][1].header['NAXIS2'])/2
    hsc_center = np.stack((hsc_center1, hsc_center2), axis = -1)

    merian_center1 = (merian_cutout[0][1].header['NAXIS1'])/2
    merian_center2 = (merian_cutout[0][1].header['NAXIS2'])/2
    merian_center = np.stack((merian_center1, merian_center2), axis = -1)
    
    # path of single pickle file, note this contains info on all the filters... 
    pkl_path = f'/data/users/dcblanco/Merian/{folder}/{obj_id}/tractor/cosmos_{obj_id}_sep_tractor.pkl'

    # change path so we're actually where we need to be
    os.chdir(f'/data/users/dcblanco/Merian/{folder}/{obj_id}/tractor/')

    if os.path.exists(pkl_path):
        with open (pkl_path, 'rb') as f: 
            model_dict = pickle.load(f)

        # I know we all hate lists but here we are:
        # this holds the index of the target source... hopefully, LOL
        target_source = []

        # new, model dict with only target sources
        target_model_dict = {}
        
        # all the channels 
        for channel in channels: 
            trac_pos1 = np.array([model_dict[channel].catalog.subs[count].pos[0] for count, source in enumerate(model_dict[channel].catalog)])
            trac_pos2 = np.array([model_dict[channel].catalog.subs[count].pos[1] for count, source in enumerate(model_dict[channel].catalog)])
            trac_pos = np.stack((trac_pos1, trac_pos2), axis = -1)
            target_source = np.argmin([np.sqrt((hsc_center[0]-pos[0])**2 + (hsc_center[1]-pos[1])**2) for ct, pos in enumerate(trac_pos)]) # I am fairly confident this works.. 
            target_model_dict[channel] = model_dict[channel].catalog.subs[target_source]
       
         
        os.chdir(f'/data/users/dcblanco/Merian/{folder}/{obj_id}/tractor/')
        with open(f'cosmos_{obj_id}_sep_tractor_target.pkl', 'wb') as f: 
            pickle.dump(target_model_dict, f)
        
        print(f'Galaxy ID {obj_id} complete.') 
