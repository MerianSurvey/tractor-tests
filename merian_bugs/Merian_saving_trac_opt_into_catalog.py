#================================
# Saving `tractor` Output Into 
#          Fits Files! 
# ===============================

# import functions 

from astropy.table import Table, Column, QTable
import astropy.units as u

import numpy as np

import os

import pickle

import sys 

# astrometry personally hates me, so have to do this every time:
sys.path.append("/home/dcblanco/astrometry.net-0.85/")

# broadcut obj_cat
obj_cat = Table.read(
    f'/home/dcblanco/Research/Data/broadcut_GAMA09H_cosmos_match.fits'
    ) # broadcut catalog
galaxies_number = len(obj_cat)

# convert tractor fluxes in nanomaggies into Pogson magnitudes
def mag(flux): 
    return(22.5 - 2.5 * np.log10(flux))

samples = [obj_cat]
tbl_names = ['obj_tbl']
tables = {} # omg a dictionary??

for count, smpl in enumerate(samples): 
    
    # obtain names of ladybugs
    nm = np.array(smpl['name'])

    # create table with names
    tables[f"{tbl_names[count]}".format(tbl_names[count])] = QTable([nm], names = {'name'})
    
    # add new column to table to hold ra 
    col = Column(name='ra', data = np.zeros(len(smpl)))
    tables[f"{tbl_names[count]}".format(tbl_names[count])].add_column(col)
    # tables[f"{tbl_names[count]}".format(tbl_names[count])]['ra'].unit 
    
    # add new column to table to hold dec
    col = Column(name='dec', data = np.zeros(len(smpl)))
    tables[f"{tbl_names[count]}".format(tbl_names[count])].add_column(col)
    # tables[f"{tbl_names[count]}".format(tbl_names[count])]['dec'].unit  

    # add new column to table to hold N708 tractor flux
    col = Column(name='N708_tractor_flux', data = np.zeros(len(smpl)))
    tables[f"{tbl_names[count]}".format(tbl_names[count])].add_column(col)
    tables[f"{tbl_names[count]}".format(tbl_names[count])]['N708_tractor_flux'].unit = u.nanomaggy
    
    # g tractor flux
    col = Column(name='g_tractor_flux', data = np.zeros(len(smpl)))
    tables[f"{tbl_names[count]}".format(tbl_names[count])].add_column(col)
    tables[f"{tbl_names[count]}".format(tbl_names[count])]['g_tractor_flux'].unit = u.nanomaggy
    
    # r tractor flux
    col = Column(name='r_tractor_flux', data = np.zeros(len(smpl)))
    tables[f"{tbl_names[count]}".format(tbl_names[count])].add_column(col)
    tables[f"{tbl_names[count]}".format(tbl_names[count])]['r_tractor_flux'].unit = u.nanomaggy
    
    # i tractor flux
    col = Column(name='i_tractor_flux', data = np.zeros(len(smpl)))
    tables[f"{tbl_names[count]}".format(tbl_names[count])].add_column(col)
    tables[f"{tbl_names[count]}".format(tbl_names[count])]['i_tractor_flux'].unit = u.nanomaggy
    
    # z tractor flux
    col = Column(name='z_tractor_flux', data = np.zeros(len(smpl)))
    tables[f"{tbl_names[count]}".format(tbl_names[count])].add_column(col)
    tables[f"{tbl_names[count]}".format(tbl_names[count])]['z_tractor_flux'].unit = u.nanomaggy
    
    # y tractor flux
    col = Column(name='y_tractor_flux', data = np.zeros(len(smpl)))
    tables[f"{tbl_names[count]}".format(tbl_names[count])].add_column(col)
    tables[f"{tbl_names[count]}".format(tbl_names[count])]['y_tractor_flux'].unit = u.nanomaggy
    
    # add new column to table to hold tractor sersic index
    col = Column(name='tractor_sersic_index', data = np.zeros(len(smpl)))
    tables[f"{tbl_names[count]}".format(tbl_names[count])].add_column(col)
    #tables[f"{tbl_names[count]}".format(tbl_names[count])]['tractor_flux'].unit = u.nanomaggy
    
    # add new column to table to hold tractor re
    col = Column(name='tractor_re', data = np.zeros(len(smpl)))
    tables[f"{tbl_names[count]}".format(tbl_names[count])].add_column(col)
    tables[f"{tbl_names[count]}".format(tbl_names[count])]['tractor_re'].unit = u.arcsec
    
    for obj in smpl: 

        # obtain names of galaxies in each bin
        # object ID
        obj_id = obj['object_id']

        # folder containing galaxy 
        folder = obj['prefix'][40:-49]  
  
        # change directory to tractor output
        pkl_path = f'/data/users/dcblanco/Merian/{folder}/{obj_id}/tractor'
        os.chdir(pkl_path)
        
        grizy_pkl_path = f'cosmos_{obj_id}_sep_tractor_target.pkl'

        N708_pkl_path = f'cosmos_{obj_id}_sep_tractor_N708_target.pkl'

        if os.path.isfile(f'{pkl_path}/{grizy_pkl_path}'):

            if os.path.isfile(grizy_pkl_path):
                ##print('yolo')
                # open pkl file and load model
                with open(grizy_pkl_path, 'rb') as f: 
                    model_dict = pickle.load(f)
            if os.path.isfile(N708_pkl_path):      
                with open(N708_pkl_path, 'rb') as f: 
                    N708_model_dict = pickle.load(f)


                # add ra and dec 
                tables[f"{tbl_names[count]}".format(tbl_names[count])][np.where(tables[f"{tbl_names[count]}".format(tbl_names[count])]['name'] == obj['name'])[0][0]]['ra'] = obj['ra']

                tables[f"{tbl_names[count]}".format(tbl_names[count])][np.where(tables[f"{tbl_names[count]}".format(tbl_names[count])]['name'] == obj['name'])[0][0]]['dec'] = obj['dec']
                
                # add N708 tractor flux to binned table
                try: 
                	tables[f"{tbl_names[count]}".format(tbl_names[count])][np.where(tables[f"{tbl_names[count]}".format(tbl_names[count])]['name'] == obj['name'])[0][0]]['N708_tractor_flux'] = N708_model_dict['N'].brightness.getValue()*u.nanomaggy
                except: 
                    pass

                tables[f"{tbl_names[count]}".format(tbl_names[count])][np.where(tables[f"{tbl_names[count]}".format(tbl_names[count])]['name'] == obj['name'])[0][0]]['g_tractor_flux'] = model_dict['g'].brightness.getValue()*u.nanomaggy
                tables[f"{tbl_names[count]}".format(tbl_names[count])][np.where(tables[f"{tbl_names[count]}".format(tbl_names[count])]['name'] == obj['name'])[0][0]]['r_tractor_flux'] = model_dict['r'].brightness.getValue()*u.nanomaggy
                tables[f"{tbl_names[count]}".format(tbl_names[count])][np.where(tables[f"{tbl_names[count]}".format(tbl_names[count])]['name'] == obj['name'])[0][0]]['i_tractor_flux'] = model_dict['i'].brightness.getValue()*u.nanomaggy
                tables[f"{tbl_names[count]}".format(tbl_names[count])][np.where(tables[f"{tbl_names[count]}".format(tbl_names[count])]['name'] == obj['name'])[0][0]]['z_tractor_flux'] = model_dict['z'].brightness.getValue()*u.nanomaggy
                tables[f"{tbl_names[count]}".format(tbl_names[count])][np.where(tables[f"{tbl_names[count]}".format(tbl_names[count])]['name'] == obj['name'])[0][0]]['y_tractor_flux'] = model_dict['y'].brightness.getValue()*u.nanomaggy

                try: 
                    tables[f"{tbl_names[count]}".format(tbl_names[count])][np.where(tables[f"{tbl_names[count]}".format(tbl_names[count])]['name'] == obj['name'])[0][0]]['tractor_sersic_index'] = N708_model_dict['N'].sersicindex.getValue()
                except: 
                    pass
                try: 
                    tables[f"{tbl_names[count]}".format(tbl_names[count])][np.where(tables[f"{tbl_names[count]}".format(tbl_names[count])]['name'] == obj['name'])[0][0]]['tractor_re'] = N708_model_dict['N'].shape.re*u.arcsec
                except: 
                    pass

                print(f'Galaxy name {obj_id} complete!')
for key in tables['obj_tbl'].keys(): 
    if 'flux' in key: 
        tables['obj_tbl'][key] *= 10**-1.8

print(tables)
os.chdir('/home/dcblanco/Merian_Post_Process') 
tables['obj_tbl'].write('broadcut_trac_opt_target.csv', format='csv', overwrite = True)
